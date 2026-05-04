// train_data.zig
const std = @import("std");
const builtin = @import("builtin");

// ==========================================================
// 🚀 CYBER-DUAL BRAIN HMOE DATALOADER (100% In-Memory)
// ==========================================================
pub const Hemisphere = enum(u8) { left = 0, right = 1 };
pub const Expert = enum(u8) { calculator = 0, syntactician = 1, futurist = 2, storyteller = 3 };

pub const HMoEBatch = struct {
    hemisphere: Hemisphere,
    expert: Expert,
    inputs: []const u32,
    targets: []const u32,
};

const MAX_SEQ_LEN_SAFE: usize = 65535;
const MAX_FILE_SIZE: usize = 2 * 1024 * 1024 * 1024;

pub const DualBrainDataloader = struct {
    allocator: std.mem.Allocator,
    batches: []HMoEBatch,
    position: usize = 0,
    token_buffer: []u32,
    prng: std.Random.DefaultPrng,
    epoch_count: usize = 0,

    // ======================================================
    // INIT — Load seluruh file ke RAM, parse sekali, selesai.
    // Setelah init(), TIDAK ada I/O lagi.
    // ======================================================
    pub fn init(allocator: std.mem.Allocator, bin_path: []const u8) !DualBrainDataloader {
        var file = try std.fs.cwd().openFile(bin_path, .{});
        defer file.close();

        const file_size = (try file.stat()).size;
        if (file_size > MAX_FILE_SIZE) return error.FileTooLarge;
        if (file_size < 8) return error.FileTooSmall;

        const file_data = try file.readToEndAlloc(allocator, MAX_FILE_SIZE);
        defer allocator.free(file_data);

        // ── PASS 1: Hitung total tokens & batches ──
        var total_tokens: usize = 0;
        var total_batches: usize = 0;
        var scan_pos: usize = 0;

        while (scan_pos + 8 <= file_data.len) {
            const hemi_val = file_data[scan_pos];
            const exp_val = file_data[scan_pos + 1];
            const in_len = std.mem.readInt(u16, file_data[scan_pos + 2 ..][0..2], .little);
            const tgt_len = std.mem.readInt(u16, file_data[scan_pos + 4 ..][0..2], .little);

            if (hemi_val > 1 or exp_val > 3) return error.CorruptDataOrMisaligned;
            if (in_len > MAX_SEQ_LEN_SAFE or tgt_len > MAX_SEQ_LEN_SAFE) return error.SeqTooLong;

            const payload_size = (@as(usize, in_len) + @as(usize, tgt_len)) * 4;
            if (scan_pos + 8 + payload_size > file_data.len) return error.TruncatedFile;

            total_tokens += in_len + tgt_len;
            total_batches += 1;
            scan_pos += 8 + payload_size;
        }

        if (total_batches == 0) return error.EmptyDataset;

        // ── Alokasi sekali jalan ──
        const token_buffer = try allocator.alloc(u32, total_tokens);
        errdefer allocator.free(token_buffer);

        const batches = try allocator.alloc(HMoEBatch, total_batches);
        errdefer allocator.free(batches);

        // ── PASS 2: Parse & isi buffer ──
        var read_pos: usize = 0;
        var token_pos: usize = 0;
        var batch_idx: usize = 0;

        while (read_pos + 8 <= file_data.len) {
            const hemi_val = file_data[read_pos];
            const exp_val = file_data[read_pos + 1];
            const in_len = std.mem.readInt(u16, file_data[read_pos + 2 ..][0..2], .little);
            const tgt_len = std.mem.readInt(u16, file_data[read_pos + 4 ..][0..2], .little);
            read_pos += 8;

            // Copy inputs
            const inputs_slice = token_buffer[token_pos..][0..in_len];
            const inputs_bytes_src = file_data[read_pos..][0 .. in_len * 4];
            @memcpy(std.mem.sliceAsBytes(inputs_slice), inputs_bytes_src);
            read_pos += in_len * 4;
            token_pos += in_len;

            // Copy targets
            const targets_slice = token_buffer[token_pos..][0..tgt_len];
            const targets_bytes_src = file_data[read_pos..][0 .. tgt_len * 4];
            @memcpy(std.mem.sliceAsBytes(targets_slice), targets_bytes_src);
            read_pos += tgt_len * 4;
            token_pos += tgt_len;

            batches[batch_idx] = HMoEBatch{
                .hemisphere = @enumFromInt(hemi_val),
                .expert = @enumFromInt(exp_val),
                .inputs = inputs_slice,
                .targets = targets_slice,
            };
            batch_idx += 1;
        }

        // ── Endianness fix (untuk big-endian CPU) ──
        if (builtin.cpu.arch.endian() != .little) {
            for (token_buffer) |*tok| {
                tok.* = @byteSwap(tok.*);
            }
        }

        std.debug.print("✅ Dataset loaded to RAM: {d} batches, {d} tokens\n", .{ total_batches, total_tokens });

        return DualBrainDataloader{
            .allocator = allocator,
            .batches = batches,
            .token_buffer = token_buffer,
            .prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp()))),
            .position = 0,
            .epoch_count = 0,
        };
    }

    // ======================================================
    // EPOCH MANAGEMENT
    // ======================================================

    /// Mulai epoch baru: shuffle seluruh batches & reset posisi.
    /// Panggil ini di awal setiap epoch.
    pub fn startEpoch(self: *DualBrainDataloader) void {
        self.shuffle();
        self.epoch_count += 1;
    }

    /// Reset posisi tanpa shuffle (untuk evaluasi deterministik)
    pub fn reset(self: *DualBrainDataloader) void {
        self.position = 0;
    }

    /// Shuffle batches in-place (Fisher-Yates)
    pub fn shuffle(self: *DualBrainDataloader) void {
        const random = self.prng.random();
        var i: usize = self.batches.len;
        while (i > 1) {
            i -= 1;
            const j = random.uintLessThan(usize, i + 1);
            const tmp = self.batches[i];
            self.batches[i] = self.batches[j];
            self.batches[j] = tmp;
        }
        self.position = 0;
    }

    // ======================================================
    // BATCH ACCESS
    // ======================================================

    /// Ambil batch berikutnya. Return null kalau epoch habis.
    pub fn getNext(self: *DualBrainDataloader) ?HMoEBatch {
        if (self.position >= self.batches.len) return null;
        const batch = self.batches[self.position];
        self.position += 1;
        return batch;
    }

    /// Ambil seluruh slice batches — untuk loop manual atau trainEpoch.
    /// Data tinggal di RAM, tidak ada copy.
    pub fn getAllBatches(self: *const DualBrainDataloader) []const HMoEBatch {
        return self.batches;
    }

    /// Jumlah total batches
    pub fn len(self: *const DualBrainDataloader) usize {
        return self.batches.len;
    }

    /// Apakah epoch sudah selesai?
    pub fn isEpochDone(self: *const DualBrainDataloader) bool {
        return self.position >= self.batches.len;
    }

    /// Epoch keberapa sudah berjalan?
    pub fn epochNumber(self: *const DualBrainDataloader) usize {
        return self.epoch_count;
    }

    // ======================================================
    // STATS
    // ======================================================
    pub fn stats(self: *const DualBrainDataloader) void {
        var left_count: usize = 0;
        var right_count: usize = 0;
        var expert_counts = [_]usize{0} ** 4;
        var total_in_tokens: usize = 0;
        var total_tgt_tokens: usize = 0;

        for (self.batches) |b| {
            if (b.hemisphere == .left) left_count += 1 else right_count += 1;
            expert_counts[@intFromEnum(b.expert)] += 1;
            total_in_tokens += b.inputs.len;
            total_tgt_tokens += b.targets.len;
        }

        std.debug.print("📊 Dataset Stats:\n", .{});
        std.debug.print("   Total batches : {d}\n", .{self.batches.len});
        std.debug.print("   Left / Right  : {d} / {d}\n", .{ left_count, right_count });
        std.debug.print("   Calc/Synt/Fut/Story : {d}/{d}/{d}/{d}\n", .{
            expert_counts[0], expert_counts[1], expert_counts[2], expert_counts[3],
        });
        std.debug.print("   Total tokens  : {d} (in: {d}, tgt: {d})\n", .{
            total_in_tokens + total_tgt_tokens, total_in_tokens, total_tgt_tokens,
        });
    }

    // ======================================================
    // DEALLOC
    // ======================================================
    pub fn deinit(self: *DualBrainDataloader) void {
        self.allocator.free(self.token_buffer);
        self.allocator.free(self.batches);
    }
};
