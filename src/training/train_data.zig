const std = @import("std");

// ==========================================================
// 🚀 CYBER-DUAL BRAIN HMOE DATALOADER (Memory Stream)
// ==========================================================
pub const Hemisphere = enum(u8) { left = 0, right = 1 };
pub const Expert = enum(u8) { calculator = 0, syntactician = 1, futurist = 2, storyteller = 3 };

pub const HMoEBatch = struct {
    hemisphere: Hemisphere,
    expert: Expert,
    inputs: []u32,
    targets: []u32,
};

pub const DualBrainDataloader = struct {
    allocator: std.mem.Allocator,
    batches: []HMoEBatch,
    position: usize = 0,

    pub fn init(allocator: std.mem.Allocator, bin_path: []const u8) !DualBrainDataloader {
        var file = try std.fs.cwd().openFile(bin_path, .{});
        defer file.close();

        // 🔥 FIX ZIG 0.15.2: Baca seluruh file ke memori sekaligus (Sangat cepat & Anti-Error)
        // Kita alokasikan maksimal 500MB untuk berjaga-jaga
        const file_data = try file.readToEndAlloc(allocator, 500 * 1024 * 1024);
        defer allocator.free(file_data); // Langsung dibebaskan setelah parsing selesai

        // Buat stream reader virtual dari memori RAM
        var fbs = std.io.fixedBufferStream(file_data);
        var reader = fbs.reader();

        var batch_list: std.ArrayList(HMoEBatch) = .empty;
        errdefer {
            for (batch_list.items) |b| {
                allocator.free(b.inputs);
                allocator.free(b.targets);
            }
            batch_list.deinit(allocator);
        }

        while (true) {
            // 1. BACA HEADER (Skema 8-Byte)
            const hemi_val = reader.readByte() catch |err| switch (err) {
                error.EndOfStream => break, // File selesai, keluar dari loop
                else => return err,
            };
            const exp_val = try reader.readByte();
            const in_len = try reader.readInt(u16, .little);
            const tgt_len = try reader.readInt(u16, .little);
            _ = try reader.readInt(u16, .little); // Buang 2 byte padding/reserved

            // 2. VALIDASI KEAMANAN
            if (hemi_val > 1 or exp_val > 3) return error.CorruptDataOrMisaligned;

            // 3. BACA PAYLOAD INPUTS
            const inputs = try allocator.alloc(u32, in_len);
            for (0..in_len) |j| inputs[j] = try reader.readInt(u32, .little);

            // 4. BACA PAYLOAD TARGETS
            const targets = try allocator.alloc(u32, tgt_len);
            for (0..tgt_len) |j| targets[j] = try reader.readInt(u32, .little);

            // 5. SIMPAN KE BATCH
            try batch_list.append(allocator, HMoEBatch{
                .hemisphere = @enumFromInt(hemi_val),
                .expert = @enumFromInt(exp_val),
                .inputs = inputs,
                .targets = targets,
            });
        }

        return DualBrainDataloader{
            .allocator = allocator,
            // Konversi ke static slice (Oper Allocator)
            .batches = try batch_list.toOwnedSlice(allocator),
        };
    }

    pub fn reset(self: *DualBrainDataloader) void {
        self.position = 0;
    }

    pub fn getNext(self: *DualBrainDataloader) ?HMoEBatch {
        if (self.position >= self.batches.len) return null;
        const batch = self.batches[self.position];
        self.position += 1;
        return batch;
    }

    pub fn deinit(self: *DualBrainDataloader) void {
        for (self.batches) |batch| {
            self.allocator.free(batch.inputs);
            self.allocator.free(batch.targets);
        }
        self.allocator.free(self.batches);
    }
};
