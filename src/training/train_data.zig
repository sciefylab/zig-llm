const std = @import("std");

// ==========================================================
// 🚀 CYBER-DUAL BRAIN HMOE DATALOADER
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
    file_buffer: []u8,
    batches: []HMoEBatch,
    position: usize = 0,

    pub fn init(allocator: std.mem.Allocator, bin_path: []const u8) !DualBrainDataloader {
        var file = try std.fs.cwd().openFile(bin_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const buffer = try allocator.alloc(u8, file_size);
        errdefer allocator.free(buffer);

        const bytes_read = try file.readAll(buffer);
        if (bytes_read != file_size) return error.ReadError;

        var count: usize = 0;
        var offset: usize = 0;
        while (offset < file_size) {
            if (offset + 6 > file_size) break;
            const in_len = std.mem.readInt(u16, buffer[offset + 2 .. offset + 4][0..2], .little);
            const tgt_len = std.mem.readInt(u16, buffer[offset + 4 .. offset + 6][0..2], .little);
            offset += 6 + (@as(usize, in_len) * 4) + (@as(usize, tgt_len) * 4);
            count += 1;
        }

        const batches = try allocator.alloc(HMoEBatch, count);
        errdefer allocator.free(batches);

        offset = 0;
        for (0..count) |i| {
            const hemi_val = buffer[offset];
            const exp_val = buffer[offset + 1];
            const in_len = std.mem.readInt(u16, buffer[offset + 2 .. offset + 4][0..2], .little);
            const tgt_len = std.mem.readInt(u16, buffer[offset + 4 .. offset + 6][0..2], .little);
            offset += 6;

            const inputs = try allocator.alloc(u32, in_len);
            for (0..in_len) |j| {
                inputs[j] = std.mem.readInt(u32, buffer[offset + j * 4 .. offset + j * 4 + 4][0..4], .little);
            }
            offset += @as(usize, in_len) * 4;

            const targets = try allocator.alloc(u32, tgt_len);
            for (0..tgt_len) |j| {
                targets[j] = std.mem.readInt(u32, buffer[offset + j * 4 .. offset + j * 4 + 4][0..4], .little);
            }
            offset += @as(usize, tgt_len) * 4;

            batches[i] = HMoEBatch{
                .hemisphere = @enumFromInt(hemi_val),
                .expert = @enumFromInt(exp_val),
                .inputs = inputs,
                .targets = targets,
            };
        }

        return DualBrainDataloader{
            .allocator = allocator,
            .file_buffer = buffer,
            .batches = batches,
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
        self.allocator.free(self.file_buffer);
    }
};
