const std = @import("std");

pub const Expert = struct {
    num_neurons: u32,
    gate: []const f32, // Shape: [num_neurons * hidden_dim]
    up: []const f32, // Shape: [num_neurons * hidden_dim]
    down: []const f32, // Shape: [hidden_dim * num_neurons]
};

pub const ZigMoE = struct {
    allocator: std.mem.Allocator,
    memory_pool: []f32,

    num_experts: u32,
    hidden_dim: u32,
    router_weights: []const f32, // Pusat kordinat untuk memilih Pakar
    experts: []Expert,

    pub fn load(allocator: std.mem.Allocator, file_path: []const u8) !ZigMoE {
        var file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();

        // Trik memori rata-4 byte (Sangat Cepat & Aman)
        const f32_count = (file_size + 3) / 4;
        const memory_pool = try allocator.alloc(f32, f32_count);
        errdefer allocator.free(memory_pool);

        const raw_bytes = std.mem.sliceAsBytes(memory_pool)[0..file_size];
        _ = try file.readAll(raw_bytes);
        const aligned_buffer = @as([]align(4) u8, @alignCast(raw_bytes));

        var offset: usize = 0;

        // 1. Cek Magic Header "ZMLP"
        const magic = aligned_buffer[offset .. offset + 4];
        if (!std.mem.eql(u8, magic, "ZMLP")) return error.InvalidFormat;
        offset += 4;

        const num_experts = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const hidden_dim = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
        offset += 4;

        // 2. Baca Bobot Router
        const router_bytes_len = num_experts * hidden_dim * 4;
        const router_slice = @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + router_bytes_len]));
        const router_weights = std.mem.bytesAsSlice(f32, router_slice);
        offset += router_bytes_len;

        var experts = try allocator.alloc(Expert, num_experts);

        // 3. Buka Memori Masing-Masing Pakar
        for (0..num_experts) |i| {
            const num_neurons = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
            offset += 4;

            const matrix_bytes = num_neurons * hidden_dim * 4;

            // Baca Gate
            const gate_slice = @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + matrix_bytes]));
            const gate = std.mem.bytesAsSlice(f32, gate_slice);
            offset += matrix_bytes;

            // Baca Up
            const up_slice = @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + matrix_bytes]));
            const up = std.mem.bytesAsSlice(f32, up_slice);
            offset += matrix_bytes;

            // Baca Down
            const down_slice = @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + matrix_bytes]));
            const down = std.mem.bytesAsSlice(f32, down_slice);
            offset += matrix_bytes;

            experts[i] = Expert{
                .num_neurons = num_neurons,
                .gate = gate,
                .up = up,
                .down = down,
            };
        }

        return ZigMoE{
            .allocator = allocator,
            .memory_pool = memory_pool,
            .num_experts = num_experts,
            .hidden_dim = hidden_dim,
            .router_weights = router_weights,
            .experts = experts,
        };
    }

    pub fn deinit(self: *ZigMoE) void {
        self.allocator.free(self.experts);
        self.allocator.free(self.memory_pool);
    }
};
