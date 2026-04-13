const std = @import("std");

pub const Cluster = struct {
    num_words: u32,
    token_ids: []const u32,
    weights: []const f32,
};

pub const ZigTreeModel = struct {
    allocator: std.mem.Allocator,
    memory_pool: []f32, // Memori induk untuk dibersihkan nanti

    num_clusters: u32,
    hidden_dim: u32,
    centroids: []const f32,
    clusters: []Cluster,

    pub fn load(allocator: std.mem.Allocator, file_path: []const u8) !ZigTreeModel {
        var file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();

        // Alokasi memori sebagai f32 agar PASTI rata 4-byte (aligned)
        const f32_count = (file_size + 3) / 4;
        const memory_pool = try allocator.alloc(f32, f32_count);
        errdefer allocator.free(memory_pool);

        // Ubah bentuk ke byte untuk membaca file
        const raw_bytes = std.mem.sliceAsBytes(memory_pool)[0..file_size];
        _ = try file.readAll(raw_bytes);

        // Pegang pointer yang sudah dijamin rata 4
        const aligned_buffer = @as([]align(4) u8, @alignCast(raw_bytes));

        var offset: usize = 0;

        // 1. Cek Magic Header "ZTR1"
        const magic = aligned_buffer[offset .. offset + 4];
        if (!std.mem.eql(u8, magic, "ZTR1")) {
            std.debug.print("Error: Bukan format .ztree!\n", .{});
            return error.InvalidFormat;
        }
        offset += 4;

        // 2. Baca Metadata
        const num_clusters = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const hidden_dim = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
        offset += 4;

        // 3. Baca Centroids
        const centroids_bytes = num_clusters * hidden_dim * 4;
        // JURUS HACKER: Paksa Zig percaya bahwa potongan ini rata 4
        const centroids_slice = @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + centroids_bytes]));
        const centroids = std.mem.bytesAsSlice(f32, centroids_slice);
        offset += centroids_bytes;

        var clusters = try allocator.alloc(Cluster, num_clusters);

        // 4. Baca Isi Laci
        for (0..num_clusters) |i| {
            const num_words = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
            offset += 4;

            const token_bytes_len = num_words * 4;
            const token_slice = @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + token_bytes_len]));
            const token_ids = std.mem.bytesAsSlice(u32, token_slice);
            offset += token_bytes_len;

            const weight_bytes_len = num_words * hidden_dim * 4;
            const weight_slice = @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + weight_bytes_len]));
            const weights = std.mem.bytesAsSlice(f32, weight_slice);
            offset += weight_bytes_len;

            clusters[i] = Cluster{
                .num_words = num_words,
                .token_ids = token_ids,
                .weights = weights,
            };
        }

        return ZigTreeModel{
            .allocator = allocator,
            .memory_pool = memory_pool,
            .num_clusters = num_clusters,
            .hidden_dim = hidden_dim,
            .centroids = centroids,
            .clusters = clusters,
        };
    }

    pub fn deinit(self: *ZigTreeModel) void {
        self.allocator.free(self.clusters);
        self.allocator.free(self.memory_pool);
    }
};
