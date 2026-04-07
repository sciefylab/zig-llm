const std = @import("std");

pub const KVCache = struct {
    allocator: std.mem.Allocator,
    max_seq_len: usize, // Batas maksimal kata yang bisa diingat (Misal: 1024 kata)
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: u32,

    // Array 3D: [Layer][Posisi_Kata][Data]
    k_cache: [][]f32,
    v_cache: [][]f32,

    pub fn init(allocator: std.mem.Allocator, max_seq_len: usize, num_layers: u32, num_kv_heads: u32, head_dim: u32) !KVCache {
        var k_cache = try allocator.alloc([]f32, num_layers);
        var v_cache = try allocator.alloc([]f32, num_layers);

        // Ukuran memori untuk 1 layer = Maksimal Kata * Jumlah Kepala KV * Dimensi Kepala
        const layer_cache_size = max_seq_len * num_kv_heads * head_dim;

        for (0..num_layers) |l| {
            k_cache[l] = try allocator.alloc(f32, layer_cache_size);
            v_cache[l] = try allocator.alloc(f32, layer_cache_size);
            @memset(k_cache[l], 0.0);
            @memset(v_cache[l], 0.0);
        }

        return KVCache{
            .allocator = allocator,
            .max_seq_len = max_seq_len,
            .num_layers = num_layers,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .k_cache = k_cache,
            .v_cache = v_cache,
        };
    }

    // Fungsi untuk menyimpan ingatan baru ke dalam Cache
    pub fn saveToken(self: *KVCache, layer_idx: usize, pos: usize, k_in: []const f32, v_in: []const f32) void {
        const head_size = self.num_kv_heads * self.head_dim;
        const start_idx = pos * head_size;
        const end_idx = start_idx + head_size;

        // Kopi K dan V baru ke urutan memori ke-[pos]
        @memcpy(self.k_cache[layer_idx][start_idx..end_idx], k_in);
        @memcpy(self.v_cache[layer_idx][start_idx..end_idx], v_in);
    }

    pub fn deinit(self: *KVCache) void {
        for (0..self.num_layers) |l| {
            self.allocator.free(self.k_cache[l]);
            self.allocator.free(self.v_cache[l]);
        }
        self.allocator.free(self.k_cache);
        self.allocator.free(self.v_cache);
    }
};
