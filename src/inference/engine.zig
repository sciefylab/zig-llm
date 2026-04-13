const std = @import("std");
const brain = @import("../core/brain_reader.zig");
const math = @import("../core/math.zig");
const moe = @import("../core/moe_reader.zig");

pub const InferenceEngine = struct {
    allocator: std.mem.Allocator,
    model: *brain.ZigBrain,

    // KV-Cache: Memori jangka pendek agar AI ingat kata sebelumnya
    k_cache: []f32,
    v_cache: []f32,

    pub fn init(allocator: std.mem.Allocator, model: *brain.ZigBrain) !InferenceEngine {
        const kv_size = model.num_layers * 1024 * (model.num_kv_heads * model.head_dim);
        const k_cache = try allocator.alloc(f32, kv_size);
        const v_cache = try allocator.alloc(f32, kv_size);
        @memset(k_cache, 0.0);
        @memset(v_cache, 0.0);

        return InferenceEngine{
            .allocator = allocator,
            .model = model,
            .k_cache = k_cache,
            .v_cache = v_cache,
        };
    }

    pub fn predictNextToken(self: *InferenceEngine, token_id: u32, pos: usize) !u32 {
        const h_dim = self.model.hidden_dim;
        const x = try self.allocator.alloc(f32, h_dim);
        defer self.allocator.free(x);

        // 1. EMBEDDING LOOKUP (Ambil makna kata dari RAM)
        const embed_offset = @as(usize, token_id) * h_dim;
        @memcpy(x, self.model.embed_weights[embed_offset .. embed_offset + h_dim]);

        // 2. LAYER PROCESSING LOOP
        for (0..self.model.num_layers) |l| {
            const layer = &self.model.layers[l];

            // --- BLOK ATTENTION (Pemahaman Konteks) ---
            const x_norm = try self.allocator.alloc(f32, h_dim);
            defer self.allocator.free(x_norm);

            math.rmsNorm(x_norm, x, layer.attn_norm, 1e-6);

            const layer_kv_offset = l * 1024 * (self.model.num_kv_heads * self.model.head_dim);

            try math.computeAttention(
                self.allocator,
                x_norm,
                layer.attn,
                self.k_cache[layer_kv_offset..],
                self.v_cache[layer_kv_offset..],
                pos,
                1024,
                self.model.num_heads,
                self.model.num_kv_heads,
                self.model.head_dim,
            );

            math.addVector(x, x_norm); // Residual Connection 1

            // --- BLOK HMOE (Cyber-Dual Brain Routing) ---
            const moe_norm = try self.allocator.alloc(f32, h_dim);
            defer self.allocator.free(moe_norm);

            math.rmsNorm(moe_norm, x, layer.moe_norm, 1e-6);

            // Routing Hirarkis: Pilih Pakar -> Eksekusi SIMD FFN
            try moe.forwardHMoE(self.allocator, layer, moe_norm, h_dim);

            math.addVector(x, moe_norm); // Residual Connection 2
        }

        // 3. FINAL NORMALIZATION
        math.rmsNorm(x, x, self.model.final_norm, 1e-6);

        // 4. CLUSTERED VOCAB PREDICTION (Stop computing 150k words!)
        // Prediksi laci kosakata mana yang benar, lalu tebak kata di dalamnya.
        return self.predictFromClusters(x);
    }

    fn predictFromClusters(self: *InferenceEngine, x: []f32) u32 {
        var max_centroid_score: f32 = -1e9;
        var best_cluster_idx: usize = 0;

        // Cari laci (cluster) yang paling relevan
        for (0..self.model.num_vocab_clusters) |i| {
            const score = math.dotProductSIMD(x, self.model.vocab_centroids[i * self.model.hidden_dim .. (i + 1) * self.model.hidden_dim]);
            if (score > max_centroid_score) {
                max_centroid_score = score;
                best_cluster_idx = i;
            }
        }

        // Hanya hitung skor untuk kata-kata di dalam laci terpilih
        const cluster = self.model.vocab_clusters[best_cluster_idx];
        var max_word_score: f32 = -1e9;
        var best_token: u32 = 0;

        for (0..cluster.num_words) |w_idx| {
            const word_weight = cluster.weights[w_idx * self.model.hidden_dim .. (w_idx + 1) * self.model.hidden_dim];
            const score = math.dotProductSIMD(x, word_weight);
            if (score > max_word_score) {
                max_word_score = score;
                best_token = cluster.token_ids[w_idx];
            }
        }

        return best_token;
    }

    pub fn deinit(self: *InferenceEngine) void {
        self.allocator.free(self.k_cache);
        self.allocator.free(self.v_cache);
    }
};
