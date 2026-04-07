const std = @import("std");

// =================================================================
// STRUKTUR DATA ORGAN AI
// =================================================================
pub const Tokenizer = struct {
    offsets: []const u32,
    blob: []const u8,
};

pub const VocabCluster = struct {
    num_words: u32,
    token_ids: []const u32,
    weights: []const f32,
};

pub const Expert = struct {
    num_neurons: u32,
    gate: []const f32,
    up: []const f32,
    down: []const f32,
};

pub const Attention = struct {
    q_proj: []const f32,
    k_proj: []const f32,
    v_proj: []const f32,
    o_proj: []const f32,
};

pub const Layer = struct {
    attn_norm: []const f32, // Katup Atensi (Input Layer Norm)
    moe_norm: []const f32, // Katup MoE (Post Attention Layer Norm)
    attn: Attention,
    router_weights: []const f32,
    experts: []Expert,
};

// =================================================================
// MESIN PEMBACA UTAMA (THE ARCHITECT V2)
// =================================================================
pub const ZigBrain = struct {
    allocator: std.mem.Allocator,
    memory_pool: []f32, // RAM Raksasa 2.35 GB

    // Meta Data
    vocab_size: u32,
    hidden_dim: u32,
    num_layers: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    num_vocab_clusters: u32,
    num_moe_experts: u32,

    // Organ-Organ Global
    tokenizer: Tokenizer,
    vocab_centroids: []const f32,
    vocab_clusters: []VocabCluster,
    embed_weights: []const f32, // Darah (Input)
    final_norm: []const f32, // Katup Akhir (Sebelum Output)
    layers: []Layer, // 24 Lapisan Otak

    pub fn load(allocator: std.mem.Allocator, file_path: []const u8) !ZigBrain {
        var file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();

        const f32_count = (file_size + 3) / 4;
        const memory_pool = try allocator.alloc(f32, f32_count);
        errdefer allocator.free(memory_pool);

        const raw_bytes = std.mem.sliceAsBytes(memory_pool)[0..file_size];
        _ = try file.readAll(raw_bytes);
        const buffer = @as([]align(4) u8, @alignCast(raw_bytes));
        var offset: usize = 0;

        // 1. CEK HEADER GLOBAL
        const magic = buffer[offset .. offset + 4];
        if (!std.mem.eql(u8, magic, "ZBRN")) return error.InvalidFormat;
        offset += 4;

        const version = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
        if (version != 2) return error.UnsupportedVersion;
        offset += 4;

        // 2. BACA META-DATA
        const vocab_size = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const hidden_dim = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const num_layers = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const num_heads = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const num_kv_heads = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const head_dim = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const num_vocab_clusters = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const num_moe_experts = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
        offset += 4;

        // 3. BACA PITA SUARA (TOKENIZER)
        const blob_size = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const offsets_len = (vocab_size + 1) * 4;

        const tok_offsets = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + offsets_len])));
        offset += offsets_len;

        const tok_blob = buffer[offset .. offset + blob_size];
        offset += blob_size;

        offset = (offset + 3) & ~@as(usize, 3); // Padding 4-byte alignment

        // 4. BACA POHON KOSAKATA
        const vocab_cent_len = num_vocab_clusters * hidden_dim * 4;
        const vocab_centroids = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + vocab_cent_len])));
        offset += vocab_cent_len;

        var vocab_clusters = try allocator.alloc(VocabCluster, num_vocab_clusters);
        for (0..num_vocab_clusters) |i| {
            const num_words = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
            offset += 4;

            const ids_len = num_words * 4;
            const token_ids = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + ids_len])));
            offset += ids_len;

            const w_len = num_words * hidden_dim * 4;
            const weights = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + w_len])));
            offset += w_len;

            vocab_clusters[i] = VocabCluster{ .num_words = num_words, .token_ids = token_ids, .weights = weights };
        }

        // 5. BACA EMBEDDINGS & FINAL NORM
        const embed_size = vocab_size * hidden_dim * 4;
        const embed_weights = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + embed_size])));
        offset += embed_size;

        const norm_size = hidden_dim * 4;
        const final_norm = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + norm_size])));
        offset += norm_size;

        // 6. BACA 24 LAPISAN OTAK
        var layers = try allocator.alloc(Layer, num_layers);
        const q_size = num_heads * head_dim * hidden_dim * 4;
        const kv_size = num_kv_heads * head_dim * hidden_dim * 4;
        const o_size = hidden_dim * (num_heads * head_dim) * 4;

        for (0..num_layers) |l| {
            // A. NORMS
            const attn_norm = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + norm_size])));
            offset += norm_size;
            const moe_norm = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + norm_size])));
            offset += norm_size;

            // B. ATTENTION
            const q_proj = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + q_size])));
            offset += q_size;
            const k_proj = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + kv_size])));
            offset += kv_size;
            const v_proj = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + kv_size])));
            offset += kv_size;
            const o_proj = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + o_size])));
            offset += o_size;

            // C. MOE ROUTER
            const r_size = num_moe_experts * hidden_dim * 4;
            const router_weights = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + r_size])));
            offset += r_size;

            // D. MOE EXPERTS
            var experts = try allocator.alloc(Expert, num_moe_experts);
            for (0..num_moe_experts) |e| {
                const num_neurons = std.mem.readInt(u32, buffer[offset .. offset + 4][0..4], .little);
                offset += 4;
                const mat_size = num_neurons * hidden_dim * 4;

                const gate = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + mat_size])));
                offset += mat_size;
                const up = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + mat_size])));
                offset += mat_size;
                // PENTING: Down Project punya ukuran mat_size yang sama, jadi tidak perlu deklarasi ulang!
                const down = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(buffer[offset .. offset + mat_size])));
                offset += mat_size;

                experts[e] = Expert{ .num_neurons = num_neurons, .gate = gate, .up = up, .down = down };
            }

            layers[l] = Layer{
                .attn_norm = attn_norm,
                .moe_norm = moe_norm,
                .attn = Attention{ .q_proj = q_proj, .k_proj = k_proj, .v_proj = v_proj, .o_proj = o_proj },
                .router_weights = router_weights,
                .experts = experts,
            };
        }

        // =========================================================
        // FINAL OFFSET CHECK
        // =========================================================
        if (offset != file_size) {
            std.debug.print("\n[CRITICAL ERROR] Kursor Memori Bergeser!\n", .{});
            std.debug.print("-> Harus membaca : {} Bytes\n", .{file_size});
            std.debug.print("-> Berhenti di   : {} Bytes (Selisih {} Bytes)\n", .{ offset, @as(i64, @intCast(file_size)) - @as(i64, @intCast(offset)) });
            return error.MemoryOffsetDrift;
        }

        return ZigBrain{
            .allocator = allocator,
            .memory_pool = memory_pool,
            .vocab_size = vocab_size,
            .hidden_dim = hidden_dim,
            .num_layers = num_layers,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .num_vocab_clusters = num_vocab_clusters,
            .num_moe_experts = num_moe_experts,
            .tokenizer = Tokenizer{ .offsets = tok_offsets, .blob = tok_blob },
            .vocab_centroids = vocab_centroids,
            .vocab_clusters = vocab_clusters,
            .embed_weights = embed_weights,
            .final_norm = final_norm,
            .layers = layers,
        };
    }

    pub fn deinit(self: *ZigBrain) void {
        for (self.layers) |layer| {
            self.allocator.free(layer.experts);
        }
        self.allocator.free(self.layers);
        self.allocator.free(self.vocab_clusters);
        self.allocator.free(self.memory_pool);
    }
};
