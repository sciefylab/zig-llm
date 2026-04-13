// src/inference/test_generation.zig
const std = @import("std");
const brain = @import("../core/brain_reader.zig");
const Tokenizer = @import("../core/tokenizer.zig").ZigTokenizer;
const math = @import("../core/math.zig");

pub fn generate(
    model: *brain.ZigBrain,
    tokenizer: *Tokenizer,
    prompt: []const u8,
    max_tokens: usize,
    temperature: f32, // Tetap di sini agar main.zig tidak error
) !void {
    _ = temperature; // MEMBERITAHU ZIG: "Iya, saya sengaja mengabaikan variabel ini"

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const safe_temp: f32 = 0.3; // Paksakan suhu rendah sementara

    std.debug.print("\n=== ZIG-LLM GENERATION TEST ===\n", .{});
    std.debug.print("Prompt: \"{s}\"\n", .{prompt});
    std.debug.print("Temperature: {d:.2} | Max Tokens: {}\n\n", .{ safe_temp, max_tokens });

    var tokens = try tokenizer.encode(alloc, prompt);

    var timer = try std.time.Timer.start();
    var generated: usize = 0;

    std.debug.print("Generating: ", .{});

    // ZIG MINTA CONST: Karena kita hanya mengubah isinya, bukan panjang array-nya
    const hidden_state = try alloc.alloc(f32, model.hidden_dim);
    const residual = try alloc.alloc(f32, model.hidden_dim);

    // ==========================================
    // ALOKASI MEMORI JANGKA PENDEK (KV-CACHE)
    // ==========================================
    const initial_prompt_len = tokens.len;
    const target_len = initial_prompt_len + max_tokens;
    const max_seq_len = target_len;

    const kv_dim = model.num_kv_heads * model.head_dim;

    // Ini juga jadikan const agar Zig bahagia
    const k_caches = try alloc.alloc([]f32, model.num_layers);
    const v_caches = try alloc.alloc([]f32, model.num_layers);

    for (0..model.num_layers) |l| {
        k_caches[l] = try alloc.alloc(f32, max_seq_len * kv_dim);
        v_caches[l] = try alloc.alloc(f32, max_seq_len * kv_dim);
    }

    var pos: usize = 0;

    // THE AUTOREGRESSIVE LOOP
    while (pos < target_len - 1) {
        const is_prefill = pos < initial_prompt_len - 1;
        const current_token = tokens[pos];

        const embed = model.embed_weights[current_token * model.hidden_dim .. (current_token + 1) * model.hidden_dim];
        @memcpy(hidden_state, embed);

        for (model.layers, 0..) |layer, l| {
            // --- BLOK A: ATTENTION ---
            @memcpy(residual, hidden_state);
            math.rmsNorm(hidden_state, hidden_state, layer.attn_norm, 1e-6);

            try math.computeAttention(alloc, hidden_state, layer.attn, k_caches[l], v_caches[l], pos, max_seq_len, model.num_heads, model.num_kv_heads, model.head_dim);
            math.addVector(hidden_state, residual);

            // --- BLOK B: MIXTURE OF EXPERTS ---
            @memcpy(residual, hidden_state);
            math.rmsNorm(hidden_state, hidden_state, layer.moe_norm, 1e-6);

            const expert_idx = try routeExpert(alloc, hidden_state, layer.router_weights, model.num_moe_experts, model.hidden_dim);
            const expert = layer.experts[expert_idx];

            try math.computeExpertFFN(alloc, hidden_state, expert.gate, expert.up, expert.down, expert.num_neurons, model.hidden_dim);
            math.addVector(hidden_state, residual);
        }

        math.rmsNorm(hidden_state, hidden_state, model.final_norm, 1e-6);

        // --- PREDIKSI KATA ---
        if (!is_prefill) {
            const drawer_id = try sampleDrawer(model, hidden_state, safe_temp);
            const token_id = try sampleToken(model, drawer_id, hidden_state, safe_temp);

            tokens = try appendToken(alloc, tokens, token_id);
            generated += 1;

            const word = decodeToken(tokenizer, token_id);
            std.debug.print("{s}", .{word});

            if (token_id == 0 or token_id == 151643) break;
        }

        pos += 1;
    }

    const time = @as(f32, @floatFromInt(timer.read())) / 1_000_000_000.0;
    std.debug.print("\n\n=== SELESAI ===\n", .{});
    std.debug.print("Generated {d} tokens | Time: {d:.2}s | Speed: {d:.1} T/s\n", .{ generated, time, @as(f32, @floatFromInt(generated)) / time });
}

// ==========================================
// FUNGSI BANTUAN
// ==========================================
fn routeExpert(alloc: std.mem.Allocator, hidden: []f32, router_weights: []const f32, num_experts: usize, hidden_dim: usize) !usize {
    var scores = try alloc.alloc(f32, num_experts);
    defer alloc.free(scores);
    for (0..num_experts) |i| {
        const r_weight = router_weights[i * hidden_dim .. (i + 1) * hidden_dim];
        scores[i] = math.dotProductSIMD(hidden, r_weight);
    }
    var best_expert: usize = 0;
    var max_score: f32 = -std.math.floatMax(f32);
    for (scores, 0..) |s, i| {
        if (s > max_score) {
            max_score = s;
            best_expert = i;
        }
    }
    return best_expert;
}

fn sampleDrawer(model: *brain.ZigBrain, hidden: []const f32, temp: f32) !u32 {
    var scores = try std.heap.page_allocator.alloc(f32, model.num_vocab_clusters);
    defer std.heap.page_allocator.free(scores);
    for (0..model.num_vocab_clusters) |i| {
        const cent = model.vocab_centroids[i * model.hidden_dim .. (i + 1) * model.hidden_dim];
        scores[i] = math.dotProductSIMD(hidden, cent);
    }
    softmax(scores, temp);
    return sampleFromProbs(scores);
}

fn sampleToken(model: *brain.ZigBrain, drawer_id: u32, hidden: []const f32, temp: f32) !u32 {
    const cluster = model.vocab_clusters[drawer_id];
    var scores = try std.heap.page_allocator.alloc(f32, cluster.num_words);
    defer std.heap.page_allocator.free(scores);
    for (0..cluster.num_words) |i| {
        const w = cluster.weights[i * model.hidden_dim .. (i + 1) * model.hidden_dim];
        scores[i] = math.dotProductSIMD(hidden, w);
    }
    softmax(scores, temp);
    const local = sampleFromProbs(scores);
    return cluster.token_ids[local];
}

fn softmax(scores: []f32, temp: f32) void {
    var max_val: f32 = -std.math.floatMax(f32);
    for (scores) |s| max_val = @max(max_val, s);
    var sum: f32 = 0.0;
    for (scores) |*s| {
        s.* = @exp((s.* - max_val) / temp);
        sum += s.*;
    }
    for (scores) |*s| s.* /= sum;
}

fn sampleFromProbs(probs: []f32) u32 {
    const r = std.crypto.random.float(f32);
    var cum: f32 = 0.0;
    for (probs, 0..) |p, i| {
        cum += p;
        if (r <= cum) return @intCast(i);
    }
    return @intCast(probs.len - 1);
}

fn decodeToken(tokenizer: *Tokenizer, token_id: u32) []const u8 {
    if (token_id >= tokenizer.offsets.len - 1) return "<unk>";
    const start = tokenizer.offsets[token_id];
    const end = tokenizer.offsets[token_id + 1];
    return tokenizer.blob[start..end];
}

fn appendToken(alloc: std.mem.Allocator, old: []u32, new_token: u32) ![]u32 {
    const new_tokens = try alloc.alloc(u32, old.len + 1);
    @memcpy(new_tokens[0..old.len], old);
    new_tokens[old.len] = new_token;
    return new_tokens;
}
