const std = @import("std");
const brain = @import("brain_reader.zig");
const mem = @import("kv_cache.zig"); // <-- Ini yang tadi hilang!

// ==========================================
// 1. MESIN MATEMATIKA HARDWARE (SIMD)
// ==========================================
pub fn dotProductSIMD(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    const vec_len = 8;
    var i: usize = 0;
    var vec_sum: @Vector(vec_len, f32) = @splat(0.0);

    while (i + vec_len <= a.len) : (i += vec_len) {
        const va: @Vector(vec_len, f32) = a[i..][0..vec_len].*;
        const vb: @Vector(vec_len, f32) = b[i..][0..vec_len].*;
        vec_sum += va * vb;
    }
    sum += @reduce(.Add, vec_sum);
    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }
    return sum;
}

pub fn silu(x: f32) f32 {
    const sigmoid = 1.0 / (1.0 + @exp(-x));
    return x * sigmoid;
}

pub fn rmsNorm(out: []f32, in: []const f32, weight: []const f32, eps: f32) void {
    var ss: f32 = 0.0;
    for (in) |x| {
        ss += x * x;
    }
    ss /= @as(f32, @floatFromInt(in.len));
    ss += eps;
    ss = 1.0 / @sqrt(ss);

    for (0..in.len) |i| {
        out[i] = weight[i] * (ss * in[i]);
    }
}

fn matVecMul(out: []f32, mat: []const f32, vec: []const f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        const row_data = mat[r * cols .. (r + 1) * cols];
        out[r] = dotProductSIMD(row_data, vec);
    }
}

fn applyRope(vec: []f32, head_dim: u32, pos: usize) void {
    var i: usize = 0;
    while (i < head_dim) : (i += 2) {
        const freq = 1.0 / std.math.pow(f32, 1000000.0, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
        const val = @as(f32, @floatFromInt(pos)) * freq;
        const fcr = @cos(val);
        const fci = @sin(val);
        const v0 = vec[i];
        const v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

// ==========================================
// 2. FUNGSI EKSEKUSI ORGAN
// ==========================================

pub fn forwardAttention(allocator: std.mem.Allocator, attn: brain.Attention, cache: *mem.KVCache, layer_idx: usize, hidden_state: []const f32, pos: usize, num_heads: u32, num_kv_heads: u32, head_dim: u32, hidden_dim: u32) ![]f32 {
    const q = try allocator.alloc(f32, num_heads * head_dim);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, num_kv_heads * head_dim);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, num_kv_heads * head_dim);
    defer allocator.free(v);
    const attention_out = try allocator.alloc(f32, num_heads * head_dim);
    defer allocator.free(attention_out);

    matVecMul(q, attn.q_proj, hidden_state, num_heads * head_dim, hidden_dim);
    matVecMul(k, attn.k_proj, hidden_state, num_kv_heads * head_dim, hidden_dim);
    matVecMul(v, attn.v_proj, hidden_state, num_kv_heads * head_dim, hidden_dim);

    for (0..num_heads) |h| {
        applyRope(q[h * head_dim .. (h + 1) * head_dim], head_dim, pos);
    }
    for (0..num_kv_heads) |h| {
        applyRope(k[h * head_dim .. (h + 1) * head_dim], head_dim, pos);
    }

    // Simpan ke Ingatan AI (RAM)
    cache.saveToken(layer_idx, pos, k, v);

    const kv_groups = num_heads / num_kv_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    var scores = try allocator.alloc(f32, pos + 1);
    defer allocator.free(scores);

    // MENGHITUNG KECOCOKAN KATA INI DENGAN KATA-KATA SEBELUMNYA (SOFTMAX)
    for (0..num_heads) |h| {
        const q_head = q[h * head_dim .. (h + 1) * head_dim];
        const kv_idx = h / kv_groups;
        const head_offset = kv_idx * head_dim;

        for (0..pos + 1) |t| {
            const k_past = cache.k_cache[layer_idx][t * (num_kv_heads * head_dim) + head_offset .. t * (num_kv_heads * head_dim) + head_offset + head_dim];
            scores[t] = dotProductSIMD(q_head, k_past) * scale;
        }

        var max_score: f32 = -999999.0;
        for (0..pos + 1) |t| {
            if (scores[t] > max_score) max_score = scores[t];
        }

        var sum_exp: f32 = 0.0;
        for (0..pos + 1) |t| {
            scores[t] = @exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        for (0..pos + 1) |t| {
            scores[t] /= sum_exp;
        }

        const out_head = attention_out[h * head_dim .. (h + 1) * head_dim];
        @memset(out_head, 0.0);

        for (0..pos + 1) |t| {
            const v_past = cache.v_cache[layer_idx][t * (num_kv_heads * head_dim) + head_offset .. t * (num_kv_heads * head_dim) + head_offset + head_dim];
            const weight = scores[t];
            for (0..head_dim) |d| {
                out_head[d] += weight * v_past[d];
            }
        }
    }

    const final_output = try allocator.alloc(f32, hidden_dim);
    @memset(final_output, 0.0);
    matVecMul(final_output, attn.o_proj, attention_out, hidden_dim, num_heads * head_dim);
    return final_output;
}

pub fn forwardExpert(allocator: std.mem.Allocator, expert: brain.Expert, hidden_state: []const f32, hidden_dim: u32) ![]f32 {
    const intermediate = try allocator.alloc(f32, expert.num_neurons);
    defer allocator.free(intermediate);
    const output = try allocator.alloc(f32, hidden_dim);
    @memset(output, 0.0);

    for (0..expert.num_neurons) |n| {
        const w_gate = expert.gate[n * hidden_dim .. (n + 1) * hidden_dim];
        const w_up = expert.up[n * hidden_dim .. (n + 1) * hidden_dim];
        intermediate[n] = silu(dotProductSIMD(hidden_state, w_gate)) * dotProductSIMD(hidden_state, w_up);
    }
    for (0..expert.num_neurons) |n| {
        const neuron_val = intermediate[n];
        const w_down = expert.down[n * hidden_dim .. (n + 1) * hidden_dim];
        for (0..hidden_dim) |d| {
            output[d] += neuron_val * w_down[d];
        }
    }
    return output;
}

// ==========================================
// 3. THE AUTOREGRESSIVE LOOP (AI BERBICARA)
// ==========================================
pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("=========================================================\n", .{});
    std.debug.print("      ZIG-LLM: THE FINAL INTELLIGENCE LOOP               \n", .{});
    std.debug.print("=========================================================\n", .{});

    var ai = try brain.ZigBrain.load(allocator, "models/qwen_0.5b_moe.zbrain");
    defer ai.deinit();

    // Nyalakan Sistem Ingatan (KV Cache)
    const max_ingatan: usize = 100;
    var cache = try mem.KVCache.init(allocator, max_ingatan, ai.num_layers, ai.num_kv_heads, ai.head_dim);
    defer cache.deinit();

    var current_state = try allocator.alloc(f32, ai.hidden_dim);
    const temp_state = try allocator.alloc(f32, ai.hidden_dim);

    // INPUT MANUSIA
    var input_token_id: u32 = 1237; // "stream"
    const w_start_input = ai.tokenizer.offsets[input_token_id];
    const w_end_input = ai.tokenizer.offsets[input_token_id + 1];

    std.debug.print("\n[MANUSIA] : '{s}'\n", .{ai.tokenizer.blob[w_start_input..w_end_input]});
    std.debug.print("[AI]      : ", .{});

    var total_timer = try std.time.Timer.start();

    // LOOP GENERASI (AI MENGHASILKAN 10 KATA DENGAN INGATAN)
    for (0..10) |posisi_kata| {

        // A. Ambil Darah (Embeddings)
        const embed = ai.embed_weights[input_token_id * ai.hidden_dim .. (input_token_id + 1) * ai.hidden_dim];
        @memcpy(current_state, embed);

        // B. Melewati 24 Lapisan Atensi & MoE
        for (0..ai.num_layers) |l| {
            const layer = ai.layers[l];

            // --- JALUR MATA (DENGAN INGATAN CACHE) ---
            rmsNorm(temp_state, current_state, layer.attn_norm, 1e-6);
            const attn_out = try forwardAttention(allocator, layer.attn, &cache, l, temp_state, posisi_kata, ai.num_heads, ai.num_kv_heads, ai.head_dim, ai.hidden_dim);
            for (0..ai.hidden_dim) |d| {
                current_state[d] += attn_out[d];
            }
            allocator.free(attn_out);

            // --- JALUR OTAK (MoE ROUTER) ---
            rmsNorm(temp_state, current_state, layer.moe_norm, 1e-6);

            var best_expert: usize = 0;
            var max_score: f32 = -999999.0;
            for (0..ai.num_moe_experts) |e| {
                const centroid = layer.router_weights[e * ai.hidden_dim .. (e + 1) * ai.hidden_dim];
                const score = dotProductSIMD(temp_state, centroid);
                if (score > max_score) {
                    max_score = score;
                    best_expert = e;
                }
            }

            const moe_out = try forwardExpert(allocator, layer.experts[best_expert], temp_state, ai.hidden_dim);
            for (0..ai.hidden_dim) |d| {
                current_state[d] += moe_out[d];
            }
            allocator.free(moe_out);
        }

        // C. Katup Akhir
        rmsNorm(current_state, current_state, ai.final_norm, 1e-6);

        // D. Pohon Kosakata (Pencarian Kata)
        var best_laci: usize = 0;
        var max_laci_score: f32 = -999999.0;
        for (0..ai.num_vocab_clusters) |i| {
            const centroid = ai.vocab_centroids[i * ai.hidden_dim .. (i + 1) * ai.hidden_dim];
            const score = dotProductSIMD(current_state, centroid);
            if (score > max_laci_score) {
                max_laci_score = score;
                best_laci = i;
            }
        }

        const winning_cluster = ai.vocab_clusters[best_laci];
        var best_token: u32 = 0;
        var max_token_score: f32 = -999999.0;
        for (0..winning_cluster.num_words) |j| {
            const word_weight = winning_cluster.weights[j * ai.hidden_dim .. (j + 1) * ai.hidden_dim];
            const score = dotProductSIMD(current_state, word_weight);
            if (score > max_token_score) {
                max_token_score = score;
                best_token = winning_cluster.token_ids[j];
            }
        }

        // E. Cetak hasil kata langsung ke layar
        const w_start = ai.tokenizer.offsets[best_token];
        const w_end = ai.tokenizer.offsets[best_token + 1];
        std.debug.print("{s}", .{ai.tokenizer.blob[w_start..w_end]});

        // F. Umpankan kembali ke input untuk putaran selanjutnya
        input_token_id = best_token;
    }

    const total_ms = @as(f32, @floatFromInt(total_timer.read() / std.time.ns_per_us)) / 1000.0;
    const tps = 10.0 / (total_ms / 1000.0);

    std.debug.print("\n\n=========================================================\n", .{});
    std.debug.print(" DIAGNOSTIK KINERJA FINAL (DENGAN INGATAN PENUH):\n", .{});
    std.debug.print(" - Menghasilkan         : 10 Kata (Token)\n", .{});
    std.debug.print(" - Total Waktu          : {d:.3} Milidetik\n", .{total_ms});
    std.debug.print(" - Kecepatan (Token/Sec): {d:.1} T/s\n", .{tps});
    std.debug.print("=========================================================\n", .{});
}
