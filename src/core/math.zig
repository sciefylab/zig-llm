const std = @import("std");
const brain = @import("brain_reader.zig"); // <-- TAMBAHKAN BARIS INI

pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (0..a.len) |i| {
        sum += a[i] * b[i];
    }
    return sum;
}

pub fn dotProductSIMD(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    const vec_len: usize = 8;
    var i: usize = 0;
    var vec_sum: @Vector(8, f32) = @splat(0.0);

    while (i + vec_len <= a.len) : (i += vec_len) {
        const va: @Vector(8, f32) = a[i..][0..vec_len].*;
        const vb: @Vector(8, f32) = b[i..][0..vec_len].*;
        vec_sum += va * vb;
    }
    sum += @reduce(.Add, vec_sum);

    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }
    return sum;
}

pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub fn rmsNorm(out: []f32, input: []const f32, weight: []const f32, eps: f32) void {
    var ss: f32 = 0.0;
    for (input) |x| ss += x * x;
    ss = 1.0 / @sqrt((ss / @as(f32, @floatFromInt(input.len))) + eps);
    for (0..input.len) |i| {
        out[i] = weight[i] * (ss * input[i]);
    }
}

// ==========================================
// VECTOR ADDITION (RESIDUAL CONNECTION)
// ==========================================
pub fn addVector(dst: []f32, src: []const f32) void {
    for (dst, src) |*d, s| {
        d.* += s;
    }
}

// ==========================================
// MIXTURE OF EXPERTS FFN (GATE * SILU(UP) * DOWN)
// ==========================================
pub fn computeExpertFFN(alloc: std.mem.Allocator, hidden: []f32, gate: []const f32, up: []const f32, down: []const f32, num_neurons: usize, hidden_dim: usize) !void {
    var gate_out = try alloc.alloc(f32, num_neurons);
    defer alloc.free(gate_out);
    var up_out = try alloc.alloc(f32, num_neurons);
    defer alloc.free(up_out);

    // TAHAP 1: hidden @ Gate & hidden @ Up
    for (0..num_neurons) |i| {
        const gate_row = gate[i * hidden_dim .. (i + 1) * hidden_dim];
        const up_row = up[i * hidden_dim .. (i + 1) * hidden_dim];

        gate_out[i] = dotProductSIMD(hidden, gate_row);
        up_out[i] = dotProductSIMD(hidden, up_row);

        // SiLU(Gate) * Up (Menggunakan fungsi silu milikmu!)
        gate_out[i] = silu(gate_out[i]) * up_out[i];
    }

    // TAHAP 2: intermediate @ Down
    for (0..hidden_dim) |i| {
        var sum: f32 = 0.0;
        for (0..num_neurons) |j| {
            sum += gate_out[j] * down[i * num_neurons + j];
        }
        hidden[i] = sum; // Overwrite hidden state
    }
}

// ==========================================
// SCALED DOT-PRODUCT ATTENTION (WITH GQA & KV-CACHE)
// ==========================================
pub fn computeAttention(
    alloc: std.mem.Allocator,
    hidden: []f32,
    attn: brain.Attention,
    k_cache: []f32,
    v_cache: []f32,
    pos: usize,
    max_seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) !void {
    _ = max_seq_len; // Tidak terpakai langsung, tapi bagus untuk safety check
    const hidden_dim = num_heads * head_dim;

    var q = try alloc.alloc(f32, num_heads * head_dim);
    defer alloc.free(q);
    var k = try alloc.alloc(f32, num_kv_heads * head_dim);
    defer alloc.free(k);
    var v = try alloc.alloc(f32, num_kv_heads * head_dim);
    defer alloc.free(v);

    // 1. QKV Projections (Mengekstrak Query, Key, Value dari Vektor Kata)
    for (0..num_heads * head_dim) |i| {
        q[i] = dotProductSIMD(hidden, attn.q_proj[i * hidden_dim .. (i + 1) * hidden_dim]);
    }
    for (0..num_kv_heads * head_dim) |i| {
        k[i] = dotProductSIMD(hidden, attn.k_proj[i * hidden_dim .. (i + 1) * hidden_dim]);
        v[i] = dotProductSIMD(hidden, attn.v_proj[i * hidden_dim .. (i + 1) * hidden_dim]);
    }

    // 2. Apply RoPE (Matematika Trigonometri agar AI Paham Urutan Kata)
    applyRoPE(q, pos, num_heads, head_dim);
    applyRoPE(k, pos, num_kv_heads, head_dim);

    // 3. Simpan K dan V ke dalam Memori Jangka Pendek (KV-Cache)
    const kv_dim = num_kv_heads * head_dim;
    const cache_offset = pos * kv_dim;
    @memcpy(k_cache[cache_offset .. cache_offset + kv_dim], k);
    @memcpy(v_cache[cache_offset .. cache_offset + kv_dim], v);

    var attn_out = try alloc.alloc(f32, hidden_dim);
    defer alloc.free(attn_out);
    var scores = try alloc.alloc(f32, pos + 1); // Skor relevansi dengan kata-kata sebelumnya
    defer alloc.free(scores);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const kv_groups = num_heads / num_kv_heads;

    // 4. Kalkulasi Attention (Menghubungkan Kata Sekarang dengan Masa Lalu)
    for (0..num_heads) |h| {
        const kv_h = h / kv_groups; // Konsep GQA (Grouped Query Attention) Qwen
        const q_head = q[h * head_dim .. (h + 1) * head_dim];

        // A. Hitung seberapa cocok Query saat ini dengan Key masa lalu (Q * K^T)
        for (0..pos + 1) |t| {
            const k_token_head = k_cache[t * kv_dim + kv_h * head_dim .. t * kv_dim + (kv_h + 1) * head_dim];
            scores[t] = dotProductSIMD(q_head, k_token_head) * scale;
        }

        // B. Softmax (Ubah skor menjadi persentase probabilitas)
        var max_val: f32 = -std.math.floatMax(f32);
        for (0..pos + 1) |t| max_val = @max(max_val, scores[t]);
        var sum: f32 = 0.0;
        for (0..pos + 1) |t| {
            scores[t] = @exp(scores[t] - max_val);
            sum += scores[t];
        }
        for (0..pos + 1) |t| scores[t] /= sum;

        // C. Kalikan persentase kecocokan dengan memori Value masa lalu (Score * V)
        var out_head = attn_out[h * head_dim .. (h + 1) * head_dim];
        @memset(out_head, 0.0);
        for (0..pos + 1) |t| {
            const v_token_head = v_cache[t * kv_dim + kv_h * head_dim .. t * kv_dim + (kv_h + 1) * head_dim];
            for (0..head_dim) |d| {
                out_head[d] += scores[t] * v_token_head[d];
            }
        }
    }

    // 5. O Projection (Finalisasi Pemahaman)
    for (0..hidden_dim) |i| {
        hidden[i] = dotProductSIMD(attn_out, attn.o_proj[i * hidden_dim .. (i + 1) * hidden_dim]);
    }
}

// ==========================================
// ROTARY POSITIONAL EMBEDDING (RoPE)
// ==========================================
fn applyRoPE(vec: []f32, pos: usize, num_heads: usize, head_dim: usize) void {
    const base: f32 = 1000000.0; // Qwen2.5 Base Frequency
    const half_dim = head_dim / 2;

    for (0..num_heads) |h| {
        const head_vec = vec[h * head_dim .. (h + 1) * head_dim];
        for (0..half_dim) |d| {
            const freq = @as(f32, @floatFromInt(pos)) / @exp(@as(f32, @floatFromInt(d * 2)) / @as(f32, @floatFromInt(head_dim)) * @log(base));
            const cos_val = @cos(freq);
            const sin_val = @sin(freq);

            const v0 = head_vec[d];
            const v1 = head_vec[d + half_dim];

            head_vec[d] = v0 * cos_val - v1 * sin_val;
            head_vec[d + half_dim] = v0 * sin_val + v1 * cos_val;
        }
    }
}
