const std = @import("std");

// Mesin SIMD Bawaan Anda
inline fn dotProductSIMD(a: []const f32, b: []const f32) f32 {
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

// Perkalian Matriks dengan Vektor (GEMV) menggunakan SIMD
fn matVecMul(out: []f32, mat: []const f32, vec: []const f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        const row_data = mat[r * cols .. (r + 1) * cols];
        out[r] = dotProductSIMD(row_data, vec);
    }
}

pub const ZigAttention = struct {
    allocator: std.mem.Allocator,
    memory_pool: []f32,

    hidden_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,

    q_proj: []const f32,
    k_proj: []const f32,
    v_proj: []const f32,
    o_proj: []const f32,

    pub fn load(allocator: std.mem.Allocator, file_path: []const u8) !ZigAttention {
        var file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();
        const file_size = try file.getEndPos();

        const f32_count = (file_size + 3) / 4;
        const memory_pool = try allocator.alloc(f32, f32_count);
        errdefer allocator.free(memory_pool);

        const raw_bytes = std.mem.sliceAsBytes(memory_pool)[0..file_size];
        _ = try file.readAll(raw_bytes);
        const aligned_buffer = @as([]align(4) u8, @alignCast(raw_bytes));
        var offset: usize = 0;

        const magic = aligned_buffer[offset .. offset + 4];
        if (!std.mem.eql(u8, magic, "ZATN")) return error.InvalidFormat;
        offset += 4;

        const hidden_dim = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const num_heads = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const num_kv_heads = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const head_dim = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
        offset += 4;

        const q_bytes = num_heads * head_dim * hidden_dim * 4;
        const q_proj = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + q_bytes])));
        offset += q_bytes;

        const kv_bytes = num_kv_heads * head_dim * hidden_dim * 4;
        const k_proj = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + kv_bytes])));
        offset += kv_bytes;
        const v_proj = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + kv_bytes])));
        offset += kv_bytes;

        const o_bytes = hidden_dim * (num_heads * head_dim) * 4;
        const o_proj = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + o_bytes])));

        return ZigAttention{
            .allocator = allocator,
            .memory_pool = memory_pool,
            .hidden_dim = hidden_dim,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .o_proj = o_proj,
        };
    }

    // MATEMATIKA 1: Rotary Positional Embedding (RoPE)
    // Ini memberi tahu AI "Ini kata ke berapa di dalam kalimat?"
    fn applyRope(vec: []f32, head_dim: u32, pos: usize) void {
        var i: usize = 0;
        while (i < head_dim) : (i += 2) {
            // Qwen menggunakan base 1,000,000 untuk RoPE
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

    // MATEMATIKA 2: Core Attention Forward Pass
    pub fn forward(self: *const ZigAttention, allocator: std.mem.Allocator, hidden_state: []const f32, pos: usize) ![]f32 {
        // 1. Siapkan Wadah Memori (Q, K, V dan Output)
        var q = try allocator.alloc(f32, self.num_heads * self.head_dim);
        defer allocator.free(q);
        var k = try allocator.alloc(f32, self.num_kv_heads * self.head_dim);
        defer allocator.free(k);
        var v = try allocator.alloc(f32, self.num_kv_heads * self.head_dim);
        defer allocator.free(v);
        var attention_out = try allocator.alloc(f32, self.num_heads * self.head_dim);
        defer allocator.free(attention_out);

        // 2. Proyeksi Q, K, V (MatMul)
        matVecMul(q, self.q_proj, hidden_state, self.num_heads * self.head_dim, self.hidden_dim);
        matVecMul(k, self.k_proj, hidden_state, self.num_kv_heads * self.head_dim, self.hidden_dim);
        matVecMul(v, self.v_proj, hidden_state, self.num_kv_heads * self.head_dim, self.hidden_dim);

        // 3. Aplikasikan RoPE ke setiap kepala Q dan K
        for (0..self.num_heads) |h| {
            applyRope(q[h * self.head_dim .. (h + 1) * self.head_dim], self.head_dim, pos);
        }
        for (0..self.num_kv_heads) |h| {
            applyRope(k[h * self.head_dim .. (h + 1) * self.head_dim], self.head_dim, pos);
        }

        // 4. Grouped-Query Attention (GQA) Math
        // Karena ini tes 1 token (tanpa KV Cache konteks sebelumnya), nilai Softmax pasti 1.0!
        // Jadi kita bisa langsung mengkopi nilai V ke output Q yang sesuai.
        const kv_groups = self.num_heads / self.num_kv_heads; // Qwen: 14 / 2 = 7 Kepala Q berbagi 1 Kepala KV

        for (0..self.num_heads) |h| {
            const kv_idx = h / kv_groups; // Cari kepala KV yang cocok untuk kepala Q ini
            const v_head = v[kv_idx * self.head_dim .. (kv_idx + 1) * self.head_dim];

            const out_head = attention_out[h * self.head_dim .. (h + 1) * self.head_dim];
            @memcpy(out_head, v_head); // (Softmax * V)
        }

        // 5. Proyeksi Akhir (Matriks O)
        const final_output = try allocator.alloc(f32, self.hidden_dim);
        @memset(final_output, 0.0);
        matVecMul(final_output, self.o_proj, attention_out, self.hidden_dim, self.num_heads * self.head_dim);

        return final_output;
    }

    pub fn deinit(self: *ZigAttention) void {
        self.allocator.free(self.memory_pool);
    }
};
