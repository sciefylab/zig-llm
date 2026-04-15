const std = @import("std");

// 🔥 UBAH KE 256 DI SINI JIKA INGIN MEMBESARKAN KAPASITAS OTAK AI
const HIDDEN_DIM: usize = 256;
const VOCAB_SIZE: usize = 5000;
const MAX_SEQ_LEN: usize = 64;

pub const DualBrainEngine = struct {
    allocator: std.mem.Allocator,

    mock_embeddings: []f32,
    pos_embeddings: []f32,
    intent_weights: []f32,
    router_l1_weights: []f32,
    router_l2_left_weights: []f32,
    router_l2_right_weights: []f32,
    expert_calc_w: []f32,
    expert_syntax_w: []f32,
    expert_future_w: []f32,
    expert_story_w: []f32,
    lm_head_w: []f32,

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !DualBrainEngine {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const d = HIDDEN_DIM;

        const engine = DualBrainEngine{
            .allocator = allocator,
            .mock_embeddings = try allocator.alloc(f32, VOCAB_SIZE * d),
            .pos_embeddings = try allocator.alloc(f32, MAX_SEQ_LEN * d),
            .intent_weights = try allocator.alloc(f32, d * d),
            .router_l1_weights = try allocator.alloc(f32, d * 2),
            .router_l2_left_weights = try allocator.alloc(f32, d * 2),
            .router_l2_right_weights = try allocator.alloc(f32, d * 2),
            .expert_calc_w = try allocator.alloc(f32, d * d),
            .expert_syntax_w = try allocator.alloc(f32, d * d),
            .expert_future_w = try allocator.alloc(f32, d * d),
            .expert_story_w = try allocator.alloc(f32, d * d),
            .lm_head_w = try allocator.alloc(f32, VOCAB_SIZE * d),
        };

        const weights = [_][]f32{ engine.mock_embeddings, engine.pos_embeddings, engine.intent_weights, engine.router_l1_weights, engine.router_l2_left_weights, engine.router_l2_right_weights, engine.expert_calc_w, engine.expert_syntax_w, engine.expert_future_w, engine.expert_story_w, engine.lm_head_w };
        for (weights) |w| _ = try file.readAll(std.mem.sliceAsBytes(w));

        return engine;
    }

    pub fn deinit(self: *DualBrainEngine) void {
        const weights = [_][]f32{ self.mock_embeddings, self.pos_embeddings, self.intent_weights, self.router_l1_weights, self.router_l2_left_weights, self.router_l2_right_weights, self.expert_calc_w, self.expert_syntax_w, self.expert_future_w, self.expert_story_w, self.lm_head_w };
        for (weights) |w| self.allocator.free(w);
    }

    pub const InferResult = struct {
        id: u32,
        is_right: bool,
        temp: f32,
    };

    pub fn inferFast(self: *DualBrainEngine, pool_sum: []const f32, seq_len: usize) !InferResult {
        const d = HIDDEN_DIM;

        const ts = try self.allocator.alloc(f32, d);
        defer self.allocator.free(ts);

        const scale = 1.0 / @as(f32, @floatFromInt(seq_len));
        @memset(ts, 0.0);
        for (0..d) |out_i| {
            for (0..d) |in_i| ts[out_i] += (pool_sum[in_i] * scale) * self.intent_weights[out_i * d + in_i];
        }

        var l1 = [2]f32{ 0.0, 0.0 };
        for (0..d) |i| {
            l1[0] += ts[i] * self.router_l1_weights[i * 2 + 0];
            l1[1] += ts[i] * self.router_l1_weights[i * 2 + 1];
        }
        const is_dex = l1[1] > l1[0];

        // 🛡️ STABLE SOFTMAX: Trik Matematika Anti-NaN (Exploding Logits)
        const max_l1 = @max(l1[0], l1[1]);
        const exp_l0 = @exp(l1[0] - max_l1);
        const exp_l1 = @exp(l1[1] - max_l1);
        const w_left = exp_l0 / (exp_l0 + exp_l1);
        const w_right = exp_l1 / (exp_l0 + exp_l1);

        const final_temp = (w_left * 0.1) + (w_right * 0.8);

        const l2w = if (is_dex) self.router_l2_right_weights else self.router_l2_left_weights;
        var l2 = [2]f32{ 0.0, 0.0 };
        for (0..d) |i| {
            l2[0] += ts[i] * l2w[i * 2 + 0];
            l2[1] += ts[i] * l2w[i * 2 + 1];
        }
        var exp_w: []f32 = undefined;
        if (!is_dex) {
            exp_w = if (l2[0] > l2[1]) self.expert_calc_w else self.expert_syntax_w;
        } else {
            exp_w = if (l2[0] > l2[1]) self.expert_future_w else self.expert_story_w;
        }

        const out = try self.allocator.alloc(f32, d);
        defer self.allocator.free(out);
        for (0..d) |oi| {
            var sum: f32 = 0.0;
            for (0..d) |ii| sum += ts[ii] * exp_w[oi * d + ii];
            out[oi] = (if (sum > 0.0) sum else 0.0) + ts[oi];
        }

        var mv: f32 = -1e9;
        var best_v: u32 = 0;
        for (0..VOCAB_SIZE) |v| {
            var s: f32 = 0.0;
            for (0..d) |h| s += out[h] * self.lm_head_w[v * d + h];
            s /= final_temp;

            if (s > mv) {
                mv = s;
                best_v = @intCast(v);
            }
        }

        return InferResult{ .id = best_v, .is_right = is_dex, .temp = final_temp };
    }

    pub fn generate(self: *DualBrainEngine, start: []const u32, max: usize, tokenizer: anytype) !void {
        const d = HIDDEN_DIM;

        const pool_sum = try self.allocator.alloc(f32, d);
        defer self.allocator.free(pool_sum);
        @memset(pool_sum, 0.0);

        var current_len: usize = 0;

        for (start) |tid| {
            const sid = @min(tid, VOCAB_SIZE - 1);
            const sp = @min(current_len, MAX_SEQ_LEN - 1);
            for (0..d) |i| pool_sum[i] += self.mock_embeddings[sid * d + i] + self.pos_embeddings[sp * d + i];
            current_len += 1;
        }

        std.debug.print("\n🤖: ", .{});
        var timer = try std.time.Timer.start();
        var generated_count: usize = 0;

        var left_hits: usize = 0;
        var right_hits: usize = 0;
        var avg_temp: f32 = 0.0;

        for (0..max) |_| {
            const result = try self.inferFast(pool_sum, current_len);
            const w = tokenizer.decode(result.id);

            if (std.mem.eql(u8, w, "<|END|>") or std.mem.eql(u8, w, "[UNK]")) break;

            std.debug.print("{s} ", .{w});

            if (result.is_right) right_hits += 1 else left_hits += 1;
            avg_temp += result.temp;

            const sid = @min(result.id, VOCAB_SIZE - 1);
            const sp = @min(current_len, MAX_SEQ_LEN - 1);
            for (0..d) |i| pool_sum[i] += self.mock_embeddings[sid * d + i] + self.pos_embeddings[sp * d + i];

            current_len += 1;
            generated_count += 1;
        }

        const elapsed_ns = timer.read();
        const elapsed_s = @as(f32, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        const tps = @as(f32, @floatFromInt(generated_count)) / @max(elapsed_s, 0.0001);

        if (generated_count > 0) avg_temp /= @as(f32, @floatFromInt(generated_count));
        const dom_rute = if (right_hits > left_hits) "Kanan (Imajinatif)" else "Kiri (Eksak)";

        std.debug.print("\n[📊 Telemetri -> Dominan: {s} | T-Rata: {d:.2}]", .{ dom_rute, avg_temp });
        std.debug.print("\n[⏱️ Benchmark -> {d} token | {d:.4} detik | {d:.2} TPS]\n", .{ generated_count, elapsed_s, tps });
    }
};
