const std = @import("std");
const train_data = @import("train_data.zig");
const math = @import("../utils/math.zig");

// ==========================================================
// 🔥 CORE CONFIG
// ==========================================================
const HIDDEN_DIM: usize = 1024;
const VOCAB_SIZE: usize = 15000;
const MAX_SEQ_LEN: usize = 128;

// EMA
const EMA_ALPHA: f32 = 0.25;
const EMA_BETA: f32 = 1.0 - EMA_ALPHA;

// GRADIENT CLIPPING
const GRAD_CLIP_VAL: f32 = 1.0;

// ==========================================================
// HELPERS
// ==========================================================
fn allocRandom(allocator: std.mem.Allocator, size: usize, rng: std.Random) ![]f32 {
    const arr = try allocator.alloc(f32, size);
    for (arr) |*v| {
        v.* = rng.float(f32) * 0.1 - 0.05;
    }
    return arr;
}

fn allocZero(allocator: std.mem.Allocator, size: usize) ![]f32 {
    const arr = try allocator.alloc(f32, size);
    @memset(arr, 0.0);
    return arr;
}

fn clipGradient(grad: []f32, max_val: f32) void {
    for (grad) |*g| {
        if (g.* > max_val) g.* = max_val;
        if (g.* < -max_val) g.* = -max_val;
    }
}

pub const DualBrainTrainer = struct {
    allocator: std.mem.Allocator,
    learning_rate: f32,
    prng: std.Random.DefaultPrng,

    // ===== WEIGHTS =====
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

    // ===== PRE-ALLOCATED WORKSPACE =====
    buf_token_state: []f32,
    buf_d_ts: []f32,
    buf_d_pooled: []f32,
    buf_exp_out: []f32,
    buf_expert_sums: []f32,
    buf_v_logits: []f32,
    buf_v_prob: []f32,
    buf_d_exp: []f32,
    buf_d_S: []f32,
    buf_pooled_cache: []f32,
    buf_seq_tokens: []u32,

    // ===== GRADIENT BUFFERS =====
    grad_lm_head: []f32,
    grad_expert_calc_w: []f32,
    grad_expert_syntax_w: []f32,
    grad_expert_future_w: []f32,
    grad_expert_story_w: []f32,
    grad_router_l1_weights: []f32,
    grad_router_l2_left_weights: []f32,
    grad_router_l2_right_weights: []f32,
    grad_intent_weights: []f32,
    grad_mock_embeddings: []f32,
    grad_pos_embeddings: []f32,

    // ======================================================
    // INIT / LOAD / DEALLOC
    // ======================================================
    pub fn init(allocator: std.mem.Allocator, lr: f32) !DualBrainTrainer {
        var seed_prng = std.Random.DefaultPrng.init(42);
        const random = seed_prng.random();
        const d = HIDDEN_DIM;

        return DualBrainTrainer{
            .allocator = allocator,
            .learning_rate = lr,
            .prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp()))),

            .mock_embeddings = try allocRandom(allocator, VOCAB_SIZE * d, random),
            .pos_embeddings = try allocRandom(allocator, MAX_SEQ_LEN * d, random),
            .intent_weights = try allocRandom(allocator, d * d, random),
            .router_l1_weights = try allocRandom(allocator, d * 2, random),
            .router_l2_left_weights = try allocRandom(allocator, d * 2, random),
            .router_l2_right_weights = try allocRandom(allocator, d * 2, random),
            .expert_calc_w = try allocRandom(allocator, d * d, random),
            .expert_syntax_w = try allocRandom(allocator, d * d, random),
            .expert_future_w = try allocRandom(allocator, d * d, random),
            .expert_story_w = try allocRandom(allocator, d * d, random),
            .lm_head_w = try allocRandom(allocator, VOCAB_SIZE * d, random),

            .buf_token_state = try allocZero(allocator, d),
            .buf_d_ts = try allocZero(allocator, d),
            .buf_d_pooled = try allocZero(allocator, d),
            .buf_exp_out = try allocZero(allocator, d),
            .buf_expert_sums = try allocZero(allocator, d),
            .buf_v_logits = try allocZero(allocator, VOCAB_SIZE),
            .buf_v_prob = try allocZero(allocator, VOCAB_SIZE),
            .buf_d_exp = try allocZero(allocator, d),
            .buf_d_S = try allocZero(allocator, d),
            .buf_pooled_cache = try allocZero(allocator, MAX_SEQ_LEN * d),
            .buf_seq_tokens = try allocator.alloc(u32, MAX_SEQ_LEN),

            .grad_lm_head = try allocZero(allocator, VOCAB_SIZE * d),
            .grad_expert_calc_w = try allocZero(allocator, d * d),
            .grad_expert_syntax_w = try allocZero(allocator, d * d),
            .grad_expert_future_w = try allocZero(allocator, d * d),
            .grad_expert_story_w = try allocZero(allocator, d * d),
            .grad_router_l1_weights = try allocZero(allocator, d * 2),
            .grad_router_l2_left_weights = try allocZero(allocator, d * 2),
            .grad_router_l2_right_weights = try allocZero(allocator, d * 2),
            .grad_intent_weights = try allocZero(allocator, d * d),
            .grad_mock_embeddings = try allocZero(allocator, VOCAB_SIZE * d),
            .grad_pos_embeddings = try allocZero(allocator, MAX_SEQ_LEN * d),
        };
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8, lr: f32) !DualBrainTrainer {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const d = HIDDEN_DIM;

        const t = DualBrainTrainer{
            .allocator = allocator,
            .learning_rate = lr,
            .prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp()))),

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

            .buf_token_state = try allocZero(allocator, d),
            .buf_d_ts = try allocZero(allocator, d),
            .buf_d_pooled = try allocZero(allocator, d),
            .buf_exp_out = try allocZero(allocator, d),
            .buf_expert_sums = try allocZero(allocator, d),
            .buf_v_logits = try allocZero(allocator, VOCAB_SIZE),
            .buf_v_prob = try allocZero(allocator, VOCAB_SIZE),
            .buf_d_exp = try allocZero(allocator, d),
            .buf_d_S = try allocZero(allocator, d),
            .buf_pooled_cache = try allocZero(allocator, MAX_SEQ_LEN * d),
            .buf_seq_tokens = try allocator.alloc(u32, MAX_SEQ_LEN),

            .grad_lm_head = try allocZero(allocator, VOCAB_SIZE * d),
            .grad_expert_calc_w = try allocZero(allocator, d * d),
            .grad_expert_syntax_w = try allocZero(allocator, d * d),
            .grad_expert_future_w = try allocZero(allocator, d * d),
            .grad_expert_story_w = try allocZero(allocator, d * d),
            .grad_router_l1_weights = try allocZero(allocator, d * 2),
            .grad_router_l2_left_weights = try allocZero(allocator, d * 2),
            .grad_router_l2_right_weights = try allocZero(allocator, d * 2),
            .grad_intent_weights = try allocZero(allocator, d * d),
            .grad_mock_embeddings = try allocZero(allocator, VOCAB_SIZE * d),
            .grad_pos_embeddings = try allocZero(allocator, MAX_SEQ_LEN * d),
        };

        const weights = [_][]f32{
            t.mock_embeddings,
            t.pos_embeddings,
            t.intent_weights,
            t.router_l1_weights,
            t.router_l2_left_weights,
            t.router_l2_right_weights,
            t.expert_calc_w,
            t.expert_syntax_w,
            t.expert_future_w,
            t.expert_story_w,
            t.lm_head_w,
        };

        for (weights) |w| {
            const bytes = std.mem.sliceAsBytes(w);
            const n = try file.readAll(bytes);
            if (n != bytes.len) return error.UnexpectedEof;
        }
        return t;
    }

    pub fn deinit(self: *DualBrainTrainer) void {
        const weights = [_][]f32{
            self.mock_embeddings,
            self.pos_embeddings,
            self.intent_weights,
            self.router_l1_weights,
            self.router_l2_left_weights,
            self.router_l2_right_weights,
            self.expert_calc_w,
            self.expert_syntax_w,
            self.expert_future_w,
            self.expert_story_w,
            self.lm_head_w,
        };
        for (weights) |w| self.allocator.free(w);

        const buffers = [_][]f32{
            self.buf_token_state,
            self.buf_d_ts,
            self.buf_d_pooled,
            self.buf_exp_out,
            self.buf_expert_sums,
            self.buf_v_logits,
            self.buf_v_prob,
            self.buf_d_exp,
            self.buf_d_S,
            self.buf_pooled_cache,
            self.grad_lm_head,
            self.grad_expert_calc_w,
            self.grad_expert_syntax_w,
            self.grad_expert_future_w,
            self.grad_expert_story_w,
            self.grad_router_l1_weights,
            self.grad_router_l2_left_weights,
            self.grad_router_l2_right_weights,
            self.grad_intent_weights,
            self.grad_mock_embeddings,
            self.grad_pos_embeddings,
        };
        for (buffers) |b| self.allocator.free(b);

        self.allocator.free(self.buf_seq_tokens);
    }

    // ======================================================
    // CONFIG
    // ======================================================
    pub fn setLearningRate(self: *DualBrainTrainer, lr: f32) void {
        self.learning_rate = lr;
    }

    pub fn zeroGradients(self: *DualBrainTrainer) void {
        @memset(self.grad_lm_head, 0.0);
        @memset(self.grad_expert_calc_w, 0.0);
        @memset(self.grad_expert_syntax_w, 0.0);
        @memset(self.grad_expert_future_w, 0.0);
        @memset(self.grad_expert_story_w, 0.0);
        @memset(self.grad_router_l1_weights, 0.0);
        @memset(self.grad_router_l2_left_weights, 0.0);
        @memset(self.grad_router_l2_right_weights, 0.0);
        @memset(self.grad_intent_weights, 0.0);
        @memset(self.grad_mock_embeddings, 0.0);
        @memset(self.grad_pos_embeddings, 0.0);
    }

    pub fn applyGradients(self: *DualBrainTrainer, scale: f32) void {
        clipGradient(self.grad_lm_head, GRAD_CLIP_VAL);
        clipGradient(self.grad_expert_calc_w, GRAD_CLIP_VAL);
        clipGradient(self.grad_expert_syntax_w, GRAD_CLIP_VAL);
        clipGradient(self.grad_expert_future_w, GRAD_CLIP_VAL);
        clipGradient(self.grad_expert_story_w, GRAD_CLIP_VAL);
        clipGradient(self.grad_router_l1_weights, GRAD_CLIP_VAL);
        clipGradient(self.grad_router_l2_left_weights, GRAD_CLIP_VAL);
        clipGradient(self.grad_router_l2_right_weights, GRAD_CLIP_VAL);
        clipGradient(self.grad_intent_weights, GRAD_CLIP_VAL);
        clipGradient(self.grad_mock_embeddings, GRAD_CLIP_VAL);
        clipGradient(self.grad_pos_embeddings, GRAD_CLIP_VAL);

        const s = -scale;

        math.axpy(self.lm_head_w, s, self.grad_lm_head);

        math.axpy(self.expert_calc_w, s, self.grad_expert_calc_w);
        math.axpy(self.expert_syntax_w, s, self.grad_expert_syntax_w);
        math.axpy(self.expert_future_w, s, self.grad_expert_future_w);
        math.axpy(self.expert_story_w, s, self.grad_expert_story_w);

        math.axpy(self.router_l1_weights, s, self.grad_router_l1_weights);
        math.axpy(self.router_l2_left_weights, s, self.grad_router_l2_left_weights);
        math.axpy(self.router_l2_right_weights, s, self.grad_router_l2_right_weights);

        math.axpy(self.intent_weights, s, self.grad_intent_weights);
        math.axpy(self.mock_embeddings, s, self.grad_mock_embeddings);
        math.axpy(self.pos_embeddings, s, self.grad_pos_embeddings);
    }

    // ======================================================
    // TRAIN STEP
    // ======================================================
    pub fn trainStep(self: *DualBrainTrainer, batch: train_data.HMoEBatch) !f32 {
        if (batch.targets.len == 0) return 0.0;

        const d = HIDDEN_DIM;
        const token_state = self.buf_token_state;
        const d_ts = self.buf_d_ts;
        const d_pooled = self.buf_d_pooled;
        const exp_out = self.buf_exp_out;
        const expert_sums = self.buf_expert_sums;
        const v_logits = self.buf_v_logits;
        const v_prob = self.buf_v_prob;
        const d_exp = self.buf_d_exp;
        const d_S = self.buf_d_S;
        const pooled_cache = self.buf_pooled_cache;
        const seq_tokens_full = self.buf_seq_tokens;

        var batch_loss: f32 = 0.0;

        // ===== SELECT ACTIVE PATH =====
        const active_l2_w: []f32 = switch (batch.hemisphere) {
            .left => self.router_l2_left_weights,
            .right => self.router_l2_right_weights,
        };

        const active_l2_grad: []f32 = switch (batch.hemisphere) {
            .left => self.grad_router_l2_left_weights,
            .right => self.grad_router_l2_right_weights,
        };

        const is_exp0 = (batch.expert == .calculator or batch.expert == .futurist);

        const active_expert_w: []f32 = switch (batch.hemisphere) {
            .left => if (is_exp0) self.expert_calc_w else self.expert_syntax_w,
            .right => if (is_exp0) self.expert_future_w else self.expert_story_w,
        };

        const active_expert_grad: []f32 = switch (batch.hemisphere) {
            .left => if (is_exp0) self.grad_expert_calc_w else self.grad_expert_syntax_w,
            .right => if (is_exp0) self.grad_expert_future_w else self.grad_expert_story_w,
        };

        // ===== BUILD FULL TOKEN STREAM =====
        var full_len: usize = 0;

        for (batch.inputs) |tid| {
            if (full_len >= MAX_SEQ_LEN) break;
            seq_tokens_full[full_len] = @min(tid, VOCAB_SIZE - 1);
            full_len += 1;
        }

        const inputs_end = full_len;

        for (batch.targets) |tid| {
            if (full_len >= MAX_SEQ_LEN) break;
            seq_tokens_full[full_len] = @min(tid, VOCAB_SIZE - 1);
            full_len += 1;
        }

        // ===== PRECOMPUTE EMA POOL =====
        for (0..full_len) |seq_i| {
            const sid: usize = @intCast(seq_tokens_full[seq_i]);
            const emb_base = sid * d;
            const pos_base = seq_i * d;
            const out_base = seq_i * d;

            if (seq_i == 0) {
                for (0..d) |i| {
                    pooled_cache[out_base + i] =
                        self.mock_embeddings[emb_base + i] +
                        self.pos_embeddings[pos_base + i];
                }
            } else {
                const prev_base = (seq_i - 1) * d;
                for (0..d) |i| {
                    const emb_val =
                        self.mock_embeddings[emb_base + i] +
                        self.pos_embeddings[pos_base + i];
                    pooled_cache[out_base + i] =
                        (EMA_ALPHA * emb_val) +
                        (EMA_BETA * pooled_cache[prev_base + i]);
                }
            }
        }

        // ===== TRAIN EACH TARGET =====
        for (batch.targets, 0..) |target_token, t_idx| {
            const total_ctx_len = @min(inputs_end + t_idx, MAX_SEQ_LEN);
            if (total_ctx_len == 0) continue;

            const target_idx: usize = @intCast(target_token);
            const pooled = pooled_cache[(total_ctx_len - 1) * d .. total_ctx_len * d];

            // -------------------------
            // Intent Projection
            // -------------------------
            for (0..d) |out_i| {
                token_state[out_i] = math.dot(
                    pooled,
                    self.intent_weights[out_i * d .. out_i * d + d],
                );
            }

            // -------------------------
            // Router L1
            // -------------------------
            var l1_logits = [2]f32{ 0.0, 0.0 };
            for (0..d) |i| {
                l1_logits[0] += token_state[i] * self.router_l1_weights[i * 2 + 0];
                l1_logits[1] += token_state[i] * self.router_l1_weights[i * 2 + 1];
            }

            const max_l1 = @max(l1_logits[0], l1_logits[1]);
            const p_l1 = [2]f32{
                @exp(l1_logits[0] - max_l1),
                @exp(l1_logits[1] - max_l1),
            };
            const sum_l1 = p_l1[0] + p_l1[1];
            var grad_l1_sig = [2]f32{ p_l1[0] / sum_l1, p_l1[1] / sum_l1 };
            grad_l1_sig[@intFromEnum(batch.hemisphere)] -= 1.0;

            // -------------------------
            // Router L2
            // -------------------------
            var l2_logits = [2]f32{ 0.0, 0.0 };
            for (0..d) |i| {
                l2_logits[0] += token_state[i] * active_l2_w[i * 2 + 0];
                l2_logits[1] += token_state[i] * active_l2_w[i * 2 + 1];
            }

            const max_l2 = @max(l2_logits[0], l2_logits[1]);
            const p_l2 = [2]f32{
                @exp(l2_logits[0] - max_l2),
                @exp(l2_logits[1] - max_l2),
            };
            const sum_l2 = p_l2[0] + p_l2[1];
            var grad_l2_sig = [2]f32{ p_l2[0] / sum_l2, p_l2[1] / sum_l2 };
            if (is_exp0) grad_l2_sig[0] -= 1.0 else grad_l2_sig[1] -= 1.0;

            for (0..d) |i| {
                self.grad_router_l1_weights[i * 2 + 0] += token_state[i] * grad_l1_sig[0];
                self.grad_router_l1_weights[i * 2 + 1] += token_state[i] * grad_l1_sig[1];

                active_l2_grad[i * 2 + 0] += token_state[i] * grad_l2_sig[0];
                active_l2_grad[i * 2 + 1] += token_state[i] * grad_l2_sig[1];
            }

            // -------------------------
            // Expert Forward
            // -------------------------
            for (0..d) |oi| {
                const sum = math.dot(
                    token_state,
                    active_expert_w[oi * d .. oi * d + d],
                );
                expert_sums[oi] = sum;
                exp_out[oi] = (if (sum > 0.0) sum else 0.0) + token_state[oi];
            }

            // -------------------------
            // LM HEAD FORWARD (single-thread, SIMD-friendly)
            // -------------------------
            for (0..VOCAB_SIZE) |v| {
                v_logits[v] = math.dot(exp_out, self.lm_head_w[v * d .. v * d + d]);
            }

            // Softmax
            var mv: f32 = -1e30;
            for (0..VOCAB_SIZE) |v| {
                if (v_logits[v] > mv) mv = v_logits[v];
            }

            var vs: f32 = 0.0;
            for (0..VOCAB_SIZE) |v| {
                v_prob[v] = @exp(v_logits[v] - mv);
                vs += v_prob[v];
            }
            const inv_vs = 1.0 / @max(vs, 1e-9);
            for (0..VOCAB_SIZE) |v| v_prob[v] *= inv_vs;

            const prob_target = @max(v_prob[target_idx], 1e-9);
            batch_loss -= @log(prob_target);

            // -------------------------
            // LM HEAD BACKWARD
            // -------------------------
            @memset(d_exp, 0.0);

            for (0..VOCAB_SIZE) |v| {
                // Fix: explicitly cast to f32 to resolve comptime_float runtime error
                const dl = v_prob[v] - @as(f32, if (v == target_idx) 1.0 else 0.0);

                // d_exp += dl * Wv
                math.axpy(d_exp, dl, self.lm_head_w[v * d .. v * d + d]);

                // grad_lm_head[v] += dl * exp_out
                math.axpy(
                    self.grad_lm_head[v * d .. v * d + d],
                    dl,
                    exp_out,
                );
            }

            // -------------------------
            // Expert Backward
            // -------------------------
            @memset(d_ts, 0.0);

            for (0..d) |o| {
                d_ts[o] += d_exp[o]; // residual path

                const dr = if (expert_sums[o] > 0.0) d_exp[o] else 0.0;
                math.axpy(
                    d_ts,
                    dr,
                    active_expert_w[o * d .. o * d + d],
                );
                math.axpy(
                    active_expert_grad[o * d .. o * d + d],
                    dr,
                    token_state,
                );
            }

            // -------------------------
            // Router Backward
            // -------------------------
            for (0..d) |i| {
                d_ts[i] += grad_l1_sig[0] * self.router_l1_weights[i * 2 + 0];
                d_ts[i] += grad_l1_sig[1] * self.router_l1_weights[i * 2 + 1];
                d_ts[i] += grad_l2_sig[0] * active_l2_w[i * 2 + 0];
                d_ts[i] += grad_l2_sig[1] * active_l2_w[i * 2 + 1];
            }

            // -------------------------
            // Intent Backward
            // -------------------------
            @memset(d_pooled, 0.0);
            for (0..d) |out_i| {
                math.axpy(
                    d_pooled,
                    d_ts[out_i],
                    self.intent_weights[out_i * d .. out_i * d + d],
                );
                math.axpy(
                    self.grad_intent_weights[out_i * d .. out_i * d + d],
                    d_ts[out_i],
                    pooled,
                );
            }

            // -------------------------
            // EMA BPTT
            // -------------------------
            std.mem.copyForwards(f32, d_S, d_pooled);

            var seq_i: usize = total_ctx_len;
            while (seq_i > 0) {
                seq_i -= 1;

                const sid: usize = @intCast(seq_tokens_full[seq_i]);

                for (0..d) |i| {
                    const grad_val = if (seq_i == 0) d_S[i] else d_S[i] * EMA_ALPHA;

                    self.grad_mock_embeddings[sid * d + i] += grad_val;
                    self.grad_pos_embeddings[seq_i * d + i] += grad_val;

                    if (seq_i > 0) {
                        d_S[i] *= EMA_BETA;
                    }
                }
            }
        }

        return batch_loss / @as(f32, @floatFromInt(batch.targets.len));
    }

    // ======================================================
    // SAVE CHECKPOINT
    // ======================================================
    pub fn save(self: *DualBrainTrainer, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        const weights = [_][]f32{
            self.mock_embeddings,
            self.pos_embeddings,
            self.intent_weights,
            self.router_l1_weights,
            self.router_l2_left_weights,
            self.router_l2_right_weights,
            self.expert_calc_w,
            self.expert_syntax_w,
            self.expert_future_w,
            self.expert_story_w,
            self.lm_head_w,
        };

        for (weights) |w| {
            try file.writeAll(std.mem.sliceAsBytes(w));
        }
    }
};
