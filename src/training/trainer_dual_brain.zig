const std = @import("std");
const train_data = @import("train_data.zig");

// 🔥 UBAH KE 256 DI SINI JIKA INGIN MEMBESARKAN KAPASITAS OTAK AI
const HIDDEN_DIM: usize = 256;
const VOCAB_SIZE: usize = 5000;
const MAX_SEQ_LEN: usize = 64;

pub const DualBrainTrainer = struct {
    allocator: std.mem.Allocator,
    learning_rate: f32,
    prng: std.Random.DefaultPrng,

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

    pub fn init(allocator: std.mem.Allocator, lr: f32) !DualBrainTrainer {
        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();
        const init_w = struct {
            fn call(alloc: std.mem.Allocator, size: usize, r: std.Random) ![]f32 {
                const arr = try alloc.alloc(f32, size);
                for (0..size) |i| arr[i] = r.float(f32) * 0.1 - 0.05;
                return arr;
            }
        }.call;

        return DualBrainTrainer{
            .allocator = allocator,
            .learning_rate = lr,
            .prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp()))),
            .mock_embeddings = try init_w(allocator, VOCAB_SIZE * HIDDEN_DIM, random),
            .pos_embeddings = try init_w(allocator, MAX_SEQ_LEN * HIDDEN_DIM, random),
            .intent_weights = try init_w(allocator, HIDDEN_DIM * HIDDEN_DIM, random),
            .router_l1_weights = try init_w(allocator, HIDDEN_DIM * 2, random),
            .router_l2_left_weights = try init_w(allocator, HIDDEN_DIM * 2, random),
            .router_l2_right_weights = try init_w(allocator, HIDDEN_DIM * 2, random),
            .expert_calc_w = try init_w(allocator, HIDDEN_DIM * HIDDEN_DIM, random),
            .expert_syntax_w = try init_w(allocator, HIDDEN_DIM * HIDDEN_DIM, random),
            .expert_future_w = try init_w(allocator, HIDDEN_DIM * HIDDEN_DIM, random),
            .expert_story_w = try init_w(allocator, HIDDEN_DIM * HIDDEN_DIM, random),
            .lm_head_w = try init_w(allocator, VOCAB_SIZE * HIDDEN_DIM, random),
        };
    }

    pub fn computeIntentContext(self: *DualBrainTrainer, ctx_tokens: []const u32) ![]f32 {
        const seq_len = ctx_tokens.len;
        const d = HIDDEN_DIM;

        const pooled = try self.allocator.alloc(f32, d);
        defer self.allocator.free(pooled);
        @memset(pooled, 0.0);

        if (seq_len == 0) {
            const final_out = try self.allocator.alloc(f32, d);
            @memset(final_out, 0.0);
            return final_out;
        }

        for (ctx_tokens, 0..) |tid, pos| {
            const sid = @min(tid, VOCAB_SIZE - 1);
            const sp = @min(pos, MAX_SEQ_LEN - 1);
            for (0..d) |i| pooled[i] += self.mock_embeddings[sid * d + i] + self.pos_embeddings[sp * d + i];
        }

        const scale = 1.0 / @as(f32, @floatFromInt(seq_len));
        for (0..d) |i| pooled[i] *= scale;

        const final_out = try self.allocator.alloc(f32, d);
        @memset(final_out, 0.0);
        for (0..d) |out_i| {
            for (0..d) |in_i| final_out[out_i] += pooled[in_i] * self.intent_weights[out_i * d + in_i];
        }

        return final_out;
    }

    pub fn trainStep(self: *DualBrainTrainer, batch: train_data.HMoEBatch) !f32 {
        if (batch.targets.len == 0) return 0.0;
        const d = HIDDEN_DIM;

        const pooled = try self.allocator.alloc(f32, d);
        defer self.allocator.free(pooled);
        const token_state = try self.allocator.alloc(f32, d);
        defer self.allocator.free(token_state);

        const d_ts = try self.allocator.alloc(f32, d);
        defer self.allocator.free(d_ts);
        const d_pooled = try self.allocator.alloc(f32, d);
        defer self.allocator.free(d_pooled);

        const exp_out = try self.allocator.alloc(f32, d);
        defer self.allocator.free(exp_out);
        const v_logits = try self.allocator.alloc(f32, VOCAB_SIZE);
        defer self.allocator.free(v_logits);
        const v_prob = try self.allocator.alloc(f32, VOCAB_SIZE);
        defer self.allocator.free(v_prob);
        const d_exp = try self.allocator.alloc(f32, d);
        defer self.allocator.free(d_exp);

        var batch_loss: f32 = 0.0;

        for (batch.targets, 0..) |target_token, t_idx| {
            const total_ctx_len = @min(batch.inputs.len + t_idx, MAX_SEQ_LEN);

            @memset(pooled, 0.0);
            var pos_iter: usize = 0;

            for (batch.inputs) |tid| {
                if (pos_iter >= MAX_SEQ_LEN) break;
                const sid = @min(tid, VOCAB_SIZE - 1);
                for (0..d) |i| pooled[i] += self.mock_embeddings[sid * d + i] + self.pos_embeddings[pos_iter * d + i];
                pos_iter += 1;
            }
            for (batch.targets[0..t_idx]) |tid| {
                if (pos_iter >= MAX_SEQ_LEN) break;
                const sid = @min(tid, VOCAB_SIZE - 1);
                for (0..d) |i| pooled[i] += self.mock_embeddings[sid * d + i] + self.pos_embeddings[pos_iter * d + i];
                pos_iter += 1;
            }

            const scale = 1.0 / @as(f32, @floatFromInt(total_ctx_len));
            for (0..d) |i| pooled[i] *= scale;

            @memset(token_state, 0.0);
            for (0..d) |out_i| {
                for (0..d) |in_i| token_state[out_i] += pooled[in_i] * self.intent_weights[out_i * d + in_i];
            }

            var l1_logits = [2]f32{ 0.0, 0.0 };
            for (0..d) |i| {
                l1_logits[0] += token_state[i] * self.router_l1_weights[i * 2 + 0];
                l1_logits[1] += token_state[i] * self.router_l1_weights[i * 2 + 1];
            }

            // Logika Softmax di sini sudah dilindungi @max sejak awal (Aman dari NaN)
            const max_l1 = @max(l1_logits[0], l1_logits[1]);
            const p_l1 = [2]f32{ @exp(l1_logits[0] - max_l1), @exp(l1_logits[1] - max_l1) };
            const sum_l1 = p_l1[0] + p_l1[1];
            var grad_l1 = [2]f32{ p_l1[0] / sum_l1, p_l1[1] / sum_l1 };
            grad_l1[@intFromEnum(batch.hemisphere)] -= 1.0;

            var active_l2_w: []f32 = undefined;
            var active_expert_w: []f32 = undefined;
            const is_exp0 = (batch.expert == .calculator or batch.expert == .futurist);
            if (batch.hemisphere == .left) {
                active_l2_w = self.router_l2_left_weights;
                active_expert_w = if (is_exp0) self.expert_calc_w else self.expert_syntax_w;
            } else {
                active_l2_w = self.router_l2_right_weights;
                active_expert_w = if (is_exp0) self.expert_future_w else self.expert_story_w;
            }

            var l2_logits = [2]f32{ 0.0, 0.0 };
            for (0..d) |i| {
                l2_logits[0] += token_state[i] * active_l2_w[i * 2 + 0];
                l2_logits[1] += token_state[i] * active_l2_w[i * 2 + 1];
            }
            const max_l2 = @max(l2_logits[0], l2_logits[1]);
            const p_l2 = [2]f32{ @exp(l2_logits[0] - max_l2), @exp(l2_logits[1] - max_l2) };
            const sum_l2 = p_l2[0] + p_l2[1];
            var grad_l2 = [2]f32{ p_l2[0] / sum_l2, p_l2[1] / sum_l2 };
            if (is_exp0) grad_l2[0] -= 1.0 else grad_l2[1] -= 1.0;

            for (0..d) |i| {
                self.router_l1_weights[i * 2 + 0] -= self.learning_rate * token_state[i] * grad_l1[0];
                self.router_l1_weights[i * 2 + 1] -= self.learning_rate * token_state[i] * grad_l1[1];
                active_l2_w[i * 2 + 0] -= self.learning_rate * token_state[i] * grad_l2[0];
                active_l2_w[i * 2 + 1] -= self.learning_rate * token_state[i] * grad_l2[1];
            }

            for (0..d) |oi| {
                var sum: f32 = 0.0;
                for (0..d) |ii| sum += token_state[ii] * active_expert_w[oi * d + ii];
                exp_out[oi] = (if (sum > 0.0) sum else 0.0) + token_state[oi];
            }

            var mv: f32 = -1e9;
            for (0..VOCAB_SIZE) |v| {
                var s: f32 = 0.0;
                for (0..d) |h| s += exp_out[h] * self.lm_head_w[v * d + h];
                v_logits[v] = s;
                if (s > mv) mv = s;
            }
            var vs: f32 = 0.0;
            for (0..VOCAB_SIZE) |v| {
                v_prob[v] = @exp(v_logits[v] - mv);
                vs += v_prob[v];
            }
            for (0..VOCAB_SIZE) |v| v_prob[v] /= vs;

            const prob_target = @max(v_prob[target_token], 1e-9);
            batch_loss -= @log(prob_target);

            @memset(d_exp, 0.0);
            for (0..VOCAB_SIZE) |v| {
                const dl = v_prob[v] - (if (v == target_token) @as(f32, 1.0) else 0.0);
                for (0..d) |h| {
                    d_exp[h] += dl * self.lm_head_w[v * d + h];
                    self.lm_head_w[v * d + h] -= self.learning_rate * dl * exp_out[h];
                }
            }

            @memset(d_ts, 0.0);
            for (0..d) |o| {
                d_ts[o] += d_exp[o];

                var sum: f32 = 0.0;
                for (0..d) |i| sum += token_state[i] * active_expert_w[o * d + i];
                const dr = if (sum > 0.0) d_exp[o] else 0.0;

                for (0..d) |i| {
                    d_ts[i] += dr * active_expert_w[o * d + i];
                    active_expert_w[o * d + i] -= self.learning_rate * dr * token_state[i];
                }
            }

            for (0..d) |i| {
                d_ts[i] += grad_l1[0] * self.router_l1_weights[i * 2 + 0];
                d_ts[i] += grad_l1[1] * self.router_l1_weights[i * 2 + 1];
                d_ts[i] += grad_l2[0] * active_l2_w[i * 2 + 0];
                d_ts[i] += grad_l2[1] * active_l2_w[i * 2 + 1];
            }

            @memset(d_pooled, 0.0);
            for (0..d) |out_i| {
                for (0..d) |in_i| {
                    d_pooled[in_i] += d_ts[out_i] * self.intent_weights[out_i * d + in_i];
                    self.intent_weights[out_i * d + in_i] -= self.learning_rate * d_ts[out_i] * pooled[in_i];
                }
            }

            pos_iter = 0;
            for (batch.inputs) |tid| {
                if (pos_iter >= MAX_SEQ_LEN) break;
                const sid = @min(tid, VOCAB_SIZE - 1);
                for (0..d) |i| {
                    self.mock_embeddings[sid * d + i] -= self.learning_rate * d_pooled[i] * scale;
                    self.pos_embeddings[pos_iter * d + i] -= self.learning_rate * d_pooled[i] * scale;
                }
                pos_iter += 1;
            }
            for (batch.targets[0..t_idx]) |tid| {
                if (pos_iter >= MAX_SEQ_LEN) break;
                const sid = @min(tid, VOCAB_SIZE - 1);
                for (0..d) |i| {
                    self.mock_embeddings[sid * d + i] -= self.learning_rate * d_pooled[i] * scale;
                    self.pos_embeddings[pos_iter * d + i] -= self.learning_rate * d_pooled[i] * scale;
                }
                pos_iter += 1;
            }
        }
        return batch_loss / @as(f32, @floatFromInt(batch.targets.len));
    }

    pub fn save(self: *DualBrainTrainer, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        const weights = [_][]f32{ self.mock_embeddings, self.pos_embeddings, self.intent_weights, self.router_l1_weights, self.router_l2_left_weights, self.router_l2_right_weights, self.expert_calc_w, self.expert_syntax_w, self.expert_future_w, self.expert_story_w, self.lm_head_w };
        for (weights) |w| try file.writeAll(std.mem.sliceAsBytes(w));
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
        };
        const weights = [_][]f32{ t.mock_embeddings, t.pos_embeddings, t.intent_weights, t.router_l1_weights, t.router_l2_left_weights, t.router_l2_right_weights, t.expert_calc_w, t.expert_syntax_w, t.expert_future_w, t.expert_story_w, t.lm_head_w };
        for (weights) |w| _ = try file.readAll(std.mem.sliceAsBytes(w));
        return t;
    }

    pub fn deinit(self: *DualBrainTrainer) void {
        const weights = [_][]f32{ self.mock_embeddings, self.pos_embeddings, self.intent_weights, self.router_l1_weights, self.router_l2_left_weights, self.router_l2_right_weights, self.expert_calc_w, self.expert_syntax_w, self.expert_future_w, self.expert_story_w, self.lm_head_w };
        for (weights) |w| self.allocator.free(w);
    }
};
