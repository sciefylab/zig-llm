const std = @import("std");
const train_data = @import("train_data.zig");

const HIDDEN_DIM: usize = 64;
const VOCAB_SIZE: usize = 500;
const MAX_SEQ_LEN: usize = 64;

pub const DualBrainTrainer = struct {
    allocator: std.mem.Allocator,
    learning_rate: f32,
    prng: std.Random.DefaultPrng,

    mock_embeddings: []f32,
    pos_embeddings: []f32,
    attn_wq: []f32,
    attn_wk: []f32,
    attn_wv: []f32,
    attn_wo: []f32,
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
            .attn_wq = try init_w(allocator, HIDDEN_DIM * HIDDEN_DIM, random),
            .attn_wk = try init_w(allocator, HIDDEN_DIM * HIDDEN_DIM, random),
            .attn_wv = try init_w(allocator, HIDDEN_DIM * HIDDEN_DIM, random),
            .attn_wo = try init_w(allocator, HIDDEN_DIM * HIDDEN_DIM, random),
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

    pub fn computeSelfAttention(self: *DualBrainTrainer, ctx_tokens: []const u32) ![]f32 {
        const seq_len = ctx_tokens.len;
        const d = HIDDEN_DIM;

        var seq_emb = try self.allocator.alloc(f32, seq_len * d);
        defer self.allocator.free(seq_emb);
        for (ctx_tokens, 0..) |tid, pos| {
            const sid = @min(tid, VOCAB_SIZE - 1);
            const sp = @min(pos, MAX_SEQ_LEN - 1);
            for (0..d) |i| seq_emb[pos * d + i] = self.mock_embeddings[sid * d + i] + self.pos_embeddings[sp * d + i];
        }

        var q = try self.allocator.alloc(f32, d);
        defer self.allocator.free(q);
        @memset(q, 0.0);
        const last_pos = seq_len - 1;
        for (0..d) |out_i| {
            for (0..d) |in_i| q[out_i] += seq_emb[last_pos * d + in_i] * self.attn_wq[out_i * d + in_i];
        }

        var keys = try self.allocator.alloc(f32, seq_len * d);
        defer self.allocator.free(keys);
        var values = try self.allocator.alloc(f32, seq_len * d);
        defer self.allocator.free(values);
        @memset(keys, 0.0);
        @memset(values, 0.0);

        for (0..seq_len) |pos| {
            for (0..d) |out_i| {
                for (0..d) |in_i| {
                    keys[pos * d + out_i] += seq_emb[pos * d + in_i] * self.attn_wk[out_i * d + in_i];
                    values[pos * d + out_i] += seq_emb[pos * d + in_i] * self.attn_wv[out_i * d + in_i];
                }
            }
        }

        var scores = try self.allocator.alloc(f32, seq_len);
        defer self.allocator.free(scores);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        var max_score: f32 = -1e9;
        for (0..seq_len) |pos| {
            var dot: f32 = 0.0;
            for (0..d) |i| dot += q[i] * keys[pos * d + i];
            scores[pos] = dot * scale;
            if (scores[pos] > max_score) max_score = scores[pos];
        }

        var sum_exp: f32 = 0.0;
        for (0..seq_len) |pos| {
            scores[pos] = @exp(scores[pos] - max_score);
            sum_exp += scores[pos];
        }
        for (0..seq_len) |pos| scores[pos] /= sum_exp;

        var attn_out = try self.allocator.alloc(f32, d);
        defer self.allocator.free(attn_out);
        @memset(attn_out, 0.0);
        for (0..seq_len) |pos| {
            for (0..d) |i| attn_out[i] += scores[pos] * values[pos * d + i];
        }

        var final_out = try self.allocator.alloc(f32, d);
        @memset(final_out, 0.0);
        for (0..d) |out_i| {
            for (0..d) |in_i| final_out[out_i] += attn_out[in_i] * self.attn_wo[out_i * d + in_i];
        }

        // --- 🔥 FIX: RESIDUAL CONNECTION ATTENTION ---
        for (0..d) |i| final_out[i] += seq_emb[last_pos * d + i];

        return final_out;
    }

    pub fn trainStep(self: *DualBrainTrainer, batch: train_data.HMoEBatch) !f32 {
        if (batch.targets.len == 0) return 0.0;
        const d = HIDDEN_DIM;

        var seq_emb = try self.allocator.alloc(f32, MAX_SEQ_LEN * d);
        defer self.allocator.free(seq_emb);
        var q = try self.allocator.alloc(f32, d);
        defer self.allocator.free(q);
        var keys = try self.allocator.alloc(f32, MAX_SEQ_LEN * d);
        defer self.allocator.free(keys);
        var values = try self.allocator.alloc(f32, MAX_SEQ_LEN * d);
        defer self.allocator.free(values);
        var scores = try self.allocator.alloc(f32, MAX_SEQ_LEN);
        defer self.allocator.free(scores);
        var attn_out = try self.allocator.alloc(f32, d);
        defer self.allocator.free(attn_out);
        var token_state = try self.allocator.alloc(f32, d);
        defer self.allocator.free(token_state);

        var d_ts = try self.allocator.alloc(f32, d);
        defer self.allocator.free(d_ts);
        var d_attn_out = try self.allocator.alloc(f32, d);
        defer self.allocator.free(d_attn_out);
        var d_scores = try self.allocator.alloc(f32, MAX_SEQ_LEN);
        defer self.allocator.free(d_scores);
        var d_scores_pre = try self.allocator.alloc(f32, MAX_SEQ_LEN);
        defer self.allocator.free(d_scores_pre);
        var d_values = try self.allocator.alloc(f32, MAX_SEQ_LEN * d);
        defer self.allocator.free(d_values);
        var d_q = try self.allocator.alloc(f32, d);
        defer self.allocator.free(d_q);
        var d_keys = try self.allocator.alloc(f32, MAX_SEQ_LEN * d);
        defer self.allocator.free(d_keys);
        var d_seq_emb = try self.allocator.alloc(f32, MAX_SEQ_LEN * d);
        defer self.allocator.free(d_seq_emb);

        var exp_out = try self.allocator.alloc(f32, d);
        defer self.allocator.free(exp_out);
        var v_logits = try self.allocator.alloc(f32, VOCAB_SIZE);
        defer self.allocator.free(v_logits);
        var v_prob = try self.allocator.alloc(f32, VOCAB_SIZE);
        defer self.allocator.free(v_prob);
        var d_exp = try self.allocator.alloc(f32, d);
        defer self.allocator.free(d_exp);

        var batch_loss: f32 = 0.0;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

        for (batch.targets, 0..) |target_token, t_idx| {
            const total_ctx_len = @min(batch.inputs.len + t_idx, MAX_SEQ_LEN);
            const last_pos = total_ctx_len - 1;

            @memset(seq_emb, 0.0);
            @memset(keys, 0.0);
            @memset(values, 0.0);
            @memset(q, 0.0);

            var pos_iter: usize = 0;
            for (batch.inputs) |tid| {
                if (pos_iter >= MAX_SEQ_LEN) break;
                const sid = @min(tid, VOCAB_SIZE - 1);
                for (0..d) |i| seq_emb[pos_iter * d + i] = self.mock_embeddings[sid * d + i] + self.pos_embeddings[pos_iter * d + i];
                pos_iter += 1;
            }
            for (batch.targets[0..t_idx]) |tid| {
                if (pos_iter >= MAX_SEQ_LEN) break;
                const sid = @min(tid, VOCAB_SIZE - 1);
                for (0..d) |i| seq_emb[pos_iter * d + i] = self.mock_embeddings[sid * d + i] + self.pos_embeddings[pos_iter * d + i];
                pos_iter += 1;
            }

            for (0..d) |out_i| {
                for (0..d) |in_i| q[out_i] += seq_emb[last_pos * d + in_i] * self.attn_wq[out_i * d + in_i];
            }
            for (0..total_ctx_len) |pos| {
                for (0..d) |out_i| {
                    for (0..d) |in_i| {
                        keys[pos * d + out_i] += seq_emb[pos * d + in_i] * self.attn_wk[out_i * d + in_i];
                        values[pos * d + out_i] += seq_emb[pos * d + in_i] * self.attn_wv[out_i * d + in_i];
                    }
                }
            }

            var max_score: f32 = -1e9;
            for (0..total_ctx_len) |pos| {
                var dot: f32 = 0.0;
                for (0..d) |i| dot += q[i] * keys[pos * d + i];
                scores[pos] = dot * scale;
                if (scores[pos] > max_score) max_score = scores[pos];
            }
            var sum_exp: f32 = 0.0;
            for (0..total_ctx_len) |pos| {
                scores[pos] = @exp(scores[pos] - max_score);
                sum_exp += scores[pos];
            }
            for (0..total_ctx_len) |pos| scores[pos] /= sum_exp;

            @memset(attn_out, 0.0);
            for (0..total_ctx_len) |pos| {
                for (0..d) |i| attn_out[i] += scores[pos] * values[pos * d + i];
            }
            @memset(token_state, 0.0);
            for (0..d) |out_i| {
                for (0..d) |in_i| token_state[out_i] += attn_out[in_i] * self.attn_wo[out_i * d + in_i];
            }
            // --- 🔥 FIX: RESIDUAL CONNECTION ATTENTION (FORWARD) ---
            for (0..d) |i| token_state[i] += seq_emb[last_pos * d + i];

            // ROUTER L1 & L2
            var l1_logits = [2]f32{ 0.0, 0.0 };
            for (0..d) |i| {
                l1_logits[0] += token_state[i] * self.router_l1_weights[i * 2 + 0];
                l1_logits[1] += token_state[i] * self.router_l1_weights[i * 2 + 1];
            }
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

            // --- 🔥 FIX: RESIDUAL CONNECTION EXPERT (FORWARD) ---
            for (0..d) |oi| {
                var sum: f32 = 0.0;
                for (0..d) |ii| sum += token_state[ii] * active_expert_w[oi * d + ii];
                exp_out[oi] = (if (sum > 0.0) sum else 0.0) + token_state[oi]; // Penjumlahan Residual
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

            // BACKWARD PASS
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
                d_ts[o] += d_exp[o]; // Jalur Balik Residual Expert

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

            @memset(d_attn_out, 0.0);
            @memset(d_seq_emb, 0.0);

            // --- 🔥 FIX: JALUR BALIK RESIDUAL ATTENTION ---
            for (0..d) |i| d_seq_emb[last_pos * d + i] += d_ts[i];

            for (0..d) |out_i| {
                for (0..d) |in_i| {
                    d_attn_out[in_i] += d_ts[out_i] * self.attn_wo[out_i * d + in_i];
                    self.attn_wo[out_i * d + in_i] -= self.learning_rate * d_ts[out_i] * attn_out[in_i];
                }
            }

            @memset(d_scores, 0.0);
            @memset(d_values, 0.0);
            var sum_s_ds: f32 = 0.0;
            for (0..total_ctx_len) |pos| {
                for (0..d) |i| {
                    d_scores[pos] += d_attn_out[i] * values[pos * d + i];
                    d_values[pos * d + i] = d_attn_out[i] * scores[pos];
                }
                sum_s_ds += scores[pos] * d_scores[pos];
            }
            for (0..total_ctx_len) |pos| {
                d_scores_pre[pos] = scores[pos] * (d_scores[pos] - sum_s_ds);
            }

            @memset(d_q, 0.0);
            @memset(d_keys, 0.0);
            for (0..total_ctx_len) |pos| {
                const ds = d_scores_pre[pos] * scale;
                for (0..d) |i| {
                    d_q[i] += ds * keys[pos * d + i];
                    d_keys[pos * d + i] += ds * q[i];
                }
            }

            for (0..d) |out_i| {
                for (0..d) |in_i| {
                    d_seq_emb[last_pos * d + in_i] += d_q[out_i] * self.attn_wq[out_i * d + in_i];
                    self.attn_wq[out_i * d + in_i] -= self.learning_rate * d_q[out_i] * seq_emb[last_pos * d + in_i];
                }
            }
            for (0..total_ctx_len) |pos| {
                for (0..d) |out_i| {
                    for (0..d) |in_i| {
                        d_seq_emb[pos * d + in_i] += d_keys[pos * d + out_i] * self.attn_wk[out_i * d + in_i];
                        self.attn_wk[out_i * d + in_i] -= self.learning_rate * d_keys[pos * d + out_i] * seq_emb[pos * d + in_i];

                        d_seq_emb[pos * d + in_i] += d_values[pos * d + out_i] * self.attn_wv[out_i * d + in_i];
                        self.attn_wv[out_i * d + in_i] -= self.learning_rate * d_values[pos * d + out_i] * seq_emb[pos * d + in_i];
                    }
                }
            }

            pos_iter = 0;
            for (batch.inputs) |tid| {
                if (pos_iter >= MAX_SEQ_LEN) break;
                const sid = @min(tid, VOCAB_SIZE - 1);
                for (0..d) |i| {
                    self.mock_embeddings[sid * d + i] -= self.learning_rate * d_seq_emb[pos_iter * d + i];
                    self.pos_embeddings[pos_iter * d + i] -= self.learning_rate * d_seq_emb[pos_iter * d + i];
                }
                pos_iter += 1;
            }
            for (batch.targets[0..t_idx]) |tid| {
                if (pos_iter >= MAX_SEQ_LEN) break;
                const sid = @min(tid, VOCAB_SIZE - 1);
                for (0..d) |i| {
                    self.mock_embeddings[sid * d + i] -= self.learning_rate * d_seq_emb[pos_iter * d + i];
                    self.pos_embeddings[pos_iter * d + i] -= self.learning_rate * d_seq_emb[pos_iter * d + i];
                }
                pos_iter += 1;
            }
        }
        return batch_loss / @as(f32, @floatFromInt(batch.targets.len));
    }

    pub fn infer(self: *DualBrainTrainer, input_tokens: []const u32) !u32 {
        const d = HIDDEN_DIM;
        const ts = try self.computeSelfAttention(input_tokens);
        defer self.allocator.free(ts);

        var l1 = [2]f32{ 0.0, 0.0 };
        for (0..d) |i| {
            l1[0] += ts[i] * self.router_l1_weights[i * 2 + 0];
            l1[1] += ts[i] * self.router_l1_weights[i * 2 + 1];
        }
        const is_dex = l1[1] > l1[0];

        std.debug.print("[Rute: {s}] ", .{if (is_dex) "Kanan" else "Kiri"});

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

        var out = try self.allocator.alloc(f32, d);
        defer self.allocator.free(out);
        for (0..d) |oi| {
            var sum: f32 = 0.0;
            for (0..d) |ii| sum += ts[ii] * exp_w[oi * d + ii];
            out[oi] = (if (sum > 0.0) sum else 0.0) + ts[oi]; // --- 🔥 FIX: RESIDUAL EXPERT ---
        }

        var mv: f32 = -1e9;
        var best_v: u32 = 0;
        for (0..VOCAB_SIZE) |v| {
            var s: f32 = 0.0;
            for (0..d) |h| s += out[h] * self.lm_head_w[v * d + h];
            if (s > mv) {
                mv = s;
                best_v = @intCast(v);
            }
        }
        return best_v;
    }

    pub fn generate(self: *DualBrainTrainer, start: []const u32, max: usize, tokenizer: anytype) !void {
        var ctx: std.ArrayList(u32) = .empty;
        defer ctx.deinit(self.allocator);
        try ctx.appendSlice(self.allocator, start);
        std.debug.print("\n🤖: ", .{});

        var timer = try std.time.Timer.start();
        var generated_count: usize = 0;

        for (0..max) |_| {
            const nxt = try self.infer(ctx.items);
            const w = tokenizer.decode(nxt);
            if (std.mem.eql(u8, w, "<|END|>") or std.mem.eql(u8, w, "[UNK]")) break;

            std.debug.print("{s} ", .{w});
            try ctx.append(self.allocator, nxt);
            generated_count += 1;
        }
        const elapsed_ns = timer.read();
        const elapsed_s = @as(f32, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        const tps = @as(f32, @floatFromInt(generated_count)) / @max(elapsed_s, 0.0001);
        std.debug.print("\n[⏱️ Benchmark: {d} token | Waktu: {d:.4} detik | Kecepatan: {d:.2} TPS]\n", .{ generated_count, elapsed_s, tps });
    }

    pub fn save(self: *DualBrainTrainer, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        const weights = [_][]f32{ self.mock_embeddings, self.pos_embeddings, self.attn_wq, self.attn_wk, self.attn_wv, self.attn_wo, self.router_l1_weights, self.router_l2_left_weights, self.router_l2_right_weights, self.expert_calc_w, self.expert_syntax_w, self.expert_future_w, self.expert_story_w, self.lm_head_w };
        for (weights) |w| try file.writeAll(std.mem.sliceAsBytes(w));
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !DualBrainTrainer {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const d = HIDDEN_DIM;
        const t = DualBrainTrainer{
            .allocator = allocator,
            .learning_rate = 0,
            .prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp()))),
            .mock_embeddings = try allocator.alloc(f32, VOCAB_SIZE * d),
            .pos_embeddings = try allocator.alloc(f32, MAX_SEQ_LEN * d),
            .attn_wq = try allocator.alloc(f32, d * d),
            .attn_wk = try allocator.alloc(f32, d * d),
            .attn_wv = try allocator.alloc(f32, d * d),
            .attn_wo = try allocator.alloc(f32, d * d),
            .router_l1_weights = try allocator.alloc(f32, d * 2),
            .router_l2_left_weights = try allocator.alloc(f32, d * 2),
            .router_l2_right_weights = try allocator.alloc(f32, d * 2),
            .expert_calc_w = try allocator.alloc(f32, d * d),
            .expert_syntax_w = try allocator.alloc(f32, d * d),
            .expert_future_w = try allocator.alloc(f32, d * d),
            .expert_story_w = try allocator.alloc(f32, d * d),
            .lm_head_w = try allocator.alloc(f32, VOCAB_SIZE * d),
        };
        const weights = [_][]f32{ t.mock_embeddings, t.pos_embeddings, t.attn_wq, t.attn_wk, t.attn_wv, t.attn_wo, t.router_l1_weights, t.router_l2_left_weights, t.router_l2_right_weights, t.expert_calc_w, t.expert_syntax_w, t.expert_future_w, t.expert_story_w, t.lm_head_w };
        for (weights) |w| _ = try file.readAll(std.mem.sliceAsBytes(w));
        return t;
    }

    pub fn deinit(self: *DualBrainTrainer) void {
        const weights = [_][]f32{ self.mock_embeddings, self.pos_embeddings, self.attn_wq, self.attn_wk, self.attn_wv, self.attn_wo, self.router_l1_weights, self.router_l2_left_weights, self.router_l2_right_weights, self.expert_calc_w, self.expert_syntax_w, self.expert_future_w, self.expert_story_w, self.lm_head_w };
        for (weights) |w| self.allocator.free(w);
    }
};
