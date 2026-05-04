const std = @import("std");
const math = @import("../utils/math.zig");

// ==========================================
// ⚙️ KONFIGURASI ARSITEKTUR V6 (DEEP HMoE)
// ==========================================
const HIDDEN_DIM: usize = 1024;
const VOCAB_SIZE: usize = 15000;
const NUM_LAYERS: usize = 2; // 🚀 OTOMATIS BERJALAN 2 LAPIS!

// Struktur Bobot untuk 1 Lapisan (Block)
pub const Layer = struct {
    norm1: []f32,
    time_decay: []f32,
    norm2: []f32,
    exp_calc: []f32,
    exp_sync: []f32,
    exp_sci: []f32,
    exp_story: []f32,
    state: []f32, // Memori independen untuk layer ini
};

pub const DualBrainEngine = struct {
    allocator: std.mem.Allocator,

    embeddings: []f32,
    router_l1: []f32,
    router_l2_left: []f32,
    router_l2_right: []f32,

    layers: []Layer,

    final_norm: []f32,
    lm_head: []f32,

    // Buffer Internal yang Melintasi Lapis demi Lapis
    buf_x: []f32,
    buf_norm: []f32,
    buf_res: []f32,
    buf_norm2: []f32,
    buf_expert: []f32,
    buf_logits: []f32,

    // Fungsi Pembantu: Root Mean Square Normalization (RMSNorm)
    fn rmsnorm(out: []f32, in: []const f32, weight: []const f32) void {
        var ss: f32 = 0.0;
        for (in) |v| ss += v * v;
        ss /= @as(f32, @floatFromInt(in.len));
        ss += 1e-6;
        const inv_sqrt = 1.0 / @sqrt(ss);
        for (0..in.len) |i| {
            out[i] = in[i] * inv_sqrt * weight[i];
        }
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !DualBrainEngine {
        std.debug.print("📥 Memuat arsitektur V6 (Deep Layer) dari: {s}\n", .{path});
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const engine = DualBrainEngine{
            .allocator = allocator,
            .embeddings = try allocator.alloc(f32, VOCAB_SIZE * HIDDEN_DIM),
            .router_l1 = try allocator.alloc(f32, HIDDEN_DIM * 2),
            .router_l2_left = try allocator.alloc(f32, HIDDEN_DIM * 2),
            .router_l2_right = try allocator.alloc(f32, HIDDEN_DIM * 2),

            .layers = try allocator.alloc(Layer, NUM_LAYERS),

            .final_norm = try allocator.alloc(f32, HIDDEN_DIM),
            .lm_head = try allocator.alloc(f32, VOCAB_SIZE * HIDDEN_DIM),

            .buf_x = try allocator.alloc(f32, HIDDEN_DIM),
            .buf_norm = try allocator.alloc(f32, HIDDEN_DIM),
            .buf_res = try allocator.alloc(f32, HIDDEN_DIM),
            .buf_norm2 = try allocator.alloc(f32, HIDDEN_DIM),
            .buf_expert = try allocator.alloc(f32, HIDDEN_DIM),
            .buf_logits = try allocator.alloc(f32, VOCAB_SIZE),
        };

        _ = try file.readAll(std.mem.sliceAsBytes(engine.embeddings));
        _ = try file.readAll(std.mem.sliceAsBytes(engine.router_l1));
        _ = try file.readAll(std.mem.sliceAsBytes(engine.router_l2_left));
        _ = try file.readAll(std.mem.sliceAsBytes(engine.router_l2_right));

        // Memuat Layer Dinamis
        for (engine.layers) |*layer| {
            layer.norm1 = try allocator.alloc(f32, HIDDEN_DIM);
            layer.time_decay = try allocator.alloc(f32, HIDDEN_DIM);
            layer.norm2 = try allocator.alloc(f32, HIDDEN_DIM);
            layer.exp_calc = try allocator.alloc(f32, HIDDEN_DIM * HIDDEN_DIM);
            layer.exp_sync = try allocator.alloc(f32, HIDDEN_DIM * HIDDEN_DIM);
            layer.exp_sci = try allocator.alloc(f32, HIDDEN_DIM * HIDDEN_DIM);
            layer.exp_story = try allocator.alloc(f32, HIDDEN_DIM * HIDDEN_DIM);
            layer.state = try allocator.alloc(f32, HIDDEN_DIM);

            _ = try file.readAll(std.mem.sliceAsBytes(layer.norm1));
            _ = try file.readAll(std.mem.sliceAsBytes(layer.time_decay));
            _ = try file.readAll(std.mem.sliceAsBytes(layer.norm2));
            _ = try file.readAll(std.mem.sliceAsBytes(layer.exp_calc));
            _ = try file.readAll(std.mem.sliceAsBytes(layer.exp_sync));
            _ = try file.readAll(std.mem.sliceAsBytes(layer.exp_sci));
            _ = try file.readAll(std.mem.sliceAsBytes(layer.exp_story));
        }

        _ = try file.readAll(std.mem.sliceAsBytes(engine.final_norm));
        _ = try file.readAll(std.mem.sliceAsBytes(engine.lm_head));

        std.debug.print("✅ Deep Cyber-Brain V6 berhasil dimuat ke RAM!\n", .{});
        return engine;
    }

    pub fn deinit(self: *DualBrainEngine) void {
        self.allocator.free(self.embeddings);
        self.allocator.free(self.router_l1);
        self.allocator.free(self.router_l2_left);
        self.allocator.free(self.router_l2_right);

        for (self.layers) |layer| {
            self.allocator.free(layer.norm1);
            self.allocator.free(layer.time_decay);
            self.allocator.free(layer.norm2);
            self.allocator.free(layer.exp_calc);
            self.allocator.free(layer.exp_sync);
            self.allocator.free(layer.exp_sci);
            self.allocator.free(layer.exp_story);
            self.allocator.free(layer.state);
        }
        self.allocator.free(self.layers);
        self.allocator.free(self.final_norm);
        self.allocator.free(self.lm_head);

        self.allocator.free(self.buf_x);
        self.allocator.free(self.buf_norm);
        self.allocator.free(self.buf_res);
        self.allocator.free(self.buf_norm2);
        self.allocator.free(self.buf_expert);
        self.allocator.free(self.buf_logits);
    }

    pub fn generate(self: *DualBrainEngine, start: []const u32, max_tokens: usize, tokenizer: anytype) !void {
        var stdout_buffer: [4096]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
        const stdout = &stdout_writer.interface;

        var timer = try std.time.Timer.start();
        var prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp())));
        const random = prng.random();

        // 1. KOSONGKAN INGATAN SEMUA LAYER
        for (self.layers) |*layer| @memset(layer.state, 0.0);

        // 2. KLASIFIKASI PROMPT (BERDASARKAN KATA TERAKHIR PROMPT)
        const last_prompt_tok = start[start.len - 1];
        const last_tok_id = @min(last_prompt_tok, VOCAB_SIZE - 1);
        const router_emb = self.embeddings[last_tok_id * HIDDEN_DIM .. (last_tok_id + 1) * HIDDEN_DIM];

        var l1 = [2]f32{ 0.0, 0.0 };
        for (0..HIDDEN_DIM) |i| {
            l1[0] += router_emb[i] * self.router_l1[i * 2 + 0];
            l1[1] += router_emb[i] * self.router_l1[i * 2 + 1];
        }
        const is_right = l1[1] > l1[0];
        const current_hemi: usize = if (is_right) 1 else 0;

        const l2w = if (is_right) self.router_l2_right else self.router_l2_left;
        var l2 = [2]f32{ 0.0, 0.0 };
        for (0..HIDDEN_DIM) |i| {
            l2[0] += router_emb[i] * l2w[i * 2 + 0];
            l2[1] += router_emb[i] * l2w[i * 2 + 1];
        }

        var current_expert: usize = 0;
        if (!is_right) {
            current_expert = if (l2[0] > l2[1]) 0 else 1;
        } else {
            current_expert = if (l2[0] > l2[1]) 2 else 3;
        }

        const active_temp: f32 = if (current_hemi == 0) 0.2 else 0.8;
        try stdout.print("\n   [🔒 EXPERT {d} | 🌡️ TEMP: {d:.1} | 🥞 LAYERS: {d}]\n   ", .{ current_expert, active_temp, NUM_LAYERS });
        try stdout.flush();

        // 3. PIPELINE UTAMA: MEMBACA PROMPT & MENEBAK
        var next_tok = start[0];
        var i: usize = 0;
        var generated: usize = 0;

        while (true) {
            // Ambil token saat ini
            const tok_id = @min(next_tok, VOCAB_SIZE - 1);
            @memcpy(self.buf_x, self.embeddings[tok_id * HIDDEN_DIM .. (tok_id + 1) * HIDDEN_DIM]);

            // 🚀 PERJALANAN MASUK KE DEEP LAYERS
            for (self.layers) |*layer| {
                // a. Normalisasi Pertama
                rmsnorm(self.buf_norm, self.buf_x, layer.norm1);

                // b. Evolusi Memori (EMA)
                for (0..HIDDEN_DIM) |j| {
                    const decay = layer.time_decay[j];
                    layer.state[j] = (layer.state[j] * decay) + (self.buf_norm[j] * (1.0 - decay));
                }

                // c. Residual 1 (Bypass)
                for (0..HIDDEN_DIM) |j| {
                    self.buf_res[j] = self.buf_x[j] + layer.state[j];
                }

                // d. Normalisasi Kedua
                rmsnorm(self.buf_norm2, self.buf_res, layer.norm2);

                // e. Mengalir ke Expert yang Aktif
                const exp_w = switch (current_expert) {
                    0 => layer.exp_calc,
                    1 => layer.exp_sync,
                    2 => layer.exp_sci,
                    3 => layer.exp_story,
                    else => unreachable,
                };

                for (0..HIDDEN_DIM) |out_i| {
                    const sum = math.dot(self.buf_norm2, exp_w[out_i * HIDDEN_DIM .. out_i * HIDDEN_DIM + HIDDEN_DIM]);
                    self.buf_expert[out_i] = if (sum > 0.0) sum else 0.0; // ReLU
                }

                // f. Residual 2 (Output Expert digabung ke Jalur Utama)
                for (0..HIDDEN_DIM) |j| {
                    self.buf_x[j] = self.buf_x[j] + self.buf_expert[j];
                }
            }
            // Selesai melewati seluruh Layer! Output akhir ada di buf_x.

            // 4. JIKA KITA SUDAH SELESAI MEMBACA PROMPT, MULAI MENEBAK!
            if (i >= start.len - 1) {
                // Normalisasi Pintu Keluar
                rmsnorm(self.buf_norm, self.buf_x, self.final_norm);

                // LM Head & Sampling Temperatur
                var max_logit: f32 = -std.math.inf(f32);
                for (0..VOCAB_SIZE) |v| {
                    const logit = math.dot(self.buf_norm, self.lm_head[v * HIDDEN_DIM .. v * HIDDEN_DIM + HIDDEN_DIM]);
                    self.buf_logits[v] = logit;
                    if (logit > max_logit) max_logit = logit;
                }

                var best_tok: u32 = 0;
                var sum_probs: f32 = 0.0;
                for (0..VOCAB_SIZE) |v| {
                    const prob = @exp((self.buf_logits[v] - max_logit) / active_temp);
                    self.buf_logits[v] = prob;
                    sum_probs += prob;
                }

                const r = random.float(f32) * sum_probs;
                var accum: f32 = 0.0;
                for (0..VOCAB_SIZE) |v| {
                    accum += self.buf_logits[v];
                    if (accum >= r) {
                        best_tok = @intCast(v);
                        break;
                    }
                }

                if (best_tok == 2 or best_tok == 0 or generated >= max_tokens) break;

                // Kata yang ditebak menjadi input untuk iterasi selanjutnya
                next_tok = best_tok;
                generated += 1;

                const word = tokenizer.decode(best_tok);
                if (!std.mem.eql(u8, word, "<UNK>")) {
                    try stdout.print("{s} ", .{word});
                    try stdout.flush();
                }
            } else {
                // Jika masih menyerap prompt, langsung lanjut ke kata prompt berikutnya
                next_tok = start[i + 1];
            }

            i += 1;
        }

        const elapsed = @as(f32, @floatFromInt(timer.read())) / 1_000_000_000.0;
        const tps = @as(f32, @floatFromInt(generated)) / elapsed;
        try stdout.print("\n\n ----Telemetri ---------------------------------------\n", .{});
        try stdout.print("   | Hemisphere : {s}\n", .{if (current_hemi == 0) "Kiri (Exact)" else "Kanan (Imaginasi)"});
        try stdout.print("   | Expert Dom : {d}\n", .{current_expert});
        try stdout.print("   | Benchmark  : {d} tokens in {d:.3}s ({d:.2} TPS)\n", .{ generated, elapsed, tps });
        try stdout.print("   |---------------------------------------------------\n\n", .{});
        try stdout.flush();
    }
};
