const std = @import("std");
const math = @import("../utils/math.zig");

// ==========================================
// ⚙️ KONFIGURASI ARSITEKTUR & HARDWARE
// ==========================================
const HIDDEN_DIM: usize = 1024;
const VOCAB_SIZE: usize = 15000;
const NUM_LAYERS: usize = 2;

// 🎚️ SAKLAR KENDALI MESIN
const USE_MULTI_THREADING: bool = true; // Ubah ke `false` jika ingin kembali ke Single-Thread
const NUM_THREADS: u32 = 8; // Sesuaikan dengan jumlah Core CPU fisik Anda
const CHUNK_SIZE: usize = HIDDEN_DIM / @as(usize, NUM_THREADS);

pub const Layer = struct {
    norm1: []f32,
    time_decay: []f32,
    norm2: []f32,
    // Memori Expert 75% Lebih Kecil (1 Byte / i8)
    exp_calc: []i8,
    exp_sync: []i8,
    exp_sci: []i8,
    exp_story: []i8,
    state: []f32,
};

const WorkerContext = struct {
    start_idx: usize,
    end_idx: usize,
    exp_w: []i8,
    fixed_x: []i32,
    buf_expert: []f32,
};

pub const DualBrainEngine = struct {
    allocator: std.mem.Allocator,

    // 🏊‍♂️ Kolam Pekerja (Thread Pool)
    pool: ?*std.Thread.Pool,

    embeddings: []f32,
    router_l1: []f32,
    router_l2_left: []f32,
    router_l2_right: []f32,
    layers: []Layer,
    final_norm: []f32,
    lm_head: []f32,

    buf_x: []f32,
    buf_norm: []f32,
    buf_res: []f32,
    buf_norm2: []f32,
    buf_expert: []f32,
    buf_logits: []f32,

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

    // 🚀 FUNGSI KERNEL INTI (Branchless Auto-SIMD Optimized)
    fn expertWorkerSequential(ctx: WorkerContext) void {
        for (ctx.start_idx..ctx.end_idx) |out_i| {
            var sum_fixed: i32 = 0;
            const w_row = ctx.exp_w[out_i * HIDDEN_DIM .. (out_i + 1) * HIDDEN_DIM];

            // ⚡ LOOP TANPA CABANG (Memaksa CPU menggunakan SIMD AVX2/Neon)
            for (0..HIDDEN_DIM) |in_i| {
                const w_int: i32 = w_row[in_i]; // Cast otomatis ke i32 untuk register

                // 1. Evaluasi Masking Murni (Tanpa If-Block)
                const is_not_zero = (w_int != 0);
                const is_pos = (w_int > 0);

                // 2. Hitung Absolut & Shift secara paralel
                const abs_w: i32 = if (is_pos) w_int else -w_int;
                const shift_amount: u5 = @intCast(if (is_not_zero) abs_w - 1 else 0);

                // 3. Eksekusi Geser Bit (Semua digeser, tidak peduli aslinya 0)
                const val = ctx.fixed_x[in_i] >> shift_amount;

                // 4. Terapkan Masking (Nol-kan hasil jika bobot aslinya 0)
                const masked_val = if (is_not_zero) val else 0;

                // 5. Akumulasi berdasarkan tanda (+ atau -)
                sum_fixed += if (is_pos) masked_val else -masked_val;
            }

            // Kembalikan ke format Float (Desimal) untuk lapisan berikutnya
            const out_f = @as(f32, @floatFromInt(sum_fixed)) / 65536.0;
            ctx.buf_expert[out_i] = if (out_f > 0.0) out_f else 0.0; // ReLU Activation
        }
    }

    // 🚀 BUNGKUSAN PEKERJA UNTUK THREAD POOL
    fn expertWorkerPool(wg: *std.Thread.WaitGroup, ctx: WorkerContext) void {
        defer wg.finish();
        expertWorkerSequential(ctx);
    }

    // ==========================================
    // 💾 INISIALISASI & LOAD MODEL
    // ==========================================
    pub fn load(allocator: std.mem.Allocator, path: []const u8) !DualBrainEngine {
        std.debug.print("📥 Memuat arsitektur V8.5 (INT8 + BRANCHLESS SIMD) dari: {s}\n", .{path});
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var engine = DualBrainEngine{
            .allocator = allocator,
            .pool = null,
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

        if (USE_MULTI_THREADING) {
            engine.pool = try allocator.create(std.Thread.Pool);
            try engine.pool.?.init(.{ .allocator = allocator, .n_jobs = NUM_THREADS });
            std.debug.print("⚡ Thread Pool Aktif: {d} Pekerja Tetap disiapkan.\n", .{NUM_THREADS});
        }

        _ = try file.readAll(std.mem.sliceAsBytes(engine.embeddings));
        _ = try file.readAll(std.mem.sliceAsBytes(engine.router_l1));
        _ = try file.readAll(std.mem.sliceAsBytes(engine.router_l2_left));
        _ = try file.readAll(std.mem.sliceAsBytes(engine.router_l2_right));

        for (engine.layers) |*layer| {
            layer.norm1 = try allocator.alloc(f32, HIDDEN_DIM);
            layer.time_decay = try allocator.alloc(f32, HIDDEN_DIM);
            layer.norm2 = try allocator.alloc(f32, HIDDEN_DIM);
            layer.exp_calc = try allocator.alloc(i8, HIDDEN_DIM * HIDDEN_DIM);
            layer.exp_sync = try allocator.alloc(i8, HIDDEN_DIM * HIDDEN_DIM);
            layer.exp_sci = try allocator.alloc(i8, HIDDEN_DIM * HIDDEN_DIM);
            layer.exp_story = try allocator.alloc(i8, HIDDEN_DIM * HIDDEN_DIM);
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
        return engine;
    }

    pub fn deinit(self: *DualBrainEngine) void {
        if (self.pool) |p| {
            p.deinit();
            self.allocator.destroy(p);
        }

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

    // ==========================================
    // 🧠 INFERENSI (GENERASI TEKS)
    // ==========================================
    pub fn generate(self: *DualBrainEngine, start: []const u32, max_tokens: usize, tokenizer: anytype) !void {
        var stdout_buffer: [4096]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
        const stdout = &stdout_writer.interface;
        var timer = try std.time.Timer.start();
        var prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp())));
        const random = prng.random();

        for (self.layers) |*layer| @memset(layer.state, 0.0);
        const last_tok_id = @min(start[start.len - 1], VOCAB_SIZE - 1);
        const router_emb = self.embeddings[last_tok_id * HIDDEN_DIM .. (last_tok_id + 1) * HIDDEN_DIM];

        // 🧭 ROUTER (L1 DISTANCE / ADDERNET)
        var l1 = [2]f32{ 0.0, 0.0 };
        for (0..2) |out_i| {
            var dist: f32 = 0.0;
            const w_row = self.router_l1[out_i * HIDDEN_DIM .. (out_i + 1) * HIDDEN_DIM];
            for (0..HIDDEN_DIM) |in_i| dist += @abs(router_emb[in_i] - w_row[in_i]);
            l1[out_i] = -dist;
        }
        const is_right = l1[1] > l1[0];
        const current_hemi: usize = if (is_right) 1 else 0;

        const l2w = if (is_right) self.router_l2_right else self.router_l2_left;
        var l2 = [2]f32{ 0.0, 0.0 };
        for (0..2) |out_i| {
            var dist: f32 = 0.0;
            const w_row = l2w[out_i * HIDDEN_DIM .. (out_i + 1) * HIDDEN_DIM];
            for (0..HIDDEN_DIM) |in_i| dist += @abs(router_emb[in_i] - w_row[in_i]);
            l2[out_i] = -dist;
        }

        var current_expert: usize = 0;
        if (!is_right) {
            current_expert = if (l2[0] > l2[1]) 0 else 1;
        } else {
            current_expert = if (l2[0] > l2[1]) 2 else 3;
        }

        const active_temp: f32 = if (current_hemi == 0) 0.2 else 0.8;

        if (USE_MULTI_THREADING) {
            try stdout.print("\n   [🔒 EXP {d} | 🌡️ TEMP: {d:.1} | 🌪️ MULTI-THREAD POOL ({d} Cores)]\n   ", .{ current_expert, active_temp, NUM_THREADS });
        } else {
            try stdout.print("\n   [🔒 EXP {d} | 🌡️ TEMP: {d:.1} | 🐌 SINGLE-THREAD SIMD]\n   ", .{ current_expert, active_temp });
        }
        try stdout.flush();

        var next_tok = start[0];
        var i: usize = 0;
        var generated: usize = 0;

        while (true) {
            const tok_id = @min(next_tok, VOCAB_SIZE - 1);
            @memcpy(self.buf_x, self.embeddings[tok_id * HIDDEN_DIM .. (tok_id + 1) * HIDDEN_DIM]);

            for (self.layers) |*layer| {
                rmsnorm(self.buf_norm, self.buf_x, layer.norm1);
                for (0..HIDDEN_DIM) |j| {
                    const decay = layer.time_decay[j];
                    layer.state[j] = (layer.state[j] * decay) + (self.buf_norm[j] * (1.0 - decay));
                    self.buf_res[j] = self.buf_x[j] + layer.state[j];
                }
                rmsnorm(self.buf_norm2, self.buf_res, layer.norm2);

                const exp_w = switch (current_expert) {
                    0 => layer.exp_calc,
                    1 => layer.exp_sync,
                    2 => layer.exp_sci,
                    3 => layer.exp_story,
                    else => unreachable,
                };

                // Siapkan data Q16 untuk digeser bit-nya
                var fixed_x: [HIDDEN_DIM]i32 = undefined;
                for (0..HIDDEN_DIM) |j| {
                    fixed_x[j] = @intFromFloat(self.buf_norm2[j] * 65536.0);
                }

                // 🎚️ EKSEKUSI EXPERT (PARALEL ATAU SEKUENSIAL)
                if (USE_MULTI_THREADING and self.pool != null) {
                    var wg = std.Thread.WaitGroup{};
                    for (0..NUM_THREADS) |t| {
                        const ctx = WorkerContext{
                            .start_idx = t * CHUNK_SIZE,
                            .end_idx = if (t == NUM_THREADS - 1) HIDDEN_DIM else (t + 1) * CHUNK_SIZE,
                            .exp_w = exp_w,
                            .fixed_x = &fixed_x,
                            .buf_expert = self.buf_expert,
                        };
                        wg.start();
                        try self.pool.?.spawn(expertWorkerPool, .{ &wg, ctx });
                    }
                    wg.wait();
                } else {
                    const ctx = WorkerContext{
                        .start_idx = 0,
                        .end_idx = HIDDEN_DIM,
                        .exp_w = exp_w,
                        .fixed_x = &fixed_x,
                        .buf_expert = self.buf_expert,
                    };
                    expertWorkerSequential(ctx);
                }

                for (0..HIDDEN_DIM) |j| self.buf_x[j] += self.buf_expert[j];
            }

            if (i >= start.len - 1) {
                rmsnorm(self.buf_norm, self.buf_x, self.final_norm);
                var max_logit: f32 = -std.math.inf(f32);
                for (0..VOCAB_SIZE) |v| {
                    const logit = math.dot(self.buf_norm, self.lm_head[v * HIDDEN_DIM .. v * HIDDEN_DIM + HIDDEN_DIM]);
                    self.buf_logits[v] = logit;
                    if (logit > max_logit) max_logit = logit;
                }

                var sum_probs: f32 = 0.0;
                for (0..VOCAB_SIZE) |v| {
                    const prob = @exp((self.buf_logits[v] - max_logit) / active_temp);
                    self.buf_logits[v] = prob;
                    sum_probs += prob;
                }

                const r = random.float(f32) * sum_probs;
                var accum: f32 = 0.0;
                var best_tok: u32 = 0;
                for (0..VOCAB_SIZE) |v| {
                    accum += self.buf_logits[v];
                    if (accum >= r) {
                        best_tok = @intCast(v);
                        break;
                    }
                }

                if (best_tok == 2 or best_tok == 0 or generated >= max_tokens) break;
                next_tok = best_tok;
                generated += 1;

                const word = tokenizer.decode(best_tok);
                if (!std.mem.eql(u8, word, "<UNK>") and !std.mem.eql(u8, word, "<PAD>")) {
                    try stdout.print("{s}", .{word});
                    try stdout.flush();
                }
            } else {
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
