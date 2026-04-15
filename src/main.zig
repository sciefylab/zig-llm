const std = @import("std");

// --- IMPORT KONTROLER UTAMA ---
const DualBrainTrainer = @import("training/trainer_dual_brain.zig").DualBrainTrainer;
const DualBrainDataloader = @import("training/train_data.zig").DualBrainDataloader;
const DualBrainEngine = @import("inference/engine.zig").DualBrainEngine;

// =================================================================
// ⚙️ KONFIGURASI PATH
// =================================================================
const DUAL_BRAIN_DATA_PATH = "data/processed/real_dual_brain.hmoe";
const VOCAB_PATH = "data/processed/real_vocab.json";
const MODEL_PATH = "models/real_dual_brain.zbrain";

// =================================================================
// 🔠 TOY TOKENIZER (BPE-ish)
// =================================================================
pub const ToyTokenizer = struct {
    allocator: std.mem.Allocator,
    id_to_word: [][]const u8,

    pub fn init(allocator: std.mem.Allocator, json_path: []const u8) !ToyTokenizer {
        const file = try std.fs.cwd().openFile(json_path, .{});
        defer file.close();

        const buf = try file.readToEndAlloc(allocator, 5 * 1024 * 1024);
        defer allocator.free(buf);

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf, .{});
        defer parsed.deinit();

        const obj = parsed.value.object;
        var max_id: usize = 0;
        var it = obj.iterator();
        while (it.next()) |e| max_id = @max(max_id, @as(usize, @intCast(e.value_ptr.integer)));

        const map = try allocator.alloc([]const u8, max_id + 1);
        @memset(map, "[UNK]");

        it = obj.iterator();
        while (it.next()) |e| {
            map[@as(usize, @intCast(e.value_ptr.integer))] = try allocator.dupe(u8, e.key_ptr.*);
        }
        return ToyTokenizer{ .allocator = allocator, .id_to_word = map };
    }

    pub fn decode(self: *const ToyTokenizer, id: u32) []const u8 {
        return if (id < self.id_to_word.len) self.id_to_word[id] else "[ERR]";
    }

    pub fn deinit(self: *ToyTokenizer) void {
        for (self.id_to_word) |w| {
            if (!std.mem.eql(u8, w, "[UNK]")) self.allocator.free(w);
        }
        self.allocator.free(self.id_to_word);
    }
};

fn encodeForDualBrain(allocator: std.mem.Allocator, tokenizer: *const ToyTokenizer, text: []const u8) ![]u32 {
    var capacity: usize = 16;
    var count: usize = 0;
    var tokens = try allocator.alloc(u32, capacity);
    errdefer allocator.free(tokens);

    var it = std.mem.tokenizeAny(u8, text, " ");
    while (it.next()) |word| {
        if (count >= capacity) {
            capacity *= 2;
            tokens = try allocator.realloc(tokens, capacity);
        }

        var id: u32 = 2; // Default ke <|UNK|>
        for (tokenizer.id_to_word, 0..) |dict_word, i| {
            if (std.mem.eql(u8, dict_word, word)) {
                id = @intCast(i);
                break;
            }
        }
        tokens[count] = id;
        count += 1;
    }
    return allocator.realloc(tokens, count);
}

// =================================================================
// 🚀 MAIN KERNEL
// =================================================================
pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const mode = if (args.len > 1) args[1] else "help";

    // 🔥 BACA ARGUMEN KE-3 UNTUK START EPOCH (Default: 1)
    const start_epoch: usize = if (args.len > 2) std.fmt.parseInt(usize, args[2], 10) catch 1 else 1;

    if (std.mem.eql(u8, mode, "train-dualbrain")) {
        try runTrainingDualBrain(allocator, start_epoch);
    } else if (std.mem.eql(u8, mode, "infer-dualbrain")) {
        try runInferenceDualBrain(allocator);
    } else {
        std.debug.print(
            \\=====================================================
            \\ 🧠 ZIG-LLM: CYBER-DUAL BRAIN HMoE
            \\=====================================================
            \\ Mode yang tersedia:
            \\   zig build run -- train-dualbrain [start_epoch] : Training / Resume
            \\   zig build run -- infer-dualbrain               : Test Generasi
            \\=====================================================
            \\
        , .{});
    }
}

fn runTrainingDualBrain(allocator: std.mem.Allocator, start_epoch: usize) !void {
    std.debug.print("--- 🧠 Memulai Pipeline Training Dual-Brain ---\n", .{});

    var loader = try DualBrainDataloader.init(allocator, DUAL_BRAIN_DATA_PATH);
    defer loader.deinit();
    std.debug.print("📦 Dataset termuat: {d} sequences.\n", .{loader.batches.len});

    const lr: f32 = 0.001;
    var trainer: DualBrainTrainer = undefined;

    var resume_success = false;
    std.fs.cwd().access(MODEL_PATH, .{}) catch |err| {
        if (err != error.FileNotFound) return err;
    };

    const file_exists = blk: {
        std.fs.cwd().access(MODEL_PATH, .{}) catch break :blk false;
        break :blk true;
    };

    if (file_exists) {
        std.debug.print("💾 Checkpoint ditemukan! Melanjutkan training...\n", .{});
        trainer = try DualBrainTrainer.load(allocator, MODEL_PATH, lr);
        resume_success = true;
    } else {
        std.debug.print("🆕 Memulai training dari nol (Bobot Acak)...\n", .{});
        trainer = try DualBrainTrainer.init(allocator, lr);
    }
    defer trainer.deinit();

    var timer = try std.time.Timer.start();

    // 🔥 LOOP MULAI DARI start_epoch
    for (start_epoch..start_epoch + 200) |epoch| {
        loader.reset();
        var batch_count: usize = 0;
        var total_loss: f32 = 0.0;

        while (loader.getNext()) |batch| {
            const loss = try trainer.trainStep(batch);
            total_loss += loss;
            batch_count += 1;

            if (batch_count % 200 == 0) {
                std.debug.print("   [Epoch {d}] Progress: {d} batch... Loss: {d:.4}\r", .{ epoch, batch_count, loss });
            }
        }

        const avg_loss = total_loss / @as(f32, @floatFromInt(batch_count));
        const elapsed_s = @as(f32, @floatFromInt(timer.read())) / 1_000_000_000.0;

        std.debug.print("✅ Epoch {d: >3} | Avg Loss: {d:.6} | Time: {d:.2}s\n", .{ epoch, avg_loss, elapsed_s });

        try trainer.save(MODEL_PATH);
        std.debug.print("   💾 [Checkpoint] Bobot terbaru diamankan.\n", .{});

        timer.reset();
    }
}

fn runInferenceDualBrain(allocator: std.mem.Allocator) !void {
    std.debug.print("--- ⚡ Mode Generasi Cyber-Dual Brain ---\n", .{});

    var tokenizer = try ToyTokenizer.init(allocator, VOCAB_PATH);
    defer tokenizer.deinit();

    var engine = try DualBrainEngine.load(allocator, MODEL_PATH);
    defer engine.deinit();

    // 🔥 PROMPT BARU: In-Distribution (Sesuai Dataset Asli)
    const prompts = [_][]const u8{
        // 1. Memancing Kiri (Orca Math - Kalkulator)
        "calculate the total number of apples if",
        "what is the perimeter of a rectangle with",

        // 2. Memancing Kanan (TinyStories - Storyteller)
        "once upon a time there was a little",
        "lily was very happy because she found a",

        // 3. Memancing Kiri (Code Alpaca - jika ada)
        "write a function in python to",

        // 4. Memancing Fallback / Indonesia (Wikineural)
        "pada tahun 1945 indonesia memproklamasikan",
    };

    for (prompts) |p| {
        std.debug.print("\n🎯 Prompt: \"{s}\"\n", .{p});
        const tokens = try encodeForDualBrain(allocator, &tokenizer, p);
        defer allocator.free(tokens);

        try engine.generate(tokens, 20, &tokenizer);
        std.debug.print("\n", .{});
    }
}
