const std = @import("std");

// --- IMPORT KONTROLER UTAMA ---
const DualBrainTrainer = @import("training/trainer_dual_brain.zig").DualBrainTrainer;
const DualBrainDataloader = @import("training/train_data.zig").DualBrainDataloader;
const DualBrainEngine = @import("inference/engine.zig").DualBrainEngine;

// =================================================================
// ⚙️ KONFIGURASI PATH V6 (DEEP HMOE ERA)
// =================================================================
const DUAL_BRAIN_DATA_PATH = "data/processed/real_dual_brain.hmoe"; // (Untuk legacy Zig training)
const VOCAB_PATH = "models/vocab_v5.json"; // Vocab tetap sama (bersih dari tanda baca)
const MODEL_PATH = "models/real_dual_brain_v6.zbrain"; // 🚀 Path ke model V6
const BEST_MODEL_PATH = "models/real_dual_brain_v6_best.zbrain";

// 🔥 HYPERPARAMETERS (Legacy untuk fitur Train Zig)
const LEARNING_RATE: f32 = 0.001;
const LR_WARMUP_EPOCHS: usize = 5;
const LR_DECAY_FACTOR: f32 = 0.995;
const DEFAULT_NUM_EPOCHS: usize = 200;
const LOG_INTERVAL: usize = 200;
const GEN_TOKENS: usize = 128;
const ACCUMULATION_STEPS: usize = 32;

// =================================================================
// 🔠 TOY TOKENIZER (100% LEAK-PROOF DENGAN ARENA ALLOCATOR)
// =================================================================
pub const ToyTokenizer = struct {
    arena: std.heap.ArenaAllocator,
    id_to_word: [][]const u8,
    word_to_id: std.StringHashMap(u32),

    pub fn init(parent_allocator: std.mem.Allocator, json_path: []const u8) !ToyTokenizer {
        var arena = std.heap.ArenaAllocator.init(parent_allocator);
        errdefer arena.deinit();
        const allocator = arena.allocator();

        const file = try std.fs.cwd().openFile(json_path, .{});
        defer file.close();

        const buf = try file.readToEndAlloc(allocator, 5 * 1024 * 1024);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf, .{});

        const obj = parsed.value.object;
        var max_id: usize = 0;
        var it = obj.iterator();
        while (it.next()) |e| {
            max_id = @max(max_id, @as(usize, @intCast(e.value_ptr.integer)));
        }

        const map = try allocator.alloc([]const u8, max_id + 1);
        @memset(map, "<UNK>");

        var word_map = std.StringHashMap(u32).init(allocator);
        try word_map.ensureTotalCapacity(@intCast(obj.count()));

        it = obj.iterator();
        while (it.next()) |e| {
            const id: u32 = @intCast(e.value_ptr.integer);
            const word_dup = try allocator.dupe(u8, e.key_ptr.*);
            map[id] = word_dup;
            try word_map.put(word_dup, id);
        }

        return ToyTokenizer{
            .arena = arena,
            .id_to_word = map,
            .word_to_id = word_map,
        };
    }

    pub fn decode(self: *const ToyTokenizer, id: u32) []const u8 {
        return if (id < self.id_to_word.len) self.id_to_word[id] else "[ERR]";
    }

    pub fn encodeWord(self: *const ToyTokenizer, word: []const u8) u32 {
        return self.word_to_id.get(word) orelse 1;
    }

    pub fn deinit(self: *ToyTokenizer) void {
        self.arena.deinit();
    }
};

// =================================================================
// 🚀 OPTIMIZED ENCODER (O(N)) + UNK Diagnostic
// =================================================================
fn encodeForDualBrain(
    allocator: std.mem.Allocator,
    tokenizer: *const ToyTokenizer,
    text: []const u8,
) ![]const u32 {
    var word_count: usize = 0;
    var it_count = std.mem.tokenizeAny(u8, text, " \t\n");
    while (it_count.next()) |_| word_count += 1;

    const tokens = try allocator.alloc(u32, word_count);
    errdefer allocator.free(tokens);

    var idx: usize = 0;
    var unk_count: usize = 0;

    var it = std.mem.tokenizeAny(u8, text, " \t\n");
    while (it.next()) |word| {
        tokens[idx] = tokenizer.encodeWord(word);
        if (tokens[idx] == 1) unk_count += 1;
        idx += 1;
    }

    if (unk_count > 0) {
        std.debug.print("   ⚠️  {d}/{d} tokens are <UNK>\n", .{ unk_count, word_count });
    }

    return tokens;
}

// =================================================================
// 🛠️ HELPER FUNCTIONS
// =================================================================
fn fileExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn formatTime(seconds: f32) struct { h: u32, m: u32, s: u32 } {
    const total: u32 = @intFromFloat(seconds);
    return .{
        .h = total / 3600,
        .m = (total % 3600) / 60,
        .s = total % 60,
    };
}

fn lrForEpoch(epoch: usize) f32 {
    if (epoch <= LR_WARMUP_EPOCHS) return LEARNING_RATE;

    var lr = LEARNING_RATE;
    var i: usize = 0;
    const decay_steps = epoch - LR_WARMUP_EPOCHS;
    while (i < decay_steps) : (i += 1) {
        lr *= LR_DECAY_FACTOR;
    }
    return lr;
}

// =================================================================
// 🚀 MAIN KERNEL
// =================================================================
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{
        .safety = true,
    }){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) std.debug.print("⚠️  Memory leak detected!\n", .{});
    }
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const mode = if (args.len > 1) args[1] else "help";

    const start_epoch: usize = if (args.len > 2)
        std.fmt.parseInt(usize, args[2], 10) catch 1
    else
        1;

    const num_epochs: usize = if (args.len > 3)
        std.fmt.parseInt(usize, args[3], 10) catch DEFAULT_NUM_EPOCHS
    else
        DEFAULT_NUM_EPOCHS;

    if (std.mem.eql(u8, mode, "train-dualbrain")) {
        try runTrainingDualBrain(allocator, start_epoch, num_epochs);
    } else if (std.mem.eql(u8, mode, "infer-dualbrain")) {
        try runInferenceDualBrain(allocator);
    } else {
        std.debug.print(
            \\=====================================================
            \\ 🧠 ZIG-LLM: CYBER-DUAL BRAIN HMoE V6 (DEEP LAYER)
            \\=====================================================
            \\ Mode yang tersedia:
            \\   zig build run -- infer-dualbrain
            \\   zig build run -- train-dualbrain [start_epoch] [num_epochs] 
            \\=====================================================
            \\
        , .{});
    }
}

// =================================================================
// 🧠 TRAINING PIPELINE (Backward Compatibility - Pytorch Only Now)
// =================================================================
fn runTrainingDualBrain(
    allocator: std.mem.Allocator,
    start_epoch: usize,
    num_epochs: usize,
) !void {
    std.debug.print("╔══════════════════════════════════════════════╗\n", .{});
    std.debug.print("║  ⚠️ PERINGATAN: FITUR INI TIDAK KOMPATIBEL   ║\n", .{});
    std.debug.print("║  DENGAN ARSITEKTUR V6. GUNAKAN PYTORCH COLAB ║\n", .{});
    std.debug.print("╚══════════════════════════════════════════════╝\n", .{});

    // Saya memotong isi fungsi ini karena struktur 'trainer' Zig Anda
    // belum diperbarui untuk mendukung NUM_LAYERS=2. Jika dieksekusi
    // akan menyebabkan error dimensi matriks. Latih di Colab!
    _ = allocator;
    _ = start_epoch;
    _ = num_epochs;
}

// =================================================================
// 💡 INFERENCE PIPELINE V6 (DEEP STATE SPACE)
// =================================================================
fn runInferenceDualBrain(allocator: std.mem.Allocator) !void {
    std.debug.print("================================================\n", .{});
    std.debug.print("||  💡 CYBER-DUAL BRAIN INFERENCE V6 (DEEP)   ||\n", .{});
    std.debug.print("================================================\n", .{});

    var tokenizer = try ToyTokenizer.init(allocator, VOCAB_PATH);
    defer tokenizer.deinit();

    const model_path = MODEL_PATH;
    std.debug.print("📦 Loading model: {s}\n\n", .{model_path});

    var engine = try DualBrainEngine.load(allocator, model_path);
    defer engine.deinit();

    // Dataset ujicoba presisi tinggi untuk 4 Expert
    const prompts = [_][]const u8{
        // 🧮 Memicu Expert 0 (Math / Matematika)
        "if john has 5 apples and buys 3 more , how many",
        "to calculate the area of a circle you need to",

        // 💻 Memicu Expert 1 (Code / Python)
        "write a python function to calculate the factorial of",
        "def is_prime ( n ) : if n < 2",

        // 🔬 Memicu Expert 2 (Science / Sains)
        "the earth revolves around the sun because of the",
        "photosynthesis is a process used by plants to convert",

        // 📖 Memicu Expert 3 (Story / Cerita Anak)
        "once upon a time in a magical forest there lived",
        "lily was very happy because she finally found her lost",
    };

    for (prompts, 0..) |p, idx| {
        std.debug.print("───────────────────────────────────────────────\n", .{});
        std.debug.print("🔹 Prompt #{d}: \"{s}\"\n", .{ idx + 1, p });

        // Encode prompt
        const tokens = try encodeForDualBrain(allocator, &tokenizer, p);
        defer allocator.free(tokens);

        std.debug.print("   Tokens: {d} words\n", .{tokens.len});
        std.debug.print("   Output: ", .{});

        // Eksekusi engine V6 dengan Dynamic Temperature
        try engine.generate(tokens, GEN_TOKENS, &tokenizer);
    }
}
