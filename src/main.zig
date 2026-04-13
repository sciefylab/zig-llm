const std = @import("std");

// --- IMPORT ZIG-LLM CORE ---
const DualBrainTrainer = @import("training/trainer_dual_brain.zig").DualBrainTrainer;
const DualBrainDataloader = @import("training/train_data.zig").DualBrainDataloader;

// =================================================================
// ⚙️ KONFIGURASI PATH
// =================================================================
const DUAL_BRAIN_DATA_PATH = "data/processed/toy_dual_brain.hmoe";
const TOY_VOCAB_PATH = "data/processed/toy_vocab.json";
const TOY_MODEL_PATH = "models/toy_dual_brain.zbrain";

// =================================================================
// 🔠 TOKENIZER
// =================================================================
pub const ToyTokenizer = struct {
    allocator: std.mem.Allocator,
    id_to_word: [][]const u8,

    pub fn init(allocator: std.mem.Allocator, json_path: []const u8) !ToyTokenizer {
        const file = try std.fs.cwd().openFile(json_path, .{});
        defer file.close();
        const buf = try file.readToEndAlloc(allocator, 2 * 1024 * 1024);
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

        var id: u32 = 0;
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
// 🚀 MAIN KERNEL: ZIG-LLM
// =================================================================
pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const mode = if (args.len > 1) args[1] else "help";

    if (std.mem.eql(u8, mode, "train-dualbrain")) {
        try runTrainingDualBrain(allocator);
    } else if (std.mem.eql(u8, mode, "infer-dualbrain")) {
        try runInferenceDualBrain(allocator);
    } else {
        std.debug.print(
            \\=====================================================
            \\ 🧠 ZIG-LLM: CYBER-DUAL BRAIN HMoE (FROM SCRATCH)
            \\=====================================================
            \\ Mode yang tersedia:
            \\   zig build run -- train-dualbrain   : Memulai training
            \\   zig build run -- infer-dualbrain   : Mode generasi
            \\=====================================================
            \\
        , .{});
    }
}

fn runTrainingDualBrain(allocator: std.mem.Allocator) !void {
    std.debug.print("--- 🧠 Memulai Training Dual-Brain HMoE ---\n", .{});

    var loader = try DualBrainDataloader.init(allocator, DUAL_BRAIN_DATA_PATH);
    defer loader.deinit();

    var trainer = try DualBrainTrainer.init(allocator, 0.01);
    defer trainer.deinit();

    for (1..101) |epoch| {
        loader.reset();
        var batch_count: usize = 0;
        var total_loss: f32 = 0.0;

        while (loader.getNext()) |batch| {
            total_loss += try trainer.trainStep(batch);
            batch_count += 1;
        }

        if (epoch % 10 == 0) {
            const avg_loss = total_loss / @as(f32, @floatFromInt(batch_count));
            std.debug.print("✅ Epoch {d} | Batches: {d} | Loss: {d:.4}\n", .{ epoch, batch_count, avg_loss });
        }
    }

    try trainer.save(TOY_MODEL_PATH);
    std.debug.print("💾 Cyber-Brain Berhasil Disimpan ke: {s}\n", .{TOY_MODEL_PATH});
}

fn runInferenceDualBrain(allocator: std.mem.Allocator) !void {
    std.debug.print("--- ⚡ Memulai Inference Cyber-Dual Brain ---\n", .{});

    var tokenizer = try ToyTokenizer.init(allocator, TOY_VOCAB_PATH);
    defer tokenizer.deinit();

    var brain_instance = try DualBrainTrainer.load(allocator, TOY_MODEL_PATH);
    defer brain_instance.deinit();

    const prompts = [_][]const u8{
        "di bawah hujan rintik robot itu",
        "const flag: bool =",
        "9 tambah 9 sama dengan",
    };

    for (prompts) |p| {
        std.debug.print("\n🎯 Prompt: \"{s}\"\n", .{p});
        const tokens = try encodeForDualBrain(allocator, &tokenizer, p);
        defer allocator.free(tokens);
        try brain_instance.generate(tokens, 15, &tokenizer);
    }

    std.debug.print("\n✅ Eksekusi Selesai.\n", .{});
}
