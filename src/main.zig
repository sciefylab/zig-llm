const std = @import("std");

// --- IMPORT KONTROLER MODULAR ---
const DualBrainEngine = @import("inference/engine.zig").DualBrainEngine;
const Tokenizer = @import("inference/tokenizer.zig").Tokenizer;

// =================================================================
// ⚙️ KONFIGURASI PATH V7 (BPE SUBWORD ERA)
// =================================================================
const VOCAB_PATH = "models/vocab_v7.json";
const MODEL_PATH = "models/real_dual_brain_v7.zbrain";

const GEN_TOKENS: usize = 128;

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

    if (std.mem.eql(u8, mode, "infer-dualbrain")) {
        try runInferenceDualBrain(allocator);
    } else {
        std.debug.print(
            \\=====================================================
            \\ 🧠 ZIG-LLM: CYBER-DUAL BRAIN HMoE V7 (BPE ERA)
            \\=====================================================
            \\ Mode yang tersedia:
            \\   zig build run -- infer-dualbrain
            \\=====================================================
            \\
        , .{});
    }
}

// =================================================================
// 💡 INFERENCE PIPELINE V7
// =================================================================
fn runInferenceDualBrain(allocator: std.mem.Allocator) !void {
    std.debug.print("================================================\n", .{});
    std.debug.print("||  💡 CYBER-DUAL BRAIN INFERENCE V7 (BPE)    ||\n", .{});
    std.debug.print("================================================\n", .{});

    // 1. Load Tokenizer BPE
    var tokenizer = try Tokenizer.load(allocator, VOCAB_PATH);
    defer tokenizer.deinit();

    std.debug.print("📦 Loading model: {s}\n\n", .{MODEL_PATH});

    // 2. Load Engine
    var engine = try DualBrainEngine.load(allocator, MODEL_PATH);
    defer engine.deinit();

    const prompts = [_][]const u8{
        "if john has 5 apples and buys 3 more , how many",
        "write a python function to calculate the factorial of",
        "photosynthesis is a process used by plants to convert",
        "once upon a time in a magical forest there lived",

        // 🚀 PROMPT JEBAKAN V7: Menguji ketahanan BPE terhadap Typo!
        // Di V6, kata "hapyy", "becuz", dan "finaly" akan langsung menjadi <UNK>.
        // Di V7, Tokenizer akan memotongnya menjadi sub-huruf dan AI tetap memahaminya!
        "lily was very hapyy becuz she finaly found her",
    };

    for (prompts, 0..) |p, idx| {
        std.debug.print("───────────────────────────────────────────────\n", .{});
        std.debug.print("🔹 Prompt #{d}: \"{s}\"\n", .{ idx + 1, p });

        // 3. Encode dengan algoritma BPE
        const tokens = try tokenizer.encode(allocator, p);
        defer allocator.free(tokens);

        std.debug.print("   Tokens: {d} pecahan (subwords)\n", .{tokens.len});
        std.debug.print("   Output: ", .{});

        // 4. Generate Output (Spasi sudah diatur natural oleh Engine V7)
        try engine.generate(tokens, GEN_TOKENS, &tokenizer);
    }
}
