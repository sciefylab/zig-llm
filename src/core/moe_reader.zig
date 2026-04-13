const std = @import("std");
const brain = @import("brain_reader.zig");
const math = @import("math.zig");

pub fn forwardHMoE(
    alloc: std.mem.Allocator,
    layer: *const brain.Layer,
    hidden: []f32,
    hidden_dim: usize,
) !void {
    // 1. ROUTING L1: Sinister vs Dexter (Sinister = Logika, Dexter = Imajinasi)
    var l1_logits = [2]f32{ 0.0, 0.0 };
    l1_logits[0] = math.dotProductSIMD(hidden, layer.router.l1_hemisphere[0..hidden_dim]);
    l1_logits[1] = math.dotProductSIMD(hidden, layer.router.l1_hemisphere[hidden_dim..]);

    const is_dexter = l1_logits[1] > l1_logits[0];

    // 2. ROUTING L2: Specialist Selection
    const l2_weights = if (is_dexter) layer.router.l2_specialist_right else layer.router.l2_specialist_left;
    var l2_logits = [2]f32{ 0.0, 0.0 };
    l2_logits[0] = math.dotProductSIMD(hidden, l2_weights[0..hidden_dim]);
    l2_logits[1] = math.dotProductSIMD(hidden, l2_weights[hidden_dim..]);

    const use_specialist_1 = l2_logits[1] > l2_logits[0];

    // 3. INDEX MAPPING (Logika "Stop Computing Everything")
    // Memilih 1 dari 4 Expert Utama berdasarkan keputusan Router
    var expert_idx: usize = 0;
    if (!is_dexter) {
        expert_idx = if (use_specialist_1) 1 else 0; // 0: Calculator, 1: Syntactician
    } else {
        expert_idx = if (use_specialist_1) 3 else 2; // 2: Futurist, 3: Storyteller
    }

    // 4. EKSEKUSI PAKAR TERPILIH (Surgical Execution)
    const expert = layer.experts[expert_idx];

    // Memanggil fungsi computeExpertFFN milikmu yang sudah pakai SIMD & SiLU
    try math.computeExpertFFN(alloc, hidden, expert.gate, expert.up, expert.down, expert.num_neurons, hidden_dim);
}
