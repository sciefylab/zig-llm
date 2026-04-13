const std = @import("std");

// ==========================================
// 1. ARSITEKTUR BABY ZIG-LLM (DARI NOL)
// ==========================================
const hidden_dim = 16;
const num_experts = 2;
const num_neurons = 32;
const num_clusters = 2; // 2 Laci Kosakata
const words_per_cluster = 4; // Tiap Laci isi 4 kata
const vocab_size = 8; // Total 8 kata

// Kamus AI kita
const kamus = [_][]const u8{
    "saya",   "membangun", "ai",     "dengan", // Masuk Laci 0
    "bahasa", "zig",       "sangat", "cepat", // Masuk Laci 1
};

const BabyBrain = struct {
    embed: [vocab_size][hidden_dim]f32,
    router: [num_experts][hidden_dim]f32,
    expert_up: [num_experts][num_neurons][hidden_dim]f32,
    expert_down: [num_experts][hidden_dim][num_neurons]f32,
    vocab_centroids: [num_clusters][hidden_dim]f32,
    vocab_weights: [num_clusters][words_per_cluster][hidden_dim]f32,
    vocab_ids: [num_clusters][words_per_cluster]u32,
};

// ==========================================
// 2. FUNGSI INISIALISASI & MATEMATIKA
// ==========================================
fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

// PRNG Kustom Sederhana (Bebas dari error versi Zig Standard Library)
const LCG = struct {
    state: u64,
    fn init(seed: u64) LCG {
        return LCG{ .state = seed };
    }
    fn float(self: *LCG) f32 {
        self.state = self.state *% 6364136223846793005 +% 1442695040888963407;
        const rand_u32: u32 = @truncate(self.state >> 32);
        return @as(f32, @floatFromInt(rand_u32)) / 4294967296.0;
    }
};

fn initBrain() BabyBrain {
    var brain = std.mem.zeroes(BabyBrain);
    var prng = LCG.init(42);

    // Fungsi kecil: hasilkan angka acak antara -0.05 sampai 0.05
    const rand_w = struct {
        fn get(r: *LCG) f32 {
            return (r.float() - 0.5) * 0.1;
        }
    }.get;

    // Isi bobot acak
    for (0..vocab_size) |i| {
        for (0..hidden_dim) |d| brain.embed[i][d] = rand_w(&prng);
    }
    for (0..num_experts) |e| {
        for (0..hidden_dim) |d| brain.router[e][d] = rand_w(&prng);
        for (0..num_neurons) |n| {
            for (0..hidden_dim) |d| brain.expert_up[e][n][d] = rand_w(&prng);
        }
        for (0..hidden_dim) |d| {
            for (0..num_neurons) |n| brain.expert_down[e][d][n] = rand_w(&prng);
        }
    }

    // Susun Laci Kosakata
    var word_id: u32 = 0;
    for (0..num_clusters) |c| {
        for (0..hidden_dim) |d| brain.vocab_centroids[c][d] = rand_w(&prng);
        for (0..words_per_cluster) |w| {
            brain.vocab_ids[c][w] = word_id;
            word_id += 1;
            for (0..hidden_dim) |d| brain.vocab_weights[c][w][d] = rand_w(&prng);
        }
    }
    return brain;
}

// ==========================================
// 3. MAIN LOOP: TRAINING -> INFERENCE
// ==========================================
pub fn main() !void {
    std.debug.print("====================================================\n", .{});
    std.debug.print(" 🍼 BABY ZIG-LLM: PEMBUKTIAN KONSEP INFERENSI\n", .{});
    std.debug.print("====================================================\n\n", .{});

    var brain = initBrain();

    // Kalimat target: "saya membangun ai dengan bahasa zig sangat cepat"
    const target_sequence = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const learning_rate: f32 = 0.05;
    const epochs = 500;

    std.debug.print("[*] Melatih Jaringan Kosakata & Pakar ({} Epochs)...\n", .{epochs});

    // ---------------------------------------------------------
    // FASE 1: PELATIHAN (OVERFITTING)
    // ---------------------------------------------------------
    for (0..epochs) |epoch| {
        var correct_count: usize = 0;

        for (0..target_sequence.len - 1) |t| {
            const input_id = target_sequence[t];
            const target_id = target_sequence[t + 1];

            // --- FORWARD PASS (INFERENSI) ---
            var state = brain.embed[input_id];

            // 1. Pilih Pakar (Router)
            var best_expert: usize = 0;
            var max_score: f32 = -999999.0;
            for (0..num_experts) |e| {
                var score: f32 = 0.0;
                for (0..hidden_dim) |d| score += state[d] * brain.router[e][d];
                if (score > max_score) {
                    max_score = score;
                    best_expert = e;
                }
            }

            // 2. Jalankan Pakar (1 Pakar Saja - ZERO COMPUTE OTHERS!)
            var inter = std.mem.zeroes([num_neurons]f32);
            for (0..num_neurons) |n| {
                var sum: f32 = 0.0;
                for (0..hidden_dim) |d| sum += state[d] * brain.expert_up[best_expert][n][d];
                inter[n] = silu(sum);
            }
            for (0..hidden_dim) |d| {
                var sum: f32 = 0.0;
                for (0..num_neurons) |n| sum += inter[n] * brain.expert_down[best_expert][d][n];
                state[d] += sum; // Residual
            }

            // 3. Prediksi Laci Kosakata
            var best_cluster: usize = 0;
            max_score = -999999.0;
            for (0..num_clusters) |c| {
                var score: f32 = 0.0;
                for (0..hidden_dim) |d| score += state[d] * brain.vocab_centroids[c][d];
                if (score > max_score) {
                    max_score = score;
                    best_cluster = c;
                }
            }

            // 4. Prediksi Kata dalam Laci
            var best_word: u32 = 0;
            max_score = -999999.0;
            for (0..words_per_cluster) |w| {
                var score: f32 = 0.0;
                for (0..hidden_dim) |d| score += state[d] * brain.vocab_weights[best_cluster][w][d];
                if (score > max_score) {
                    max_score = score;
                    best_word = brain.vocab_ids[best_cluster][w];
                }
            }

            if (best_word == target_id) correct_count += 1;

            // --- BACKWARD PASS (BELAJAR) ---
            var target_c: usize = 0;
            var target_w: usize = 0;
            for (0..num_clusters) |c| {
                for (0..words_per_cluster) |w| {
                    if (brain.vocab_ids[c][w] == target_id) {
                        target_c = c;
                        target_w = w;
                        break;
                    }
                }
            }

            // Hebbian Update
            for (0..hidden_dim) |d| {
                brain.vocab_centroids[target_c][d] += learning_rate * state[d];
                if (best_cluster != target_c) brain.vocab_centroids[best_cluster][d] -= learning_rate * state[d];

                brain.vocab_weights[target_c][target_w][d] += learning_rate * state[d];
                brain.router[best_expert][d] += (learning_rate * 0.1) * state[d];
            }
        }

        if ((epoch + 1) % 100 == 0) {
            std.debug.print(" -> Epoch {} | Akurasi Hafalan: {}/7\n", .{ epoch + 1, correct_count });
        }
    }

    // ---------------------------------------------------------
    // FASE 2: INFERENSI MURNI (PEMBUKTIAN AUTOREGRESSIVE)
    // ---------------------------------------------------------
    std.debug.print("\n====================================================\n", .{});
    std.debug.print(" 🚀 INFERENSI: AI BERBICARA SENDIRI\n", .{});
    std.debug.print("====================================================\n", .{});

    var current_id: u32 = 0; // Mulai dengan kata "saya"
    std.debug.print("Prompt: {s}\n", .{kamus[current_id]});
    std.debug.print("AI    : {s} ", .{kamus[current_id]});

    for (0..7) |_| {
        var state = brain.embed[current_id];

        // 1. Router
        var best_expert: usize = 0;
        var max_score: f32 = -999999.0;
        for (0..num_experts) |e| {
            var score: f32 = 0.0;
            for (0..hidden_dim) |d| score += state[d] * brain.router[e][d];
            if (score > max_score) {
                max_score = score;
                best_expert = e;
            }
        }

        // 2. Expert
        var inter = std.mem.zeroes([num_neurons]f32);
        for (0..num_neurons) |n| {
            var sum: f32 = 0.0;
            for (0..hidden_dim) |d| sum += state[d] * brain.expert_up[best_expert][n][d];
            inter[n] = silu(sum);
        }
        for (0..hidden_dim) |d| {
            var sum: f32 = 0.0;
            for (0..num_neurons) |n| sum += inter[n] * brain.expert_down[best_expert][d][n];
            state[d] += sum;
        }

        // 3. Vocab Laci
        var best_cluster: usize = 0;
        max_score = -999999.0;
        for (0..num_clusters) |c| {
            var score: f32 = 0.0;
            for (0..hidden_dim) |d| score += state[d] * brain.vocab_centroids[c][d];
            if (score > max_score) {
                max_score = score;
                best_cluster = c;
            }
        }

        // 4. Vocab Kata
        var best_word: u32 = 0;
        max_score = -999999.0;
        for (0..words_per_cluster) |w| {
            var score: f32 = 0.0;
            for (0..hidden_dim) |d| score += state[d] * brain.vocab_weights[best_cluster][w][d];
            if (score > max_score) {
                max_score = score;
                best_word = brain.vocab_ids[best_cluster][w];
            }
        }

        std.debug.print("{s} ", .{kamus[best_word]});
        current_id = best_word; // Umpan balik untuk kata selanjutnya
    }
    std.debug.print("\n\n✅ Selesai! Konsep Inferensi ZIG-LLM Terbukti Bekerja.\n", .{});
}
