🧠 ZIG-LLM: Hierarchical CPU-Native Inference Engine
Motto: "Stop computing everything. Classify, route, and infer."

1. The Problem (Background)
Modern Large Language Models (like LLaMA, Qwen, GPT) are built on Dense Matrix Multiplication (Brute Force) architectures:

To predict just 1 next word, the AI must calculate the probabilities of all 150,000 vocabulary tokens simultaneously.
To process a single context word, the AI must activate all billions of parameters in its Feed-Forward Network (FFN), even if the word is a simple conjunction like "and".
This Brute-Force method is highly efficient for GPUs (which have thousands of simple cores). However, it is devastating for local CPUs, as the CPU gets bottlenecked by the "Memory Wall" (the need to move gigabytes of matrix data from RAM to the CPU for every single token).

2. Core Innovation (The Concept)
Zig-LLM completely discards the Dense Brute-Force approach. Inspired by the human brain and Sparse Mixture-of-Experts (MoE) architectures, Zig-LLM utilizes Conditional Computation:

Output Classification (Vocabulary Tree):
Instead of a flat array, the vocabulary is clustered into semantic "Drawers" using K-Means Clustering. The AI first predicts the correct Category Drawer (e.g., the "Programming" drawer), and then only calculates the token probabilities inside that specific drawer. This reduces the computational load from 150,000 MatMuls to roughly ~124 MatMuls.
Internal Routing (Classified Learning):
The extremely heavy Feed-Forward Network (FFN) layers are surgically dismantled into tiny "Experts" (e.g., Coding Expert, Logic Expert, Grammar Expert). When a token enters, a fast Router written in Zig classifies the token and activates only 1 relevant Expert, leaving millions of other parameters uncomputed and resting in RAM.
3. Ecosystem Architecture (The Pipeline)
This project stands on two complementary programming languages:

🐍 The Surgeon (Python): An offline script pipeline utilizing Hugging Face & Scikit-Learn to surgically dissect Dense matrices (.gguf / .safetensors), cluster the neurons, and export them into a custom, highly optimized binary architecture.
⚡ The Racecar (Zig): A pure, online inference engine. It reads the custom binaries using Zero-Copy Memory Mapping (loading 500MB in < 0.1 seconds) and executes the Drawers/Experts using hardware-level AVX SIMD instructions and Pointer Chasing.
4. Ultimate Goals
CPU Dominance: Run 1.5B - 7B parameter models blazingly fast using purely local PC CPUs (No GPU required).
Ultra-Low Memory Bandwidth: Read the absolute minimum amount of data from RAM during the Generation phase to solve the Decoding Bottleneck.
MatMul-Free Architecture: In the final stages, replace drawer and expert calculations with pure addition (Shift-Add / In-Register LUTs), entirely bypassing the CPU's multiplier circuits.


## 🗺️ MASTER TO-DO LIST (ROADMAP) - FINAL VISION

### ✅ PHASE 0 & 1: I/O, Senses, & Proof of Concept (Selesai!)
- [x] Ekstraksi Matriks Kosakata & Clustering 151.936 kata ke 64 Laci Semantic (K-Means).
- [x] Desain Format Biner Custom (Pohon Kosakata & Kamus O(1)).
- [x] Bangun Telinga AI (*Encoder* Byte-Pair) & Pita Suara (*Decoder*) di Zig.

### ✅ PHASE 2: The Master Architecture & MoE Surgery (Selesai!)
- [x] Membedah matriks Dense `MLP/FFN` Qwen menjadi 8 "Pakar" (Experts) terisolasi.
- [x] Ekstrak lapisan *Self-Attention* (Matriks Q, K, V, O).
- [x] Desain **`.zbrain`**: Format file raksasa 2.35 GB penyatu seluruh organ AI.
- [x] Bangun `brain_reader.zig`: Zero-Copy Loader 2.35 GB dalam **~500 milidetik**.

### ✅ PHASE 3: Memory, Context, & The Math (Selesai!)
- [x] Tulis `kv_cache.zig`: Sistem memori jangka pendek (*Key-Value Cache*).
- [x] Tulis `attention.zig`: Rumus murni *Scaled Dot-Product Attention* (Softmax $Q \times K^T \times V$) & RoPE.
- [x] Implementasi *Autoregressive Generate Loop*: Loop AI stabil mencapai kecepatan **118 Token/Detik** murni di CPU lokal.

### 🔵 PHASE 4: The "One-Click" Training Studio (Dense-to-MoE Upcycling) -> *TARGET UTAMA SELANJUTNYA*
*Visi: Menyediakan End-to-End Pipeline di mana pengguna cukup memasukkan ID Hugging Face, dan skrip akan otomatis melatih ulang model tersebut menjadi `.zbrain` yang cerdas dan sangat cepat untuk CPU.*
- [ ] Buat folder `scripts/train/` di dalam repositori.
- [ ] Tulis `upcycle_moe.py` (Python/PyTorch): Skrip *Knowledge Distillation* yang memaksa arsitektur 8 Pakar (MoE) dan 64 Laci Kosakata buatan kita untuk meniru kecerdasan model aslinya (Qwen/LLaMA) menggunakan dataset teks publik.
- [ ] Integrasikan *Export Pipeline* agar setelah proses *training* selesai, skrip langsung memuntahkan file `.zbrain` versi final.

### 🔴 PHASE 5: Extreme Hardware Optimization (MatMul-Free)
- [ ] *Quantization*: Modifikasi *pipeline* ekspor untuk mengubah matriks Float32 (FP32) menjadi Integer 8-bit (Q8_0) atau 4-bit (Q4) guna memangkas ukuran 2.35 GB menjadi ~600 MB.
- [ ] Tulis `math_cpu.zig`: Implementasi instruksi perangkat keras murni (*Bitwise Shift-Add* atau *Look-Up Tables*) untuk menghapus operator `*` (perkalian) secara total dari seluruh sistem inferensi.

## 🚀 Cara Menjalankan

### Persyaratan
* [Zig Compiler](https://ziglang.org/download/) (Versi terbaru/Master)
* Python 3.10+ (untuk script Surgeon)

### Build
Untuk performa maksimal (mengaktifkan optimasi compiler):
```bash
zig build run -Doptimize=ReleaseFast