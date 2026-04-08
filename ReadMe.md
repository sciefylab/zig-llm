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

# 🗺️ MASTER TO-DO LIST (ROADMAP) - THE NEXT FRONTIER

## ✅ PHASE 0 - 3: The Engine & Architecture (SELESAI! 🚀)

- [x] Ekstraksi Matriks Kosakata (Qwen 0.5B) & Clustering 151,936 token ke 64 Laci Semantic
- [x] Membedah matriks Dense MLP/FFN menjadi 8 "Pakar" (MoE) terisolasi
- [x] Desain Format Biner Custom `.zbrain`: Arsip 2.35 GB penyatu organ AI
  - ⚡ Zero-Copy Loader dalam ~500 ms
- [x] Bangun Telinga (Encoder) & Pita Suara (Decoder) O(1) di Zig
- [x] Implementasi SIMD AVX Router Inference & Scaled Dot-Product Attention dengan KV Cache
- [x] **Pencapaian Puncak: Loop Autoregressive mencapai 121.6 Token/Detik murni di CPU lokal** 🎯

---

## 🔵 PHASE 4: The "One-Click" Training Studio (Dense-to-MoE Upcycling)
### → KITA MULAI DI SINI

**Tujuan:** Menyembuhkan "Halusinasi Alien" AI. Mengajarkan model Qwen agar terbiasa dengan arsitektur 8 Pakar (MoE) dan 64 Laci Kosakata buatan kita menggunakan teknik **Knowledge Distillation** (Guru-Murid).

### 📋 Tasks:

#### 1. Persiapan Dataset
- [ ] Buat skrip Python untuk mengunduh dataset teks berkualitas tinggi
  - Dataset kandidat: Wikitext, Fine-Web, atau dataset kode
  - Preprocessing & tokenization
  - Data loader untuk training pipeline

#### 2. Tulis `upcycle_moe.py` (PyTorch)
- [ ] Bangun arsitektur **Student** (Murid):
  - Model dengan Router MoE (mirip struktur `.zbrain`)
  - 8 Expert modules
  - 64-way hierarchical output layer
- [ ] Load arsitektur **Teacher** (Guru):
  - Model Qwen asli (Dense)
  - Freeze weights (inference-only mode)
- [ ] Implementasi Knowledge Distillation framework

#### 3. Custom Loss Function
- [ ] Tulis fungsi Loss di PyTorch yang menghukum Murid jika:
  - **Output Distillation Loss**: 
    - MSE Loss / KL Divergence antara logits Student vs Teacher
  - **Load Balancing Loss**: 
    - Router terlalu sering memilih Pakar yang sama
    - Enforce distribusi merata ke-8 Pakar
  - **Auxiliary Loss**: 
    - Regularisasi untuk mencegah overfitting

#### 4. Hierarchical Softmax Training
- [ ] Latih ulang lapisan Output (Pohon Kosakata):
  - Stage 1: Prediksi "Laci Kategori" (64 drawers)
  - Stage 2: Prediksi "Kata" dalam laci
  - Implementasi two-stage softmax
- [ ] Evaluasi akurasi hierarchical prediction

#### 5. The Exporter
- [ ] Bangun `export_zbrain_v2.py`:
  - Ekstrak trained weights dari PyTorch model
  - Convert ke format `.zbrain v2`
  - Validasi dimensi & integrity check
- [ ] Automated testing: Load di Zig engine & verify outputs

---

## 🔴 PHASE 5: Extreme Hardware Optimization (Quantization & MatMul-Free)

**Tujuan:** Memampatkan ukuran `.zbrain` dari 2.35 GB → ~600 MB, dan secara radikal membuang sirkuit perkalian (Multiplier) dari CPU saat Inference.

### 📋 Tasks:

#### 1. Q8_0 Binarization
- [ ] Modifikasi `surgeon_compiler.py`:
  - Convert semua matriks Float32 (FP32) → Integer 8-bit (int8)
  - Hitung Scale Factor per tensor (Float16)
  - Implementasi symmetric/asymmetric quantization
- [ ] Validasi quantization error (< 1% degradasi)

#### 2. Desain `.zbrain v3`
- [ ] Update struktur header biner:
  - Magic bytes untuk v3
  - Metadata quantization (scale factors, zero points)
  - Support mixed precision (Q8/Q4/FP16)
- [ ] Dokumentasi format specification

#### 3. MatMul-Free Math (`math_cpu.zig`)
- [ ] Tulis ulang `dotProductSIMD` di Zig:
  - **Bitwise Shift-Add**: 
    - Approximasi perkalian dengan shift & penjumlahan
    - Implementasi untuk int8 × int8
  - **In-Register Look-Up Tables (LUT)**:
    - Gunakan instruksi `pshufb` (AVX2)
    - Precompute partial products
    - Zero multiplication operations
- [ ] Benchmark: FP32 vs Q8 vs MatMul-Free
  - Throughput (tokens/sec)
  - Latency per layer
  - Memory bandwidth

#### 4. Multithreading (Opsional)
- [ ] Integrasikan `std.Thread.Pool` di Zig:
  - Paralel Multi-Head Attention (Q, K, V computation)
  - Paralel Expert processing dalam MoE
  - Work stealing scheduler
- [ ] NUMA-aware memory allocation (untuk multi-socket systems)
- [ ] Benchmark scaling: 1 core → 4 cores → 8+ cores

---

## 📊 Performance Targets

| Metric | Phase 3 (Current) | Phase 4 (Target) | Phase 5 (Target) |
|--------|-------------------|------------------|------------------|
| Throughput | 121.6 T/s | 100+ T/s (acceptable degradation) | 150+ T/s |
| Model Size | 2.35 GB | 2.35 GB | ~600 MB |
| Memory Usage | ~3 GB | ~3 GB | ~800 MB |
| Perplexity | N/A (no training) | < 15 (post-distillation) | < 16 (post-quant) |
| Multiplications | Full FP32 | Full FP32 | **Zero** ✨ |

---

## 🎯 Success Criteria

### Phase 4:
- ✅ Model dapat generate teks koheren (tidak random/alien)
- ✅ Router load distribution: setiap expert digunakan 10-15% waktu
- ✅ Validation loss turun minimal 80% dari initial

### Phase 5:
- ✅ File size < 700 MB
- ✅ Inference throughput > 100 T/s (dengan Q8)
- ✅ MatMul-free mode functional (meski slower dari SIMD FP32)
- ✅ Perplexity degradation < 10% vs unquantized

---

## 💡 Research Notes

**Phase 4 Challenges:**
- Balancing distillation loss vs load balancing loss
- Preventing mode collapse (all tokens → 1 expert)
- Hierarchical softmax convergence

**Phase 5 Challenges:**
- Quantization-aware training might be needed
- LUT size explosion (trade memory for speed)
- Integer overflow handling in shift-add

---

**Next Immediate Step:** 
📥 Setup dataset download script (`prepare_dataset.py`)

## 🚀 Cara Menjalankan

### Persyaratan
* [Zig Compiler](https://ziglang.org/download/) (Versi terbaru/Master)
* Python 3.10+ (untuk script Surgeon)

### Build
Untuk performa maksimal (mengaktifkan optimasi compiler):
```bash
zig build run -Doptimize=ReleaseFast