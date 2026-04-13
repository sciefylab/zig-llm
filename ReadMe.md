# 🧠 ZIG-LLM: Cyber-Dual Brain HMoE

> *"Stop computing everything. Classify, route, and infer."*

---

## 1. The Problem (Background)
Modern Large Language Models (like LLaMA, Qwen, GPT) rely on **Dense Matrix Multiplication** (Brute Force). To predict just 1 next word, the AI must activate billions of parameters in its Feed-Forward Network (FFN), even for simple conjunctions. 

- **The Memory Wall:** This Brute-Force method is devastating for local CPUs due to the bottleneck of moving gigabytes of matrix data from RAM to the CPU for every single token.
- **The Temperature Dilemma:** Traditional models struggle with balancing creativity and accuracy. A *high temperature* makes the AI creative but prone to math errors; a *low temperature* makes it logical but robotic.

## 2. Core Innovation: The Cyber-Dual Brain
Zig-LLM discards the Dense Brute-Force approach. Inspired by biological lateralization, Zig-LLM utilizes a **Hierarchical Mixture-of-Experts (HMoE)** with 2-Level Gating:

### Level 1: The Hemispheres (Lateralization)
A primary router evaluates the prompt and determines the domain:
*   **Sinister (Left Brain):** Handles rigid data, rules, exact math, and certainty.
*   **Dexter (Right Brain):** Handles fluid data, speculation, poetry, and narrative.

### Level 2: The Specialists (Sub-Experts)
Each hemisphere contains isolated experts (Zero Compute for unselected experts):

**Left Brain Department:**
*   **The Calculator:** Operations, algebra, pure logic (True/False).
*   **The Syntactician:** Programming syntax (Zig, C, Python) and grammar rules.

**Right Brain Department:**
*   **The Futurist:** Technological predictions, sci-fi concepts, and theories.
*   **The Storyteller:** Narrative, emotions, metaphors, and AI personification.

## 3. Dynamic Control Mechanism (The Brain Logic)
To solve the Temperature Dilemma, Zig-LLM implements **Adaptive Temperature**. The system calculates the ideal sampling temperature dynamically based on the Router's decision:

$$T_{final} = (W_{left} \cdot T_{exact}) + (W_{right} \cdot T_{creative})$$

*   If $W_{left}$ is dominant $\rightarrow$ Output is cold, rigid, and strictly accurate.
*   If $W_{right}$ is dominant $\rightarrow$ Output is warm, varied, and imaginative.

## 4. Ecosystem Architecture (The Pipeline)

🐍 **The Surgeon (Python):**
An offline script pipeline utilizing Hugging Face & Scikit-Learn to dissect Dense matrices, cluster vocabulary into "Drawers", and export them into a custom highly optimized `.zbrain` binary.

⚡ **The Racecar (Zig):**
A pure, online inference engine. It reads custom binaries using Zero-Copy Memory Mapping and executes the Drawers/Experts using hardware-level SIMD instructions and Pointer Chasing.

## 5. Inference Flow (The Loop)

1.  **Input Vectorizer:** Converts the prompt into initial coordinates.
2.  **Lvl 1 Gating:** Router determines the % involvement of Left vs Right.
3.  **Lvl 2 Gating:** Sub-router selects the most relevant specialist (e.g., The Syntactician).
4.  **HMoE Processing:** Only the selected expert matrix computes the data.
5.  **Temp Calculation:** Calculates ideal temperature based on route weights.
6.  **Clustered Sampling:** Selects the next token from the correct domain vocabulary drawer.
7.  **Auto-Regression:** Feeds the new word back into the KV-Cache.

## 🗺️ MASTER TO-DO LIST (ROADMAP)

### ✅ PHASE 1: The Foundation & The "Final Boss" (COMPLETED 🚀)
*   [x] **Arsitektur Dasar:** Desain HMoE (Router & Experts).
*   [x] **I/O Engine:** Bangun Telinga (Encoder) & Pita Suara (Decoder) O(1) di Zig.
*   [x] **Attention:** Implementasi Causal Self-Attention (Mata Batin Model).
*   [x] **Backprop:** Menulis kalkulus manual untuk Softmax dan matriks Q, K, V.
*   [x] **Stabilisasi:** Implementasi Residual Connections untuk menyembuhkan "Amnesia Identitas" (Halusinasi).
*   [x] **Benchmark:** Loss mencapai 0.0000 pada dataset lokal & kecepatan CPU absolut ~10.000+ Tokens/Detik.

---

### 🔵 PHASE 2: The Real World Scale-Up
*Tujuan: Mengangkat status model dari "Bayi Jenius" (Toy Dataset) menjadi LLM sesungguhnya.*

#### 1. Byte-Pair Encoding (BPE) Tokenizer
- [ ] Ganti ToyTokenizer (Word-level) dengan BPE Sub-word tokenizer.
- [ ] Tulis skrip Python untuk melatih BPE vocab (misal: 10.000 token) dari dataset nyata.
- [ ] Update `tokenizer.zig` agar bisa memecah dan menggabung kata (misal: "rintik" + "nya").

#### 2. Multi-Head Attention (MHA)
- [ ] Upgrade Single-Head ke Multi-Head Attention.
- [ ] Pecah dimensi Q, K, V (contoh: `HIDDEN_DIM` 128 dibagi menjadi 4 Heads @ 32).
- [ ] Implementasi Concatenation hasil 4 Heads sebelum masuk ke Router.

#### 3. Deep Layers & Scaling
- [ ] Arsitektur N-Lapis (Looping `[Attention -> Router -> Expert]` x 4 Layers).
- [ ] Ganti aktivasi ReLU dengan SiLU / SwiGLU.
- [ ] Tambahkan Layer Normalization (RMSNorm) untuk kestabilan training.

#### 4. The "TinyStories" Real Dataset
- [ ] Download dataset "TinyStories".
- [ ] Kompilasi jutaan token ke dalam `.hmoe`.
- [ ] Training hingga Loss konvergen.

---

### 🔴 PHASE 3: Extreme Hardware Optimization (The Racecar)
*Tujuan: Mengubah engine dari "Bisa Jalan" menjadi "Sangat Cepat untuk Skala Besar".*

- [ ] **Real KV-Cache System:** Mencegah kalkulasi ulang Query/Key dari kata masa lalu.
- [ ] **SIMD Vectorization (@Vector):** Rombak MatMul menggunakan `@Vector` (AVX2/AVX-512) untuk paralelisasi level CPU register.
- [ ] **MatMul-Free & Quantization:** Eksplorasi format int8 (Q8) dan Look-Up Tables (LUT) untuk mem-bypass sirkuit Multiplier CPU.

---

## 🚀 How to Run

### Requirements
- **Zig Compiler:** [Download Versi Master/Terbaru](https://ziglang.org/learn/getting-started/)
- **Python 3.10+:** Hanya digunakan untuk pemrosesan/kompilasi dataset awal.

### Build & Inference
Jalankan perintah berikut untuk mengoptimalkan performa melalui *compiler flags*:

```bash
# 1. Melatih Model dari Nol (Training)
zig build run -Doptimize=ReleaseFast -- train-dualbrain

# 2. Menguji Kecerdasan Model (Inference / Generation)
zig build run -Doptimize=ReleaseFast -- infer-dualbrain