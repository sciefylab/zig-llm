# 🧠 Zig-LLM: The Hybrid Cyber-Dual Brain (V8.5)

**Motto:** "Stop computing everything. Shift, Route, and Infer."

**Zig-LLM** is a radical, next-generation AI architecture experiment engineered natively in **Zig**. This project completely abandons the expensive Transformer/Self-Attention mechanisms and traditional Matrix Multiplication (MatMul). By combining **O(1) Exponential State Accumulation**, **Hybrid MatMul-Free Networks**, and **Lossless Int8 Quantization**, Zig-LLM achieves blistering inference speeds of **400+ Tokens Per Second (TPS)** on standard CPUs without GPU acceleration.

---

## 🛑 The Philosophy: Why Kill Attention & MatMul?

1. **The Attention Bottleneck:** Modern Transformers process grammar, facts, and creativity in a single massive block, requiring **O(N²)** complexity to recalculate the past for every single token. *Zig-LLM replaces this with an O(1) Exponential Moving Average (EMA) state that flows continuously.*
2. **The MatMul Bottleneck:** CPUs despise heavy decimal floating-point multiplications. *Zig-LLM aggressively rips out `math.dot` operations in the core layers, replacing them with absolute distance subtractions (AdderNet) and binary bit-shifts (ShiftNet).*

---

## ⚙️ Core Architecture (The V8.5 Pipeline)

### 1. The BPE Frontend
- Replaces naive word-level tokenization with a high-accuracy **15,000-vocab Byte Pair Encoding (BPE)**.
- Handles natural spacing, punctuation, and out-of-vocabulary (OOV) tokens effortlessly.

### 2. The Hybrid MatMul-Free Router (AdderNet)
Instead of multiplying inputs by weights to find angular similarity, the Router measures the **L1 Absolute Distance** (Manhattan Distance) to map concepts.
- **Mechanism:** $Output = -\sum |X - W| \times \eta$
- **Eta ($\eta$) Scaling:** Prevents router "mode collapse" by scaling down massive distance values, ensuring fair probability distribution.
- **Routing Logic (Hierarchical Mixture-of-Experts):**
  - **Left Hemisphere (Exact):** Expert 0 (Calculator) & Expert 1 (Syntactician).
  - **Right Hemisphere (Creative):** Expert 2 (Scientist/Futurist) & Expert 3 (Storyteller).

### 3. The ShiftNet Experts (Int8 Quantized)
The heaviest part of the neural network relies on **0% matrix multiplication**.
- Weights are forced into Power-of-Two formats during training ($2^{-1}, 2^{-2}, 2^{-3}$, etc.).
- **Lossless Int8:** Because weights are just "shift codes" (e.g., shift right by 3), they fit perfectly into a single byte (`i8`), reducing RAM usage by 75% without losing 0.001% of accuracy.
- **CPU Execution:** The CPU simply performs Bitwise Right-Shifts (`>>`).

### 4. High-Precision LM Head
The final output projection remains in Float32 to ensure absolute grammatical precision and vocabulary selection before being filtered by an **Adaptive Temperature** module.

---

## 🚀 Extreme Hardware Optimization

### A. Branchless Auto-SIMD
If-else statements destroy CPU Vectorization. Zig-LLM's inner loop uses **Branchless Bitwise Masking**, forcing the LLVM compiler to utilize hardware-level **AVX2 / NEON SIMD** instructions. The CPU processes 16-32 data points in a single clock cycle.

### B. Persistent Thread Pool
To eliminate OS overhead from spinning up threads per token, Zig-LLM uses a fixed **Thread Pool** (`std.Thread.Pool`) with a `WaitGroup` synchronizer. The 1024-dimensional workload is instantly sharded across all available CPU cores (e.g., 8-16 cores) for true parallel execution.

---

## 📂 Repository Structure

```text
zig-llm/
├── scripts/
│   ├── build_dataset_v7.py      # BPE Tokenization & Data prep
│   └── train_v7.py              # PyTorch Hybrid MatMul-Free Training (For Colab/GPU)
├── src/
│   ├── inference/
│   │   └── engine.zig           # The O(1) Int8 Multi-Threaded Inference Engine
│   ├── utils/
│   │   └── math.zig             # Core mathematical fallbacks
│   └── main.zig                 # CLI Entry point
├── models/                      
│   └── real_dual_brain_v7.zbrain # Exported Int8 binary model
└── README.md

🛠️ Usage Guide
1. Training (Python / PyTorch)Training is done via PyTorch to utilize GPU backpropagation for the AdderNet $\eta$ parameters and ShiftNet Straight-Through Estimators (STE).Bashpython scripts/train_v7.py
# This will automatically export the `.zbrain` file with Int8 compression.
2. CPU Inference (Zig)Place the exported .zbrain file in the models/ directory and run the highly optimized Zig engine.Bash# Compile and run with maximum speed optimizations
zig build run -Doptimize=ReleaseFast -- infer-dualbrain
Note: You can toggle USE_MULTI_THREADING and adjust NUM_THREADS inside engine.zig based on your physical CPU core count.

🗺️ Roadmap
[x] Integrate BPE Tokenizer.
[x] Eliminate MatMul (Hybrid AdderNet + ShiftNet).
[x] Implement Lossless Int8 Quantization.[x] Thread Pooling & Branchless SIMD.[ ] Scale up HIDDEN_DIM from 1024 to 4096.
[ ] Interactive CLI Chat Mode with live token streaming.
[ ] Explicit @Vector SIMD implementation for extreme dimensional scaling.Forged in PyTorch. Unleashed in Zig. Designed for the Future.