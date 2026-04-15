# 🧠 Zig-LLM: The Interlingua Cyber-Dual Brain

**Motto:** "Stop computing everything. Translate, Route, and Infer."

Zig-LLM adalah eksperimen arsitektur AI generasi baru yang ditulis **murni dalam bahasa Zig**. Proyek ini secara radikal meninggalkan arsitektur Transformer dan mekanisme Self-Attention yang mahal, menggantinya dengan **Mean Pooling O(N)** dan **State Accumulator O(1)**, sehingga mencapai kecepatan inferensi ekstrem di atas **10.000+ Tokens Per Second** pada CPU biasa.

---

## Filosofi: Mengapa Membunuh Attention?

Arsitektur Transformer modern memproses tata bahasa, fakta, logika, dan kreativitas dalam satu ruang komputasi yang sama. Akibatnya, untuk memprediksi satu token saja, Self-Attention harus menghitung ulang seluruh konteks dengan kompleksitas **O(N²)** — membuang banyak daya komputasi dan memori.

**Zig-LLM** memperlakukan bahasa manusia seperti kompiler:
- Teks manusia diterjemahkan terlebih dahulu menjadi **"Intent Vector"** menggunakan Mean Pooling.
- Vektor ini kemudian dirutekan ke sistem **Hierarchical Mixture-of-Experts (HMoE)** yang berisi spesialisasi berbeda.
- Mesin inferensi tidak pernah menghitung ulang masa lalu — hanya menambahkan makna baru ke dalam **State Accumulator**.

---

## Arsitektur (3-Layer Pipeline)

### 1. Translator IN (Frontend)
- Mengubah teks menjadi **Intent Vector** secara instan.
- Menggunakan **Mean Pooling** (O(N)) sebagai pengganti Self-Attention.
- Dilengkapi **Soft Router** berbasis regex bilingual (Indonesia & Inggris) untuk mendeteksi intent sebelum masuk ke core.

### 2. Core Engine — Hierarchical Mixture-of-Experts (HMoE)

**Level 1 — Hemispheres (Lateralisasi):**
- **Sinister (Kiri)**: Logika eksak, matematika, kode, aturan.
- **Dexter (Kanan)**: Imajinasi, narasi, spekulasi, obrolan kasual.

**Level 2 — Specialists:**
- **The Calculator** (Kiri)
- **The Syntactician** (Kiri)
- **The Futurist** (Kanan)
- **The Storyteller** (Kanan) — juga berfungsi sebagai fallback

### 3. Translator OUT & Telemetry
Menerjemahkan vektor solusi dari para pakar kembali menjadi teks alami dengan proyeksi LM Head yang telah dilewati oleh **Adaptive Temperature**.

---

## Inovasi Utama

### A. State Accumulator O(1)
Tidak ada lagi perhitungan ulang konteks panjang. Zig-LLM hanya menyimpan satu array `pool_sum`. Setiap token baru cukup ditambahkan ke dalam pool tersebut. Kecepatan inferensi token ke-1000 sama cepatnya dengan token pertama.

**Hasil:** ~10.000+ TPS di CPU.

### B. Adaptive Temperature
Suhu tidak lagi diatur manual. Model menghitung sendiri suhu secara dinamis berdasarkan kepercayaan router:

$$
T_{final} = (W_{left} \cdot T_{exact}) + (W_{right} \cdot T_{creative})
$$

- Pertanyaan matematika/logika → suhu otomatis turun (~0.10)
- Permintaan cerita/kreatif → suhu naik (~0.80)

### C. Stable Softmax
Mencegah exploding gradients dan nilai NaN dengan teknik *exponential shifting*:

$$
e^{(x - \max(x))}
$$

---

## Struktur Repository

```bash
zig-llm/
├── scripts/
│   └── preprocessing/
│       └── hf_to_hmoe.py          # Dataset compiler & Soft Router
├── src/
│   ├── training/
│   │   ├── trainer_dual_brain.zig # Training & Backpropagation
│   │   └── train_data.zig
│   ├── inference/
│   │   └── engine.zig             # O(1) Inference Engine
│   └── main.zig
├── models/                            # File .zbrain (biner model)
└── README.md

Cara Penggunaan
1. Persiapan Data
Bash

python scripts/preprocessing/hf_to_hmoe.py
2. Training
Bash

zig build run -Doptimize=ReleaseFast -- train-dualbrain
3. Inference
Bash

zig build run -Doptimize=ReleaseFast -- infer-dualbrain
Roadmap
 Meningkatkan HIDDEN_DIM (64 → 256/512) untuk mengurangi mode collapse
 Menambahkan Repetition Penalty pada LM Head
 Memperkuat Cross-Lingual Dictionary dan oversampling bahasa Indonesia
 Optimasi lebih lanjut + eksperimen arsitektur lanjutan
Built with Zig. Engineered for Speed. Designed for Logic.