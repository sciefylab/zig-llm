
# 🧠 ZIG-LLM: The Interlingua Cyber-Dual Brain HMoE

> **Motto:** *"Stop computing everything. Translate, Route, and Infer."*

**Visi:** Membangun *Natural Language Compiler* yang memisahkan tata bahasa dari ilmu pengetahuan, mengeliminasi komputasi $O(N^2)$ yang mubazir, dan meniru spesialisasi biologis otak manusia untuk performa CPU yang ekstrem.

---

## 1. Konsep Filosofis (The Problem & The Innovation)

### Masalah di LLM Modern
* **"Sup Matriks"**: Arsitektur Transformer konvensional memproses aturan tata bahasa, struktur kalimat, dan fakta sains di dalam matriks yang sama. 
* **Komputasi Berlebih**: Untuk menebak satu kata, model menggunakan *Self-Attention* yang mengkalkulasi ulang seluruh masa lalu ($O(N^2)$), membakar RAM dan CPU secara eksponensial.
* **Suhu Statis**: Menggunakan *temperature* (suhu) yang statis membuat model sering kali berhalusinasi saat disuruh berhitung, atau terlalu kaku saat disuruh bercerita.

### Solusi Zig-LLM (The Interlingua Method)
Zig-LLM bertindak seperti Kompiler (GCC/Zig Build). Ia menerjemahkan bahasa manusia menjadi **"Sinyal Niat" (IR/Intermediate Representation)** terlebih dahulu. Sinyal ini kemudian dilempar ke inti otak (HMoE) yang murni berisi pakar logika dan fakta, tanpa perlu lagi memikirkan *grammar*. Hasil dari pakar tersebut diterjemahkan kembali ke bahasa manusia di pintu keluar.

---

## 2. Blueprint Arsitektur (The 3-Layer Pipeline)

Aliran data di dalam Zig-LLM terbagi menjadi tiga lapisan fisik dan logis yang terisolasi:

### 🟢 Lapis 1: Translator IN (Frontend / Niat Eksplisit)
Mengubah teks manusia menjadi *Intent Vector* (Sinyal Niat) secara instan.
* **Pembunuh Attention:** Membuang fungsi *Self-Attention* yang lambat. Menggantinya dengan **Mean Pooling**, yaitu merata-ratakan seluruh vektor kata dalam input menjadi satu "Vektor Konteks Global" yang menghemat komputasi menjadi linier $O(N)$.
* **Soft Routing:** Mengizinkan semua teks masuk (tanpa validasi kaku). Matriks Intent akan mendeteksi: *Apakah ini perintah matematika? Pertanyaan fakta? Atau sekadar basa-basi?*

### 🔵 Lapis 2: The Core Engine (Hierarchical Mixture-of-Experts)
Mesin penalaran murni. Vektor Niat dari Lapis 1 dievaluasi oleh sistem 2-Level Gating (Pakar):
* **Level 1: The Hemispheres (Lateralisasi Kiri/Kanan)**
  * **Sinister (Kiri):** Memproses niat eksak, aturan pasti, dan perhitungan matriks absolut.
  * **Dexter (Kanan):** Memproses niat spekulatif, imajinasi, dan pengetahuan umum.
* **Level 2: The Specialists (Sub-Expert) — *Zero Compute* untuk pakar yang mati**
  * **The Calculator (Kiri):** Ahli angka dan logika.
  * **The Syntactician (Kiri):** Ahli pemformatan, perbaikan struktur, dan kode.
  * **The Futurist (Kanan):** Ahli penjelasan fakta sains, prediksi, dan teori.
  * **The Storyteller (Kanan):** Ahli narasi, metafora, dan obrolan kasual.

### 🔴 Lapis 3: Translator OUT (Backend / Juru Bicara)
Menerjemahkan *Vektor Solusi* dari pakar kembali menjadi kalimat manusia (Output).
* **Clustered Vocabulary:** Memiliki "Laci Kata" yang terpisah. Jika Otak Kiri yang menyala, probabilitas keluarnya kata-kata angka, simbol logika, dan bahasa baku akan meroket tajam.

---

## 3. Mekanisme Otak Kognitif (The Brain Logic)

### A. Graceful Degradation (Soft Fallback)
Zig-LLM tidak pernah *crash* atau memarahi pengguna jika inputnya tidak jelas (misal: "Halo", "Kucing").
* Jika Vektor Niat gagal menemukan perintah spesifik, Router L1 & L2 akan secara *default* melempar prompt tersebut ke **Otak Kanan $\rightarrow$ The Storyteller**. 
* Pakar ini dilatih khusus untuk memberikan respons *chit-chat* yang pendek, aman, dan anggun untuk memancing pengguna memperjelas niatnya.

### B. Adaptive Temperature (Suhu Adaptif Dinamis)
Suhu generasi teks (kreativitas) tidak lagi diatur manual oleh pengguna, melainkan dihitung secara matematis oleh model setiap kali Router mengambil keputusan:

$$T_{final} = (W_{left} \cdot T_{exact}) + (W_{right} \cdot T_{creative})$$

* **Skenario A:** Prompt: *"Hitung 50 + 40"*. Router melempar ke Kiri ($W_{left}$ tinggi). Model secara otomatis membekukan suhunya menjadi `0.1` agar jawaban eksak dan tidak berhalusinasi.
* **Skenario B:** Prompt: *"Buatkan puisi"*. Router melempar ke Kanan ($W_{right}$ tinggi). Suhu otomatis naik ke `0.8` agar output kaya akan kosa kata.

---

## 4. ROADMAP & TO-DO LIST (Tahap Eksekusi)

Berikut adalah panduan pengerjaan langkah demi langkah untuk melakukan *upgrade* dari basis kode saat ini menuju arsitektur V3.0:

### 🛠️ Fase 1: Data Mastery & Pipeline (Python)
**Fokus:** Mempersiapkan dataset yang memiliki pemisahan intent dan *fallback*.
- [ ] **Klasifikasi Rute Otomatis:** Modifikasi `hf_to_hmoe.py` untuk secara otomatis memberikan label Header 8-Byte (`hemi_val`, `exp_val`) berdasarkan kata kunci *(Jelaskan, Hitung, Perbaiki)*.
- [ ] **Injeksi Data Soft-Fallback:** Masukkan 5.000+ baris dataset *chit-chat* kasual dan labeli secara absolut ke *Right Brain (Storyteller)* agar AI pandai berbasa-basi saat intent tidak jelas.
- [ ] **Cross-Lingual Blending:** Tarik 20% dataset bahasa Indonesia ke dalam skrip agar matriks tokenizer dan *embedding* selaras antara Inggris dan Indonesia.
- [ ] **Generate `.hmoe`:** Kompilasi dataset baru.

### ⚡ Fase 2: Zig Core Refactoring (The Engine)
**Fokus:** Menghancurkan *bottleneck* $O(N^2)$ dan mengaktifkan Mean Pooling.
- [ ] **Hapus Attention:** Nuke/Hapus seluruh blok fungsi `computeSelfAttention` dari `trainer_dual_brain.zig`.
- [ ] **Bangun Translator IN:** Tulis fungsi `computeIntentContext` yang melakukan penjumlahan & rata-rata (*Mean Pooling*) pada token input, lalu diproyeksikan dengan matriks `intent_weights`.
- [ ] **Hubungkan Ulang Router:** Pastikan input yang dikonsumsi oleh `router_l1_weights` berasal dari hasil akhir `computeIntentContext`.

### 🧠 Fase 3: Logika Inferensi (Zig Main)
**Fokus:** Membuat model "hidup", mandiri, dan responsif.
- [ ] **Clean-up Main:** Pastikan tidak ada validasi blokir kaku di `main.zig`. Semua teks harus bisa masuk ke model.
- [ ] **Implementasi Adaptive Temp:** Pada fungsi `infer()`, suntikkan rumus matematika $T_{final}$ berdasarkan nilai `l1[0]` (Kiri) dan `l1[1]` (Kanan).
- [ ] **State Accumulator:** Di dalam fungsi `generate()`, simpan state vektor hasil pooling agar tidak perlu dihitung ulang dari awal setiap kali token baru ditebak (Sangat menghemat siklus CPU).

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

