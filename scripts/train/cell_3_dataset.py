import os
import json
import re
import torch
from collections import Counter
from datasets import load_dataset
from tqdm.auto import tqdm

# ==========================================
# ⚙️ KONFIGURASI DATASET V5 (STATE SPACE)
# ==========================================
DRIVE_BASE = "/content/drive/MyDrive/Dual-Brain"
DRIVE_DATA = os.path.join(DRIVE_BASE, "data")
os.makedirs(DRIVE_DATA, exist_ok=True)

VOCAB_FILE = os.path.join(DRIVE_DATA, "vocab_v5.json")
DATASET_FILE = os.path.join(DRIVE_DATA, "dataset_v5.pt")

# 🚀 THE UPGRADE: Kita paksa AI mengingat 32 kata ke belakang (Bukan 8 lagi!)
# Karena ini State Space Model, RAM GPU tidak akan meledak meski ini dinaikkan.
SEQ_LEN = 32
MAX_VOCAB = 15000
MAX_SAMPLES_PER_DOMAIN = 10000

SOURCES = [
    # Hemisphere 0 (KIRI / EXACT)
    {"repo": "microsoft/orca-math-word-problems-200k", "split": "train", "key_in": "question", "key_out": "answer", "exp": 0, "hemi": 0, "name": "Math"},
    {"repo": "iamtarun/python_code_instructions_18k_alpaca", "split": "train", "key_in": "instruction", "key_out": "output", "exp": 1, "hemi": 0, "name": "Code"},
    # Hemisphere 1 (KANAN / IMAGINASI)
    {"repo": "sciq", "split": "train", "key_in": "question", "key_out": "support", "exp": 2, "hemi": 1, "name": "Science"},
    {"repo": "roneneldan/TinyStories", "split": "train", "key_in": "text", "key_out": None, "exp": 3, "hemi": 1, "name": "Story"},
]

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"([.,!?;:\"'()\[\]\+\-\*/=<>])", r" \1 ", text)
    return re.sub(r'\s+', ' ', text).strip().split()

def main():
    print("🔍 TAHAP 1: Mengunduh data dan menghitung kosa kata...")
    word_counter = Counter()
    raw_data = []

    for src in SOURCES:
        print(f"   -> Memproses domain: {src['name']}...")
        ds = load_dataset(src["repo"], split=src["split"], streaming=True)
        count = 0

        for entry in ds:
            if count >= MAX_SAMPLES_PER_DOMAIN: break

            if src["key_out"] is not None:
                words = clean_text(entry.get(src["key_in"])) + ["<SEP>"] + clean_text(entry.get(src["key_out"]))
            else:
                words = clean_text(entry.get(src["key_in"]))

            # Hanya ambil data yang panjangnya melebihi SEQ_LEN
            if len(words) > SEQ_LEN + 1:
                word_counter.update(words)
                raw_data.append({"words": words, "exp": src["exp"], "hemi": src["hemi"]})
                count += 1

    print("\n📝 TAHAP 2: Membangun Kamus (Vocab)...")
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
    for w, _ in word_counter.most_common(MAX_VOCAB - len(vocab)):
        if w not in vocab: vocab[w] = len(vocab)

    with open(VOCAB_FILE, "w", encoding="utf-8") as f: json.dump(vocab, f)
    print(f"   -> {len(vocab)} kata berhasil disimpan ke {VOCAB_FILE}")

    print("\n⏱️ TAHAP 3: Membuat Garis Waktu (Sequence Timeline)...")
    seqs_list, targets_list, hemis_list, experts_list = [], [], [], []

    for data in tqdm(raw_data, desc="Memotong sekuens"):
        tokens = [vocab.get(w, 1) for w in data["words"]]

        # Mengambil riwayat 32 kata untuk menebak kata ke-33
        for i in range(len(tokens) - SEQ_LEN):
            seqs_list.append(tokens[i : i + SEQ_LEN])
            targets_list.append(tokens[i + SEQ_LEN])
            hemis_list.append(data["hemi"])
            experts_list.append(data["exp"])

    print("\n💾 TAHAP 4: Menyimpan ke PyTorch Tensor (.pt)...")
    dataset_dict = {
        "windows": torch.tensor(seqs_list, dtype=torch.long), # Tetap pakai key "windows" agar cocok dengan train_v5.py
        "targets": torch.tensor(targets_list, dtype=torch.long),
        "hemis": torch.tensor(hemis_list, dtype=torch.long),
        "experts": torch.tensor(experts_list, dtype=torch.long),
    }
    torch.save(dataset_dict, DATASET_FILE)
    print(f"✅ Selesai! Tersimpan di: {DATASET_FILE}")
    print(f"   -> Jumlah sampel siap latih: {len(seqs_list)}")

if __name__ == "__main__":
    main()