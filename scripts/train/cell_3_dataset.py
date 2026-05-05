import os
import json
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tqdm.auto import tqdm

try:
    from google.colab import drive
    COLAB_MODE = True
except ImportError:
    COLAB_MODE = False

# ==========================================
# ⚙️ KONFIGURASI PATH & DATASET V7 (BPE)
# ==========================================
if COLAB_MODE: drive.mount('/content/drive')

DRIVE_BASE = "/content/drive/MyDrive/Dual-Brain" if COLAB_MODE else "./local_workspace"
DRIVE_DATA = os.path.join(DRIVE_BASE, "data")
os.makedirs(DRIVE_DATA, exist_ok=True)

VOCAB_FILE = os.path.join(DRIVE_DATA, "vocab_v7.json")
DATASET_FILE = os.path.join(DRIVE_DATA, "dataset_v7.pt")

SEQ_LEN = 32
VOCAB_SIZE = 15000
MAX_SAMPLES_PER_DOMAIN = 10000

SOURCES = [
    {"repo": "microsoft/orca-math-word-problems-200k", "split": "train", "key_in": "question", "key_out": "answer", "exp": 0, "hemi": 0},
    {"repo": "iamtarun/python_code_instructions_18k_alpaca", "split": "train", "key_in": "instruction", "key_out": "output", "exp": 1, "hemi": 0},
    {"repo": "sciq", "split": "train", "key_in": "question", "key_out": "support", "exp": 2, "hemi": 1},
    {"repo": "roneneldan/TinyStories", "split": "train", "key_in": "text", "key_out": None, "exp": 3, "hemi": 1},
]

def main():
    print("==================================================")
    print(" 📦 MEMBANGUN DATASET V7 (BPE SUBWORD ERA)")
    print("==================================================\n")

    # 1. KUMPULKAN TEKS MENTAH
    print("🔍 TAHAP 1: Mengunduh data untuk melatih Tokenizer...")
    raw_texts = []
    metadata = []

    for src in SOURCES:
        ds = load_dataset(src["repo"], split=src["split"], streaming=True)
        count = 0
        for entry in ds:
            if count >= MAX_SAMPLES_PER_DOMAIN: break

            text = str(entry.get(src["key_in"]) or "")
            if src["key_out"] is not None:
                text += " <SEP> " + str(entry.get(src["key_out"]) or "")

            raw_texts.append(text)
            metadata.append({"exp": src["exp"], "hemi": src["hemi"]})
            count += 1

    # 2. LATIH TOKENIZER BPE (Seperti GPT)
    print("\n📝 TAHAP 2: Melatih BPE Tokenizer dari nol...")
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # 🚀 FIX: Disesuaikan dengan HuggingFace tokenizers versi terbaru
    tokenizer.pre_tokenizer = Metaspace(replacement=" ")

    trainer = BpeTrainer(special_tokens=["<PAD>", "<UNK>", "<SEP>"], vocab_size=VOCAB_SIZE)
    tokenizer.train_from_iterator(raw_texts, trainer=trainer)

    # Ekstrak Vocab dan ubah Metaspace kembali ke Spasi literal (agar Zig mudah membacanya)
    raw_vocab = tokenizer.get_vocab()
    clean_vocab = {}
    for token, id in raw_vocab.items():
        clean_token = token.replace(" ", " ") # Ganti simbol metaspace dengan spasi asli
        clean_vocab[clean_token] = id

    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_vocab, f)
    print(f"   ✅ {len(clean_vocab)} sub-kata berhasil disimpan ke {VOCAB_FILE}")

    # 3. BUAT DATASET TENSOR
    print("\n⏱️ TAHAP 3: Membuat Garis Waktu Tensor (BPE Sequence)...")
    seqs_list, targets_list, hemis_list, experts_list = [], [], [], []

    for text, meta in tqdm(zip(raw_texts, metadata), total=len(raw_texts), desc="Memotong sekuens"):
        # Encode teks utuh menggunakan BPE Tokenizer yang baru dilatih
        tokens = tokenizer.encode(text).ids

        if len(tokens) > SEQ_LEN + 1:
            for i in range(len(tokens) - SEQ_LEN):
                seqs_list.append(tokens[i : i + SEQ_LEN])
                targets_list.append(tokens[i + SEQ_LEN])
                hemis_list.append(meta["hemi"])
                experts_list.append(meta["exp"])

    print("\n💾 TAHAP 4: Menyimpan ke PyTorch Tensor (.pt)...")
    dataset_dict = {
        "windows": torch.tensor(seqs_list, dtype=torch.long),
        "targets": torch.tensor(targets_list, dtype=torch.long),
        "hemis": torch.tensor(hemis_list, dtype=torch.long),
        "experts": torch.tensor(experts_list, dtype=torch.long),
    }
    torch.save(dataset_dict, DATASET_FILE)
    print(f" ✅ DATASET V7 SELESAI DIBUAT: {len(seqs_list)} sampel siap latih!")

if __name__ == "__main__":
    main()