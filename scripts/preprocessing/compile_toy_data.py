import json
import struct
import os

# Konfigurasi Path
INPUT_JSONL = "data/raw/toy_dual_brain.jsonl"
OUTPUT_BIN = "data/processed/toy_dual_brain.hmoe"
OUTPUT_VOCAB = "data/processed/toy_vocab.json"

# Peta Label -> ID (Sesuai dengan Enum di Zig)
HEMISPHERE_MAP = {"left": 0, "right": 1}
EXPERT_MAP = {
    "calculator": 0,
    "syntactician": 1,
    "futurist": 2,
    "storyteller": 3
}

def main():
    print(f"Membaca {INPUT_JSONL}...")
    
    # Pastikan folder output ada
    os.makedirs(os.path.dirname(OUTPUT_BIN), exist_ok=True)
    
    records = []
    vocab = {"<|PAD|>": 0, "<|END|>": 1} # ID Dasar
    
    # Fungsi pembantu untuk tokenisasi sederhana berbasis spasi
    def tokenize(text):
        tokens = []
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
            tokens.append(vocab[word])
        return tokens

    # 1. Parsing JSONL dan Tokenisasi
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            input_tokens = tokenize(data["input"])
            target_tokens = tokenize(data["target"])
            
            records.append({
                "hemisphere": HEMISPHERE_MAP[data["hemisphere"]],
                "expert": EXPERT_MAP[data["expert"]],
                "inputs": input_tokens,
                "targets": target_tokens
            })

    # 2. Tulis ke file Biner (.hmoe)
    print(f"Mengkompilasi {len(records)} baris menjadi biner...")
    with open(OUTPUT_BIN, "wb") as f_bin:
        for rec in records:
            in_len = len(rec["inputs"])
            tgt_len = len(rec["targets"])
            
            # Pack Header: B (u8), B (u8), H (u16), H (u16)
            # Menggunakan '<' untuk Little-Endian (standar format memori yang disukai Zig)
            header = struct.pack('<BBHH', rec["hemisphere"], rec["expert"], in_len, tgt_len)
            f_bin.write(header)
            
            # Pack Input Tokens: Array of u32 (I)
            in_bytes = struct.pack(f'<{in_len}I', *rec["inputs"])
            f_bin.write(in_bytes)
            
            # Pack Target Tokens: Array of u32 (I)
            tgt_bytes = struct.pack(f'<{tgt_len}I', *rec["targets"])
            f_bin.write(tgt_bytes)

    # 3. Simpan Kamus Vocab (untuk keperluan debug/inference)
    with open(OUTPUT_VOCAB, "w", encoding="utf-8") as f_voc:
        json.dump(vocab, f_voc, indent=4)

    print(f"Selesai!")
    print(f"-> File Biner : {OUTPUT_BIN} ({os.path.getsize(OUTPUT_BIN)} bytes)")
    print(f"-> File Vocab : {OUTPUT_VOCAB} ({len(vocab)} kata unik)")

if __name__ == "__main__":
    main()