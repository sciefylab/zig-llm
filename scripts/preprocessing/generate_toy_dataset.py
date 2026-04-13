import json
import struct
import os
import random

# Konfigurasi Path
RAW_FILE = "data/raw/toy_dual_brain.jsonl"
BIN_FILE = "data/processed/toy_dual_brain.hmoe"
VOCAB_FILE = "data/processed/toy_vocab.json"

HEMISPHERE_MAP = {"left": 0, "right": 1}
EXPERT_MAP = {"calculator": 0, "syntactician": 1, "futurist": 2, "storyteller": 3}

def generate_logic_data():
    dataset = []
    
    # --- OTAK KIRI: SYNTACTICIAN (Aturan Pasti) ---
    rules = {
        "count": ("u32", "0;"),
        "score": ("i32", "100;"),
        "flag": ("bool", "true;"),
        "index": ("usize", "1;")
    }
    for _ in range(500):
        v = random.choice(list(rules.keys()))
        t, val = rules[v]
        dataset.append({
            "input": f"const {v}: {t} =",
            "target": f"{val} <|END|>",
            "hemisphere": "left", "expert": "syntactician"
        })

    # --- OTAK KIRI: CALCULATOR (Matematika Sederhana) ---
    for _ in range(500):
        a = random.randint(1, 20)
        dataset.append({
            "input": f"{a} tambah {a} sama dengan",
            "target": f"{a+a} <|END|>",
            "hemisphere": "left", "expert": "calculator"
        })

    # --- OTAK KANAN: STORYTELLER (Narasi Konsisten) ---
    for _ in range(500):
        dataset.append({
            "input": "di bawah hujan rintik robot itu",
            "target": "meneteskan air mata digital <|END|>",
            "hemisphere": "right", "expert": "storyteller"
        })

    # --- OTAK KANAN: FUTURIST (Prediksi Tekno) ---
    for _ in range(500):
        dataset.append({
            "input": "di masa depan fusi nuklir akan",
            "target": "mengubah tatanan realitas <|END|>",
            "hemisphere": "right", "expert": "futurist"
        })

    random.shuffle(dataset)
    return dataset

def compile():
    dataset = generate_logic_data()
    vocab = {"<|PAD|>": 0, "<|END|>": 1}
    
    def tokenize(text):
        return [vocab.setdefault(w, len(vocab)) for w in text.split()]

    records = []
    for d in dataset:
        records.append({
            "h": HEMISPHERE_MAP[d["hemisphere"]],
            "e": EXPERT_MAP[d["expert"]],
            "in": tokenize(d["input"]),
            "tg": tokenize(d["target"])
        })

    with open(BIN_FILE, "wb") as f:
        for r in records:
            f.write(struct.pack('<BBHH', r["h"], r["e"], len(r["in"]), len(r["tg"])))
            f.write(struct.pack(f'<{len(r["in"])}I', *r["in"]))
            f.write(struct.pack(f'<{len(r["tg"])}I', *r["tg"]))

    with open(VOCAB_FILE, "w") as f:
        json.dump(vocab, f, indent=4)
    print(f"✅ Dataset & Vocab ({len(vocab)} words) siap!")

if __name__ == "__main__":
    compile()