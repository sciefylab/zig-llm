import struct
import json
import re
import random
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

# =================================================================
# ⚙️ KONFIGURASI PATH & HYPERPARAMETER
# =================================================================
BIN_FILE = "data/processed/real_dual_brain.hmoe"
VOCAB_FILE = "data/processed/real_vocab.json"

MAX_SAMPLES_PER_SOURCE = 5000
SEQ_LEN = 64
MAX_VOCAB_SIZE = 15000
MIN_TOKENS_PER_SEQ = 8
MIN_WORD_FREQ = 2

HEADER_FORMAT = "<BBHHH"

PAD_ID = 0
END_ID = 1
UNK_ID = 2
SPECIAL_TOKENS = {"<|PAD|>": PAD_ID, "<|END|>": END_ID, "<|UNK|>": UNK_ID}

random.seed(42)

# =================================================================
# 🌐 SUMBER DATASET (DITAMBAH CODE DATASETS!)
# =================================================================
SOURCES = [
    # ================================================
    # 🧠 KIRI - CALCULATOR (Math)
    # ================================================
    {
        "repo": "microsoft/orca-math-word-problems-200k",
        "split": "train", "key": "question", "name": None,
        "force": (0, 0), "category": "math",
    },
    {
        "repo": "openai/gsm8k",
        "split": "train", "key": "question", "name": "main",
        "force": (0, 0), "category": "math",
    },
    
    # ================================================
    # 🧠 KIRI - SYNTACTICIAN (Code) 🔥 DITAMBAH BANYAK!
    # ================================================
    {
        "repo": "sahil2801/CodeAlpaca-20k",
        "split": "train", "key": "instruction", "name": None,
        "force": (0, 1), "category": "code",
        "combine_keys": ["instruction", "output"],  # 🔥 gabung prompt + code
    },
    {
        "repo": "iamtarun/python_code_instructions_18k_alpaca",
        "split": "train", "key": "instruction", "name": None,
        "force": (0, 1), "category": "code",
        "combine_keys": ["instruction", "output"],
    },
    {
        "repo": "mbpp",
        "split": "train", "key": "text", "name": "full",
        "force": (0, 1), "category": "code",
        "combine_keys": ["text", "code"],
    },
    # Opsional tambahan (uncomment kalau mau lebih banyak):
    # {
    #     "repo": "glaiveai/glaive-code-assistant",
    #     "split": "train", "key": "question", "name": None,
    #     "force": (0, 1), "category": "code",
    #     "combine_keys": ["question", "answer"],
    # },
    
    # ================================================
    # 💡 KANAN - FUTURIST (Science)
    # ================================================
    {
        "repo": "sciq",
        "split": "train", "key": "support", "name": None,
        "force": (1, 2), "category": "science",
    },
    
    # ================================================
    # 💡 KANAN - GENERAL (Wikipedia Indo)
    # ================================================
    {
        "repo": "wikimedia/wikipedia",
        "split": "train", "key": "text", "name": "20231101.id",
        "force": None, "category": "general",
    },
    
    # ================================================
    # 💡 KANAN - STORYTELLER
    # ================================================
    {
        "repo": "roneneldan/TinyStories",
        "split": "train", "key": "text", "name": None,
        "force": (1, 3), "category": "story",
    },
]

# =================================================================
# 🧹 TEXT CLEANING
# =================================================================
def clean_text(text: str, preserve_code: bool = False) -> str:
    """
    Bersihkan teks. Jika preserve_code=True, JANGAN lowercase dan pertahankan 
    whitespace/punctuation yang penting untuk kode.
    """
    if not isinstance(text, str):
        text = str(text)
    
    if preserve_code:
        # 🔥 Untuk kode: jangan lowercase, minimal cleaning
        # Hapus URL
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # Normalisasi newline ke space (agar fit SEQ_LEN)
        text = re.sub(r'\n+', ' \n ', text)  # simpan newline sebagai token
        text = re.sub(r'[ \t]+', ' ', text).strip()
        return text
    
    # Untuk text biasa
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_words(text: str, is_code: bool = False) -> list:
    """Tokenizer. Untuk kode, pertahankan simbol yang penting."""
    if is_code:
        # 🔥 Pisahkan simbol kode penting sebagai token terpisah
        # def, =, (, ), {, }, [, ], ;, :, ,, ., ->, ==, !=, +, -, *, /, %, <, >
        text = re.sub(r"([(){}\[\];:,\.=\+\-\*/%<>!])", r" \1 ", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()
    else:
        text = re.sub(r"([.,!?;:\"'()\[\]])", r" \1 ", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

# =================================================================
# 🧠 ROUTER
# =================================================================
def route_intent(text: str):
    text_lower = text.lower()
    
    # Math (equation atau keyword)
    has_equation = bool(re.search(r'\d+\s*[\+\-\*/=]\s*\d+', text_lower))
    math_keywords = r"\b(calculate|solve|equation|sum|total cost|how many|how much|add|subtract|multiply|divide|percent|average|hitung|berapa|jumlah|kalikan|bagi|rata[- ]rata|persentase)\b"
    
    if has_equation or re.search(math_keywords, text_lower):
        return 0, 0
    
    # Code detection
    code_keywords = r"\b(def |function |class |import |return |print\(|console\.log|var |let |const |public static|#include|package )\b"
    code_symbols = bool(re.search(r'[\{\}]|\(\s*\)|==|!=|=>|\-\>', text))
    
    if re.search(code_keywords, text_lower) or code_symbols:
        return 0, 1
    
    # Science
    science_keywords = r"\b(theory|science|physics|chemistry|biology|quantum|molecule|force|velocity|energy|experiment|hypothesis|neutron|proton|teori|sains|fisika|kimia|biologi|molekul|energi|eksperimen)\b"
    
    if re.search(science_keywords, text_lower):
        return 1, 2
    
    # Story
    story_keywords = r"\b(once upon a time|story|tale|fairy|prince|princess|dragon|lily|tom|happy|sad|cerita|dongeng|pangeran|putri|naga|bahagia|sedih)\b"
    
    if re.search(story_keywords, text_lower):
        return 1, 3
    
    return 1, 3

# =================================================================
# 🎯 EXTRACT TEXT DARI ENTRY (Support multi-key untuk code datasets)
# =================================================================
def extract_text(entry: dict, src: dict) -> str:
    """
    Extract teks dari entry. Jika ada `combine_keys`, gabung beberapa field.
    Contoh code dataset biasanya punya 'instruction' + 'output' (code).
    """
    combine_keys = src.get("combine_keys")
    
    if combine_keys:
        parts = []
        for key in combine_keys:
            val = entry.get(key, "")
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        return " \n ".join(parts)
    else:
        raw = entry.get(src["key"], "")
        return str(raw) if not isinstance(raw, str) else raw

# =================================================================
# 📊 PASS 1: BUILD VOCAB
# =================================================================
def build_vocab_pass():
    print("\n📊 PASS 1: Membangun vocabulary berdasarkan frekuensi kata...")
    word_counter = Counter()
    raw_samples = []
    
    for src in SOURCES:
        is_code = (src.get("category") == "code")
        print(f"\n📥 Scanning: {src['repo']} [{src.get('category', 'unknown')}]")
        
        try:
            if src.get("name"):
                ds = load_dataset(src["repo"], src["name"], split=src["split"], streaming=True)
            else:
                ds = load_dataset(src["repo"], split=src["split"], streaming=True)
        except Exception as e:
            print(f"⚠️  Gagal: {e}")
            continue
        
        count = 0
        pbar = tqdm(total=MAX_SAMPLES_PER_SOURCE, desc=f"  {src['repo'][:35]}")
        try:
            for entry in ds:
                if count >= MAX_SAMPLES_PER_SOURCE:
                    break
                
                raw_text = extract_text(entry, src)
                text = clean_text(raw_text, preserve_code=is_code)
                
                if len(text) < 20:
                    continue
                
                words = tokenize_words(text, is_code=is_code)[:SEQ_LEN]
                if len(words) < MIN_TOKENS_PER_SEQ:
                    continue
                
                word_counter.update(words)
                raw_samples.append({
                    "words": words,
                    "force": src.get("force"),
                    "text": text[:500],
                    "is_code": is_code,
                })
                count += 1
                pbar.update(1)
        except Exception as e:
            print(f"\n   ⚠️  Error saat iterasi: {e}")
        
        pbar.close()
        print(f"  ✅ {count} samples scanned.")
    
    # Build vocab
    vocab = dict(SPECIAL_TOKENS)
    available_slots = MAX_VOCAB_SIZE - len(vocab)
    
    frequent_words = [w for w, c in word_counter.most_common() if c >= MIN_WORD_FREQ]
    
    for word in frequent_words[:available_slots]:
        vocab[word] = len(vocab)
    
    print(f"\n📚 Vocab dibangun: {len(vocab)}/{MAX_VOCAB_SIZE}")
    print(f"   Total unique words scanned: {len(word_counter)}")
    print(f"   Words with freq >= {MIN_WORD_FREQ}: {len(frequent_words)}")
    
    # 🔍 Stats vocab per kategori
    code_words_in_vocab = sum(1 for w in vocab if w in {
        'def', 'function', 'class', 'import', 'return', 'print', 'var', 'let',
        'const', '(', ')', '{', '}', '[', ']', '=', '==', '!=', '+=', '->',
    })
    print(f"   Code-related tokens in vocab: {code_words_in_vocab}")
    
    return vocab, raw_samples

# =================================================================
# 📦 PASS 2: ENCODE & WRITE BIN
# =================================================================
def encode_and_write(vocab, raw_samples):
    print("\n📦 PASS 2: Encoding & menulis binary file...")
    
    expert_stats = {"0_0": 0, "0_1": 0, "1_2": 0, "1_3": 0}
    all_records = []
    
    random.shuffle(raw_samples)
    
    for sample in tqdm(raw_samples, desc="  Encoding"):
        words = sample["words"]
        tokens = [vocab.get(w, UNK_ID) for w in words]
        
        if len(tokens) < MIN_TOKENS_PER_SEQ:
            continue
        
        split_point = len(tokens) // 2
        inputs = tokens[:split_point]
        targets = tokens[split_point:] + [END_ID]
        
        inputs = inputs[:SEQ_LEN]
        targets = targets[:SEQ_LEN]
        
        if len(inputs) < 2 or len(targets) < 2:
            continue
        
        if sample["force"] is not None:
            hemi, expert = sample["force"]
        else:
            hemi, expert = route_intent(sample["text"])
        
        all_records.append({
            "h": hemi, "e": expert,
            "in": inputs, "tg": targets
        })
        expert_stats[f"{hemi}_{expert}"] += 1
    
    print("\n📊 STATISTIK ROUTING PAKAR (HMoE):")
    total = sum(expert_stats.values())
    names = {
        "0_0": "🧠 Kiri-Calculator    (Math)",
        "0_1": "🧠 Kiri-Syntactician  (Code)",
        "1_2": "💡 Kanan-Futurist     (Science)",
        "1_3": "💡 Kanan-Storyteller  (Story+General)",
    }
    for key, count in expert_stats.items():
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {names[key]:40s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    # Warning kalau imbalance
    min_count = max(min(expert_stats.values()), 1)
    max_count = max(expert_stats.values())
    ratio = max_count / min_count
    if ratio > 10:
        print(f"\n⚠️  PERINGATAN: Distribusi tidak seimbang (ratio {ratio:.1f}x)")
    elif ratio > 5:
        print(f"\n⚡ INFO: Distribusi cukup tapi bisa lebih baik (ratio {ratio:.1f}x)")
    else:
        print(f"\n✅ Distribusi expert seimbang (ratio {ratio:.1f}x)")
    
    random.shuffle(all_records)
    
    print(f"\n💾 Menulis {len(all_records)} records ke {BIN_FILE}...")
    buffer = bytearray()
    for r in all_records:
        buffer.extend(struct.pack(HEADER_FORMAT, r["h"], r["e"], len(r["in"]), len(r["tg"]), 0))
        buffer.extend(struct.pack(f'<{len(r["in"])}I', *r["in"]))
        buffer.extend(struct.pack(f'<{len(r["tg"])}I', *r["tg"]))
    
    with open(BIN_FILE, "wb") as f:
        f.write(buffer)
    
    file_size_mb = len(buffer) / (1024 * 1024)
    print(f"   ✅ Binary size: {file_size_mb:.2f} MB")
    
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Vocab saved: {VOCAB_FILE}")

# =================================================================
# 🚀 MAIN
# =================================================================
def main():
    import os
    os.makedirs("data/processed", exist_ok=True)
    
    print("=" * 60)
    print("🧠 CYBER-DUAL BRAIN DATASET COMPILER v2")
    print("   + CODE DATASETS (Syntactician Boost)")
    print("=" * 60)
    
    vocab, raw_samples = build_vocab_pass()
    encode_and_write(vocab, raw_samples)
    
    print("\n" + "=" * 60)
    print("🚀 FASE 1 SELESAI!")
    print("=" * 60)

if __name__ == "__main__":
    main()