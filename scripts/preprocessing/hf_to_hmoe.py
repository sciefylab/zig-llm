import struct
import json
import re
from datasets import load_dataset

# =================================================================
# ⚙️ KONFIGURASI PATH & HYPERPARAMETER
# =================================================================
BIN_FILE = "data/processed/real_dual_brain.hmoe"
VOCAB_FILE = "data/processed/real_vocab.json"
MAX_SAMPLES_PER_SOURCE = 2000 # Cukup untuk memastikan keseimbangan distribusi pakar
SEQ_LEN = 64
MAX_VOCAB_SIZE = 5000 

# 🛠️ HEADER 8-BYTES: hemi_val (u8), exp_val (u8), in_len (u16), tg_len (u16), padding (2 bytes)
HEADER_FORMAT = "<BBHHH"

# 🌐 MULTILINGUAL & FALLBACK SOURCES
SOURCES = [
    # Kiri: Ahli Hitung
    {"repo": "microsoft/orca-math-word-problems-200k", "split": "train", "key": "question"},
    # Kiri/Kanan: Pengetahuan umum & Sains (Wikipedia Indo) - Cross Lingual
    {"repo": "Babelscape/wikineural", "split": "train_id", "key": "text"}, 
    # Kanan: Ahli Cerita & Basa-basi (Fallback)
    {"repo": "roneneldan/TinyStories", "split": "train", "key": "text"},
    # Tambahan: Instruksi umum untuk memicu deteksi dinamis
    {"repo": "tatsu-lab/alpaca", "split": "train", "key": "instruction"}
]

# =================================================================
# 🧠 THE INTERLINGUA ROUTER
# =================================================================
def route_intent(text):
    """
    Klasifikasi Rute Otomatis yang sangat komprehensif.
    """
    text_lower = text.lower()
    
    # 1. KIRI - Ahli Matematika & Logika (Calculator)
    math_keywords = r"\b(hitung|berapa|kalkulasi|tambah|kurang|kali|bagi|jumlahkan|selisih|persentase|persen|pecahan|kuadrat|akar|matematika|aljabar|geometri|statistika|peluang|rumus|math|calculate|sum|equation|subtract|multiply|divide|algebra|geometry|statistics|probability|percentage|fraction|formula|solve)\b"
    math_symbols = r"[\+\-\*\/\=\%\^0-9]"
    
    if re.search(math_keywords, text_lower) or re.search(math_symbols, text_lower):
        return 0, 0  # Left (0), Calculator (0)

    # 2. KIRI - Ahli Sintaks & Kode (Syntactician)
    code_keywords = r"\b(kode|program|fungsi|variabel|algoritma|perbaiki|error|bug|debug|kompilasi|sintaks|skrip|def|const|fn|struct|import|function|return|print|let|var|class|interface|async|await|python|zig|javascript|html|css|cpp|java|bash|json|xml|yaml|script|compile)\b"
    code_symbols = r"[\{\}\[\]\<\>\;\#\&\$\@]"
    
    if re.search(code_keywords, text_lower) or re.search(code_symbols, text_lower):
        return 0, 1  # Left (0), Syntactician (1)

    # 3. KANAN - Ahli Teori, Sains & Pengetahuan (Futurist)
    science_keywords = r"\b(apa|siapa|kapan|di mana|mengapa|bagaimana|jelaskan|teori|sains|fakta|sejarah|penemuan|bumi|alam semesta|biologi|fisika|kimia|astronomi|filosofi|prediksi|masa depan|rangkum|definisi|what|who|when|where|why|how|explain|science|fact|history|discovery|earth|universe|biology|physics|chemistry|astronomy|philosophy|predict|future|summarize|summary|definition|describe|concept)\b"
    
    if re.search(science_keywords, text_lower):
        return 1, 2  # Right (1), Futurist (2)

    # 4. KANAN - Ahli Cerita & Basa-basi (Storyteller)
    story_keywords = r"\b(cerita|puisi|pantun|dongeng|lelucon|fiksi|metafora|ngobrol|hai|halo|selamat|sedih|senang|story|poem|joke|fairy tale|fiction|metaphor|chat|hi|hello|good morning|sad|happy|once upon a time)\b"
    
    if re.search(story_keywords, text_lower):
        return 1, 3  # Right (1), Storyteller (3)

    # 5. GRACEFUL DEGRADATION (Absolute Fallback)
    return 1, 3      # Right (1), Storyteller (3)

# =================================================================
# 📦 COMPILER ENGINE
# =================================================================
def compile_hf_to_hmoe():
    vocab = {"<|PAD|>": 0, "<|END|>": 1, "<|UNK|>": 2}
    all_records = []
    expert_stats = {"0_0": 0, "0_1": 0, "1_2": 0, "1_3": 0}

    for src in SOURCES:
        print(f"\n📥 Mengekstrak data dari HuggingFace: {src['repo']}...")
        try:
            ds = load_dataset(src["repo"], split=src["split"], streaming=True, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️ Gagal memuat {src['repo']}: {e}")
            continue
        
        count = 0
        for entry in ds:
            if count >= MAX_SAMPLES_PER_SOURCE: break
            
            raw_text = entry.get(src["key"], "")
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)
                
            text = raw_text.lower()
            hemi, expert = route_intent(text)
            
            tokens = []
            for w in text.split()[:SEQ_LEN]:
                if w not in vocab:
                    if len(vocab) < MAX_VOCAB_SIZE:
                        vocab[w] = len(vocab)
                        tokens.append(vocab[w])
                    else:
                        tokens.append(vocab["<|UNK|>"])
                else:
                    tokens.append(vocab[w])
            
            if len(tokens) < 5: continue 
            
            target = tokens[1:] + [vocab["<|END|>"]]

            all_records.append({
                "h": hemi,
                "e": expert,
                "in": tokens,
                "tg": target
            })
            expert_stats[f"{hemi}_{expert}"] += 1
            count += 1
            
        print(f"✅ Berhasil merangkum {count} sampel dari {src['repo']}.")

    print("\n📊 STATISTIK ROUTING PAKAR (HMoE):")
    print(f"   🧠 Kiri  - Calculator   : {expert_stats['0_0']} sequences")
    print(f"   🧠 Kiri  - Syntactician : {expert_stats['0_1']} sequences")
    print(f"   💡 Kanan - Futurist     : {expert_stats['1_2']} sequences")
    print(f"   💡 Kanan - Storyteller  : {expert_stats['1_3']} sequences (Fallback)")

    print(f"\n📦 Menyusun total {len(all_records)} records ke format biner {BIN_FILE}...")
    with open(BIN_FILE, "wb") as f:
        for r in all_records:
            f.write(struct.pack(HEADER_FORMAT, r["h"], r["e"], len(r["in"]), len(r["tg"]), 0))
            f.write(struct.pack(f'<{len(r["in"])}I', *r["in"]))
            f.write(struct.pack(f'<{len(r["tg"])}I', *r["tg"]))

    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=4, ensure_ascii=False)
    
    print(f"🚀 FASE 1 SELESAI! Final Vocab size: {len(vocab)} / {MAX_VOCAB_SIZE}")

if __name__ == "__main__":
    compile_hf_to_hmoe()