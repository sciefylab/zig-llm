# scripts/train/prepare_dataset.py
import argparse
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/wikitext")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--min_seq_len", type=int, default=32)
    parser.add_argument("--max_examples", type=int, default=200_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Wikitext-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Gabung teks dan pecah menjadi paragraf yang bermakna
    text = "\n".join([line for line in dataset["text"] if len(line.strip()) > 30])
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]

    print(f"Found {len(paragraphs)} paragraphs. Tokenizing...")

    all_input_ids = []
    total_tokens = 0

    for paragraph in tqdm(paragraphs[:args.max_examples], desc="Tokenizing"):
        tokens = tokenizer.encode(paragraph, truncation=True, max_length=args.max_seq_len)
        
        if len(tokens) >= args.min_seq_len:
            all_input_ids.append(np.array(tokens, dtype=np.uint32))
            total_tokens += len(tokens)

    # ====================== SAVE BINARY FORMAT ======================
    # Format:
    # [u32: num_sequences]
    # [u32: seq_len][u32 * seq_len: tokens]  ← diulang untuk setiap sequence

    output_path = output_dir / "train.bin"
    
    with open(output_path, "wb") as f:
        # Header
        f.write(np.uint32(len(all_input_ids)).tobytes())
        
        for seq in all_input_ids:
            f.write(np.uint32(len(seq)).tobytes())
            f.write(seq.tobytes())

    print("\n" + "="*70)
    print("✅ DATASET PREPARATION BERHASIL!")
    print("="*70)
    print(f"Jumlah sequence : {len(all_input_ids):,}")
    print(f"Total tokens    : {total_tokens:,}")
    print(f"File output     : {output_path}")
    print(f"Ukuran file     : {output_path.stat().st_size / (1024*1024):.2f} MB")
    print("\nFormat binary siap dibaca oleh Zig dengan zero-copy.")
    print("="*70)


if __name__ == "__main__":
    main()