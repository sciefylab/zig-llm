import os
import struct
from pathlib import Path
from transformers import AutoTokenizer

BASE_PATH = Path("./")
MODEL_ID = "Qwen/Qwen2.5-3B"
OUTPUT_PATH = BASE_PATH / "models/qwen_vocab.zdict"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

vocab_size = len(tokenizer)
offsets = []
blob = bytearray()
current_offset = 0

for i in range(vocab_size):
    offsets.append(current_offset)

    token_str = tokenizer.convert_ids_to_tokens(i)
    if token_str is None:
        token_str = "<unk>"

    token_bytes = token_str.encode("utf-8", errors="replace")
    blob.extend(token_bytes)
    current_offset += len(token_bytes)

offsets.append(current_offset)

with open(OUTPUT_PATH, "wb") as f:
    f.write(b"ZDCT")
    f.write(struct.pack("<I", vocab_size))
    f.write(struct.pack("<I", current_offset))

    for off in offsets:
        f.write(struct.pack("<I", off))

    f.write(blob)

print("Saved:", OUTPUT_PATH)
print("vocab_size:", vocab_size)
print("size_mb:", OUTPUT_PATH.stat().st_size / (1024 * 1024))