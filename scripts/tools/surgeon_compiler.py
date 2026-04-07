import os
import struct
import torch
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM

print("======================================================")
print("     ZIG-LLM: THE FINAL COMPILER (.zbrain v2.1)       ")
print("======================================================")

model_id = "Qwen/Qwen2.5-Coder-0.5B"
output_path = "models/qwen_0.5b_moe.zbrain"

num_vocab_clusters = 64
num_moe_experts = 8

print(f"\n[0/5] Menyiapkan Pasien FP16: {model_id}...")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

vocab_size = len(tokenizer)
num_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
num_heads = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
head_dim = hidden_dim // num_heads

with open(output_path, "wb") as f:
    
    # =================================================================
    # HEADER GLOBAL
    # =================================================================
    print("\n[1/5] Menulis Header & Meta-Data Global...")
    f.write(b"ZBRN") 
    f.write(struct.pack("<I", 2)) # Version = 2
    
    f.write(struct.pack("<I", vocab_size))
    f.write(struct.pack("<I", hidden_dim))
    f.write(struct.pack("<I", num_layers))
    f.write(struct.pack("<I", num_heads))
    f.write(struct.pack("<I", num_kv_heads))
    f.write(struct.pack("<I", head_dim))
    f.write(struct.pack("<I", num_vocab_clusters))
    f.write(struct.pack("<I", num_moe_experts))

    # =================================================================
    # BAGIAN 1: TOKENIZER DENGAN PADDING
    # =================================================================
    print("[2/5] Mengekstrak Tokenizer O(1)...")
    offsets, blob, current_offset = [], bytearray(), 0
    for i in range(vocab_size):
        offsets.append(current_offset)
        token_bytes = tokenizer.decode([i]).encode('utf-8', errors='replace')
        blob.extend(token_bytes)
        current_offset += len(token_bytes)
    offsets.append(current_offset)

    f.write(struct.pack("<I", current_offset))
    for off in offsets: f.write(struct.pack("<I", off))
    f.write(blob)

    # ---> INI DIA PENYELAMAT KITA (PADDING ALIGNMENT) <---
    # Memastikan data selanjutnya jatuh tepat di kelipatan 4 bytes!
    padding_len = (4 - (len(blob) % 4)) % 4
    f.write(b'\x00' * padding_len)

    # =================================================================
    # BAGIAN 2: POHON KOSAKATA
    # =================================================================
    print(f"[3/5] Membedah Pohon Kosakata menjadi {num_vocab_clusters} Laci (~1 Menit)...")
    lm_head = model.lm_head.weight.detach().float().numpy()[:vocab_size, :]
    kmeans_vocab = KMeans(n_clusters=num_vocab_clusters, random_state=42, n_init="auto")
    labels_vocab = kmeans_vocab.fit_predict(lm_head)

    f.write(kmeans_vocab.cluster_centers_.astype(np.float32).tobytes())
    for laci_id in range(num_vocab_clusters):
        token_ids = np.where(labels_vocab == laci_id)[0]
        f.write(struct.pack("<I", len(token_ids)))
        f.write(token_ids.astype(np.uint32).tobytes())
        f.write(lm_head[token_ids].astype(np.float32).tobytes())

    # =================================================================
    # BAGIAN 3: EMBEDDINGS & FINAL NORM
    # =================================================================
    print("\n[4/5] Mengekstrak Darah (Embeddings) & Katup Jantung (RMSNorm)...")
    embed_weights = model.model.embed_tokens.weight.detach().float().numpy()[:vocab_size, :]
    f.write(embed_weights.tobytes())
    
    final_norm = model.model.norm.weight.detach().float().numpy()
    f.write(final_norm.tobytes())

    # =================================================================
    # BAGIAN 4: 24 LAPISAN OTAK
    # =================================================================
    print(f"\n[5/5] MEMBEDAH {num_layers} LAPISAN OTAK...")
    print("      (CPU akan bekerja keras ~5 Menit. Ambil kopi Anda ☕)")
    for layer_idx in range(num_layers):
        print(f"      -> [Layer {layer_idx:02d}/{num_layers-1}] Mengekstrak Atensi, MoE, dan Norms...")
        layer = model.model.layers[layer_idx]
        
        attn_norm = layer.input_layernorm.weight.detach().float().numpy()
        moe_norm = layer.post_attention_layernorm.weight.detach().float().numpy()
        f.write(attn_norm.tobytes())
        f.write(moe_norm.tobytes())

        attn = layer.self_attn
        f.write(attn.q_proj.weight.detach().float().numpy().tobytes())
        f.write(attn.k_proj.weight.detach().float().numpy().tobytes())
        f.write(attn.v_proj.weight.detach().float().numpy().tobytes())
        f.write(attn.o_proj.weight.detach().float().numpy().tobytes())

        mlp = layer.mlp
        gate_proj = mlp.gate_proj.weight.detach().float().numpy()
        up_proj   = mlp.up_proj.weight.detach().float().numpy()
        down_proj = mlp.down_proj.weight.detach().float().numpy()

        kmeans_mlp = KMeans(n_clusters=num_moe_experts, random_state=42, n_init="auto")
        labels_mlp = kmeans_mlp.fit_predict(gate_proj)

        f.write(kmeans_mlp.cluster_centers_.astype(np.float32).tobytes())
        for expert_id in range(num_moe_experts):
            neuron_idxs = np.where(labels_mlp == expert_id)[0]
            f.write(struct.pack("<I", len(neuron_idxs)))
            f.write(gate_proj[neuron_idxs, :].astype(np.float32).tobytes())
            f.write(up_proj[neuron_idxs, :].astype(np.float32).tobytes())
            f.write(down_proj[:, neuron_idxs].astype(np.float32).tobytes())

print("\n======================================================")
print(" [SUCCESS] OTAK V2.1 (.zbrain) BERHASIL DICIPTAKAN!   ")
print("======================================================")