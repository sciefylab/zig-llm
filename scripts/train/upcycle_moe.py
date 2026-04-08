import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM

print("======================================================")
print("     ZIG-LLM: AI TRAINING STUDIO (MoE UPCYCLING)      ")
print("======================================================")

model_id = "Qwen/Qwen2.5-Coder-0.5B"
num_experts = 8

# ==========================================================
# 1. PERSIAPAN DATA DAN GURU
# ==========================================================
print("\n[1/4] Membangunkan Guru (Teacher Model)...")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Untuk training, kita WAJIB menggunakan float32 (bukan float16) agar Kalkulusnya presisi
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval() # Guru tidak boleh ikut belajar, otaknya dibekukan.

# Kita ambil 1 kalimat untuk bahan belajar
teks_pelajaran = "Zig is a general-purpose programming language and toolchain for maintaining robust, optimal, and reusable software."
tokens = tokenizer(teks_pelajaran, return_tensors="pt")
print(f"      -> Bahan Belajar: '{teks_pelajaran}'")

# ==========================================================
# 2. MEMBANGUN ARSITEKTUR MURID (STUDENT MoE)
# ==========================================================
print("\n[2/4] Membangun Otak Murid (Arsitektur 8 Pakar Zig)...")

# Ekstrak FFN Asli (Guru)
mlp_guru = model.model.layers[0].mlp
hidden_dim = mlp_guru.gate_proj.weight.shape[1]

# Lakukan Operasi Bedah (Seperti di surgeon_compiler)
gate_w = mlp_guru.gate_proj.weight.detach().numpy()
kmeans = KMeans(n_clusters=num_experts, random_state=42, n_init="auto")
labels = kmeans.fit_predict(gate_w)

# Bikin Class PyTorch yang meniru struktur struct Zig Anda!
# Bikin Class PyTorch yang meniru struktur struct Zig Anda!
class ZigMoEStudent(nn.Module):
    def __init__(self, num_experts, hidden_dim, kmeans_labels, centroids):
        super().__init__()
        self.num_experts = num_experts
        
        # Router (Ini yang akan DILATIH / DIAJARI)
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.router.weight.data = torch.tensor(centroids, dtype=torch.float32)
        
        # Pakar-Pakar (Dibekukan, tidak ikut belajar)
        self.experts_gate = nn.ParameterList()
        self.experts_up = nn.ParameterList()
        self.experts_down = nn.ParameterList()
        
        for i in range(num_experts):
            idxs = np.where(kmeans_labels == i)[0]
            self.experts_gate.append(nn.Parameter(torch.tensor(mlp_guru.gate_proj.weight[idxs, :].detach().numpy(), dtype=torch.float32), requires_grad=False))
            self.experts_up.append(nn.Parameter(torch.tensor(mlp_guru.up_proj.weight[idxs, :].detach().numpy(), dtype=torch.float32), requires_grad=False))
            self.experts_down.append(nn.Parameter(torch.tensor(mlp_guru.down_proj.weight[:, idxs].detach().numpy(), dtype=torch.float32), requires_grad=False))

    def forward(self, x):
        # 1. Router menghasilkan skor mentah
        router_logits = self.router(x) # Shape: [Batch, Seq, Experts]
        
        # 2. TRIK KALKULUS: Ubah skor jadi persentase bergradien (Differentiable)
        routing_probs = torch.nn.functional.softmax(router_logits, dim=-1)
        
        output = torch.zeros_like(x)
        
        # 3. Eksekusi Pakar secara diferensiabel (Soft Routing)
        seq_len = x.shape[1]
        for pos in range(seq_len):
            token_x = x[0, pos, :]
            token_probs = routing_probs[0, pos, :] # Persentase masing-masing pakar
            
            token_output_total = torch.zeros_like(token_x)
            
            # Gabungkan hasil dari semua pakar (dikalikan dengan persentase kecerdasannya)
            for expert_id in range(self.num_experts):
                w_gate = self.experts_gate[expert_id]
                w_up = self.experts_up[expert_id]
                w_down = self.experts_down[expert_id]
                
                # SiLU (Gate * x) * (Up * x)
                # Gunakan torch.matmul agar Kalkulus tetap hidup!
                gate_out = torch.nn.functional.silu(torch.matmul(w_gate, token_x))
                up_out = torch.matmul(w_up, token_x)
                intermediate = gate_out * up_out
                
                # Down Proj
                expert_output = torch.matmul(w_down, intermediate)
                
                # Kalikan hasil pakar ini dengan "Kepercayaan" Router
                token_output_total += expert_output * token_probs[expert_id]
                
            output[0, pos, :] = token_output_total
            
        return output

# Buat instansi Murid (Pastikan model dibuat ulang)
murid_moe = ZigMoEStudent(num_experts, hidden_dim, labels, kmeans.cluster_centers_)


# ==========================================================
# 3. MENDAPATKAN KUNCI JAWABAN DARI GURU
# ==========================================================
print("\n[3/4] Mendapatkan Kunci Jawaban (Sinyal Sempurna) dari Guru...")
with torch.no_grad(): # Tanpa kalkulus
    # Dapatkan Hidden State sebelum masuk ke MLP Layer 0
    hidden_states = model.model.embed_tokens(tokens.input_ids)
    
    # Lewatkan ke MLP Guru yang asli (Dense)
    sinyal_guru_sempurna = mlp_guru(hidden_states)

# ==========================================================
# 4. PROSES PELATIHAN (TRAINING LOOP)
# ==========================================================
print("\n[4/4] Memulai Pelatihan Router MoE (Backpropagation)...")

# Kita menggunakan Optimizer Adam (Standar Industri AI)
# Hanya Router yang diajari (lr = 0.01 adalah kecepatan belajar)
optimizer = optim.Adam(murid_moe.router.parameters(), lr=0.01)
loss_function = nn.MSELoss() # Mean Squared Error (Menghitung selisih sinyal)

epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad() # Bersihkan ingatan kalkulus
    
    # 1. Murid mencoba menjawab (Forward Pass)
    sinyal_murid_cacat = murid_moe(hidden_states)
    
    # 2. Hitung seberapa bodoh Murid dibandingkan Guru (Hitung Loss)
    # Tujuan kita adalah membuat angka Loss ini menjadi 0.0000
    loss = loss_function(sinyal_murid_cacat, sinyal_guru_sempurna)
    
    # 3. HUKUMAN! (Backward Pass / Kalkulus Turunan)
    loss.backward()
    
    # 4. Murid memperbaiki otaknya berdasarkan hukuman (Step)
    optimizer.step()
    
    # Cetak perkembangan otak Murid setiap 10 putaran
    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(f"      -> Putaran [{(epoch + 1):2d}/{epochs}] | Tingkat Kesalahan (Loss): {loss.item():.6f}")

print("\n======================================================")
print(" [SUCCESS] PELATIHAN SELESAI! OTAK MURID MAKIN PINTAR!")
print("======================================================")
print(" Catatan: Di dunia nyata, proses ini dilakukan pada jutaan kalimat,")
print(" dan bobot Router hasil belajar ini akan diekspor ke .zbrain!")