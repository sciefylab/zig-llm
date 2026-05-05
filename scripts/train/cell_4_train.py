import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

try:
    from google.colab import drive
    COLAB_MODE = True
except ImportError:
    COLAB_MODE = False

# =================================================================
# 🔄 SETUP PATHS (HYBRID MATMUL-FREE ERA)
# =================================================================
if COLAB_MODE: drive.mount('/content/drive')

DRIVE_BASE = "/content/drive/MyDrive/Dual-Brain" if COLAB_MODE else "./local_workspace"
DRIVE_DATA = os.path.join(DRIVE_BASE, "data")
DRIVE_MODELS = os.path.join(DRIVE_BASE, "models")
os.makedirs(DRIVE_DATA, exist_ok=True)
os.makedirs(DRIVE_MODELS, exist_ok=True)

LOCAL_BASE = "/content/workspace" if COLAB_MODE else "./local_workspace"
LOCAL_DATA = os.path.join(LOCAL_BASE, "data")
LOCAL_MODELS = os.path.join(LOCAL_BASE, "models")
os.makedirs(LOCAL_DATA, exist_ok=True)
os.makedirs(LOCAL_MODELS, exist_ok=True)

DATASET_PATH = os.path.join(DRIVE_DATA, "dataset_v7.pt")
CHECKPOINT_NAME = "checkpoint_v8_hybrid.pth"
ZBRAIN_NAME = "real_dual_brain_v7.zbrain"

# ==========================================
# ⚙️ KONFIGURASI HYBRID ARCHITECTURE
# ==========================================
HIDDEN_DIM = 1024
VOCAB_SIZE = 15000
BATCH_SIZE = 4096
EPOCHS = 20
NUM_LAYERS = 2

# ==========================================
# 📊 DATASET
# ==========================================
class WindowDataset(Dataset):
    def __init__(self, pt_file):
        local_pt = os.path.join(LOCAL_DATA, "dataset_v7.pt")
        if not os.path.exists(local_pt): shutil.copy(pt_file, local_pt)
        data = torch.load(local_pt, weights_only=True)
        self.windows, self.targets = data["windows"], data["targets"]
        self.hemis, self.experts = data["hemis"], data["experts"]
        self.num_samples = len(self.targets)
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        return self.hemis[idx], self.experts[idx], self.targets[idx], self.windows[idx]

# ==========================================
# 🚀 IDE 1 DIPERBAIKI: ADDER-NET + ETA SCALING
# Mencegah Mode Collapse pada Router HMoE
# ==========================================
class AdderLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Inisialisasi bobot otak secara normal
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / (in_features**0.5))

        # 🌟 MAGIC FIX: Parameter skala (Suhu) yang bisa dipelajari oleh AI.
        # Dimulai dari 0.05 agar jarak L1 yang besar menjadi sangat kecil di awal training.
        self.eta = nn.Parameter(torch.tensor([0.05]))

    def forward(self, x):
        # 1. Hitung Jarak Absolut (L1 Distance) murni tanpa perkalian
        dist = torch.cdist(x, self.weight, p=1.0)

        # 2. Jinakkan angka yang meledak dengan mengalikannya dengan Eta.
        # Kita menggunakan torch.abs(self.eta) agar skalanya selalu positif,
        # sehingga tidak membalikkan logika "jarak terdekat = terbaik".
        return -dist * torch.abs(self.eta)

# ==========================================
# 🚀 IDE 2: SHIFT-NET (Untuk Expert)
# Membulatkan bobot ke Pangkat 2 Terdekat
# ==========================================
class ShiftLinear(nn.Linear):
    def forward(self, x):
        w_c = torch.clamp(self.weight, -1.0, 1.0)
        # Cari eksponen (pangkat 2 terdekat)
        log_w = torch.round(torch.log2(torch.abs(w_c) + 1e-8))
        log_w = torch.clamp(log_w, -7.0, 0.0) # Resolusi 8-bit shift

        mask = (log_w >= -7.0).float() # Abaikan bobot yg terlalu kecil
        w_q = torch.sign(w_c) * (2.0 ** log_w) * mask

        # Tipuan PyTorch agar bisa backprop
        w_ste = (w_q - self.weight).detach() + self.weight
        return F.linear(x, w_ste, self.bias)

    def get_shift_codes_for_zig(self):
        w_c = torch.clamp(self.weight, -1.0, 1.0)
        log_w = torch.round(torch.log2(torch.abs(w_c) + 1e-8))
        log_w = torch.clamp(log_w, -7.0, 0.0)
        mask = (log_w >= -7.0).float()

        # 🚀 ZIG ENCODER: Menyandikan Pangkat 2 menjadi Angka Kode untuk CPU Zig
        # Contoh: Pangkat 2^-3 akan diekspor sebagai nilai absolut "4" beserta tandanya.
        k = torch.abs(log_w)
        sign = torch.sign(w_c)
        exported = sign * (k + 1.0) * mask
        return exported.detach()

# ==========================================
# 🚀 KOMPONEN: RMSNORM
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class DualBrainBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(HIDDEN_DIM)
        self.time_decay_logit = nn.Parameter(torch.ones(HIDDEN_DIM) * 2.0)
        self.norm2 = RMSNorm(HIDDEN_DIM)

        # EXPERT MENGGUNAKAN IDE 2 (BIT-SHIFT)
        self.exp_calc = ShiftLinear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.exp_sync = ShiftLinear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.exp_sci = ShiftLinear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.exp_story = ShiftLinear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x, experts):
        batch_size, seq_len, _ = x.size()
        x_norm = self.norm1(x)
        decay = torch.sigmoid(self.time_decay_logit)

        state = torch.zeros(batch_size, HIDDEN_DIM, device=x.device)
        state_seq = []
        for t in range(seq_len):
            x_t = x_norm[:, t, :]
            state = (state * decay) + (x_t * (1.0 - decay))
            state_seq.append(state.unsqueeze(1))

        x = x + torch.cat(state_seq, dim=1)
        x_norm_2 = self.norm2(x)
        pool_state = x_norm_2[:, -1, :]
        final_out = torch.zeros(batch_size, HIDDEN_DIM, device=x.device)

        for i, layer in enumerate([self.exp_calc, self.exp_sync, self.exp_sci, self.exp_story]):
            mask = (experts == i)
            if mask.any(): final_out[mask] = F.relu(layer(pool_state[mask]))
        return final_out

class CyberBrainV7(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

        # ROUTER MENGGUNAKAN IDE 1 (ADDER NET L1)
        self.router_l1 = AdderLinear(HIDDEN_DIM, 2)
        self.router_l2_left = AdderLinear(HIDDEN_DIM, 2)
        self.router_l2_right = AdderLinear(HIDDEN_DIM, 2)

        self.layers = nn.ModuleList([DualBrainBlock() for _ in range(NUM_LAYERS)])
        self.final_norm = RMSNorm(HIDDEN_DIM)
        self.lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

    def forward(self, windows, hemis, experts, targets=None):
        x = self.embeddings(windows)
        pool_emb = x[:, -1, :]

        l1_logits = self.router_l1(pool_emb)
        loss_l1 = F.cross_entropy(l1_logits, hemis) if targets is not None else 0

        mask_left, mask_right = (hemis == 0), (hemis == 1)
        loss_l2 = torch.tensor(0.0, device=windows.device)

        if targets is not None:
            if mask_left.any(): loss_l2 += F.cross_entropy(self.router_l2_left(pool_emb[mask_left]), experts[mask_left])
            if mask_right.any(): loss_l2 += F.cross_entropy(self.router_l2_right(pool_emb[mask_right]), experts[mask_right] - 2)

        for layer in self.layers:
            out_state = layer(x, experts)
            x = x + out_state.unsqueeze(1).expand(-1, x.size(1), -1)

        final_state = self.final_norm(x[:, -1, :])
        logits = self.lm_head(final_state)

        if targets is not None:
            loss_lm = F.cross_entropy(logits, targets)
            return loss_lm + (0.1 * loss_l1) + (0.1 * loss_l2), loss_lm
        return logits

def save_checkpoint_v7(model, optimizer, epoch, best_loss):
    state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_loss}
    local_pt = os.path.join(LOCAL_MODELS, CHECKPOINT_NAME)
    torch.save(state, local_pt)
    shutil.copy(local_pt, os.path.join(DRIVE_MODELS, CHECKPOINT_NAME))

def load_checkpoint_v7(model, optimizer, device):
    drive_pt = os.path.join(DRIVE_MODELS, CHECKPOINT_NAME)
    if os.path.exists(drive_pt):
        local_pt = os.path.join(LOCAL_MODELS, CHECKPOINT_NAME)
        shutil.copy(drive_pt, local_pt)
        checkpoint = torch.load(local_pt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Memulihkan riwayat ke Epoch {checkpoint['epoch']+1}")
        return checkpoint['epoch'] + 1, checkpoint['best_loss']
    return 1, float('inf')

# Jalankan cell ini di Colab saat training sedang berjalan untuk menimpa fungsi ekspor
def export_to_zbrain_v7(model):
    local_zb = os.path.join(LOCAL_MODELS, ZBRAIN_NAME)

    with open(local_zb, "wb") as f:
        # 1. PINTU MASUK & ROUTER (Tetap Float32 untuk akurasi presisi L1 Distance)
        f.write(model.embeddings.weight.data.cpu().numpy().astype('float32').flatten().tobytes())
        f.write(model.router_l1.weight.data.cpu().numpy().astype('float32').flatten().tobytes())
        f.write(model.router_l2_left.weight.data.cpu().numpy().astype('float32').flatten().tobytes())
        f.write(model.router_l2_right.weight.data.cpu().numpy().astype('float32').flatten().tobytes())

        # 2. LAPISAN OTAK (Blok Expert dikompresi ke Int8!)
        for layer in model.layers:
            # Norm & Decay (Float32)
            f.write(layer.norm1.weight.data.cpu().numpy().astype('float32').flatten().tobytes())
            f.write(torch.sigmoid(layer.time_decay_logit).detach().cpu().numpy().astype('float32').flatten().tobytes())
            f.write(layer.norm2.weight.data.cpu().numpy().astype('float32').flatten().tobytes())

            # 🚀 EXPERT (INT8 / 1 Byte) - Di sinilah penghematan 75% memori terjadi!
            f.write(layer.exp_calc.get_shift_codes_for_zig().cpu().numpy().astype('int8').flatten().tobytes())
            f.write(layer.exp_sync.get_shift_codes_for_zig().cpu().numpy().astype('int8').flatten().tobytes())
            f.write(layer.exp_sci.get_shift_codes_for_zig().cpu().numpy().astype('int8').flatten().tobytes())
            f.write(layer.exp_story.get_shift_codes_for_zig().cpu().numpy().astype('int8').flatten().tobytes())

        # 3. PINTU KELUAR (Tetap Float32)
        f.write(model.final_norm.weight.data.cpu().numpy().astype('float32').flatten().tobytes())
        f.write(model.lm_head.weight.data.cpu().numpy().astype('float32').flatten().tobytes())

    shutil.copy(local_zb, os.path.join(DRIVE_MODELS, ZBRAIN_NAME))
    print(f"   💾 ZBrain V8.5 (INT8 QUANTIZED) berhasil diekspor!")

# PENTING: Jika Anda tidak sabar menunggu epoch selanjutnya selesai,
# Anda bisa memaksa ekspor sekarang dengan menghapus tanda pagar di bawah ini:
# export_to_zbrain_v7(model)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 === MEMULAI PELATIHAN V7 (HYBRID BPE) {device.type.upper()} ===")

    model = CyberBrainV7().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)

    start_epoch, best_loss = load_checkpoint_v7(model, optimizer, device)
    dataset = WindowDataset(pt_file=DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d}")
        for hemis, experts, targets, windows in pbar:
            hemis, experts, targets, windows = hemis.to(device), experts.to(device), targets.to(device), windows.to(device)
            optimizer.zero_grad()
            loss, lm_loss = model(windows, hemis, experts, targets)
            loss.backward()
            optimizer.step()
            total_loss += lm_loss.item()
            pbar.set_postfix({"loss": f"{lm_loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch} Selesai | Avg Loss: {avg_loss:.4f}")
        save_checkpoint_v7(model, optimizer, epoch, best_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            export_to_zbrain_v7(model)