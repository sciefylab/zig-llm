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
# 🔄 SETUP PATHS
# =================================================================
if COLAB_MODE: drive.mount('/content/drive')

DRIVE_BASE = "/content/drive/MyDrive/Dual-Brain" if COLAB_MODE else "./drive_mock"
DRIVE_DATA = os.path.join(DRIVE_BASE, "data")
DRIVE_MODELS = os.path.join(DRIVE_BASE, "models")
os.makedirs(DRIVE_DATA, exist_ok=True)
os.makedirs(DRIVE_MODELS, exist_ok=True)

LOCAL_BASE = "/content/workspace" if COLAB_MODE else "./local_workspace"
LOCAL_DATA = os.path.join(LOCAL_BASE, "data")
LOCAL_MODELS = os.path.join(LOCAL_BASE, "models")
os.makedirs(LOCAL_DATA, exist_ok=True)
os.makedirs(LOCAL_MODELS, exist_ok=True)

# Dataset tetap pakai V5, hanya arsitektur model (V6) yang berubah
DATASET_PATH = os.path.join(DRIVE_DATA, "dataset_v5.pt") 
CHECKPOINT_NAME = "checkpoint_v6.pth"
ZBRAIN_NAME = "real_dual_brain_v6.zbrain"

# ==========================================
# ⚙️ KONFIGURASI V6 (DEEP ARCHITECTURE)
# ==========================================
HIDDEN_DIM = 1024
VOCAB_SIZE = 15000
BATCH_SIZE = 2048   # 🚀 Standar Industri untuk Deep Sequence agar GPU tidak OOM
EPOCHS = 20
NUM_LAYERS = 2      # 🚀 2 Lapis Pemrosesan!

# ==========================================
# 📊 DATASET
# ==========================================
class WindowDataset(Dataset):
    def __init__(self, pt_file):
        local_pt = os.path.join(LOCAL_DATA, "dataset_v5.pt")
        if not os.path.exists(local_pt): shutil.copy(pt_file, local_pt)
        data = torch.load(local_pt, weights_only=True)
        self.windows, self.targets = data["windows"], data["targets"]
        self.hemis, self.experts = data["hemis"], data["experts"]
        self.num_samples = len(self.targets)

    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        return self.hemis[idx], self.experts[idx], self.targets[idx], self.windows[idx]

# ==========================================
# 🚀 KOMPONEN: RMSNORM
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Mencegah angka meledak dengan menormalkan varians
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

# ==========================================
# 🧱 KOMPONEN: SATU LAPIS OTAK (BLOCK)
# ==========================================
class DualBrainBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(HIDDEN_DIM)
        self.time_decay_logit = nn.Parameter(torch.ones(HIDDEN_DIM) * 2.0)
        
        self.norm2 = RMSNorm(HIDDEN_DIM)
        self.exp_calc = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.exp_sync = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.exp_sci = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.exp_story = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x, experts):
        batch_size, seq_len, _ = x.size()
        
        # 1. Normalisasi sebelum masuk ke memori (State)
        x_norm = self.norm1(x)
        decay = torch.sigmoid(self.time_decay_logit)
        
        state = torch.zeros(batch_size, HIDDEN_DIM, device=x.device)
        state_seq = []
        
        # EMA Loop
        for t in range(seq_len):
            x_t = x_norm[:, t, :]
            state = (state * decay) + (x_t * (1.0 - decay))
            state_seq.append(state.unsqueeze(1))
        
        # Residual Connection (Kabel bypass agar tidak lupa diri)
        x = x + torch.cat(state_seq, dim=1)
        
        # 2. Normalisasi sebelum masuk ke Expert
        x_norm_2 = self.norm2(x)
        
        # Karena x_norm_2 adalah [Batch, Seq, Dim], kita ambil state akhirnya saja
        pool_state = x_norm_2[:, -1, :] 
        final_out = torch.zeros(batch_size, HIDDEN_DIM, device=x.device)
        
        for i, layer in enumerate([self.exp_calc, self.exp_sync, self.exp_sci, self.exp_story]):
            mask = (experts == i)
            if mask.any(): 
                final_out[mask] = F.relu(layer(pool_state[mask]))
                
        return final_out

# ==========================================
# 🧠 ARSITEKTUR V6 (DEEP HMoE)
# ==========================================
class CyberBrainV6(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        
        # Router Task-Level
        self.router_l1 = nn.Linear(HIDDEN_DIM, 2, bias=False)
        self.router_l2_left = nn.Linear(HIDDEN_DIM, 2, bias=False)
        self.router_l2_right = nn.Linear(HIDDEN_DIM, 2, bias=False)

        # Menumpuk Block sesuai NUM_LAYERS
        self.layers = nn.ModuleList([DualBrainBlock() for _ in range(NUM_LAYERS)])
        
        self.final_norm = RMSNorm(HIDDEN_DIM)
        self.lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

    def forward(self, windows, hemis, experts, targets=None):
        x = self.embeddings(windows) 
        
        # Router Loss Logic
        pool_emb = x[:, -1, :] 
        l1_logits = self.router_l1(pool_emb)
        loss_l1 = F.cross_entropy(l1_logits, hemis) if targets is not None else 0

        mask_left, mask_right = (hemis == 0), (hemis == 1)
        loss_l2 = torch.tensor(0.0, device=windows.device)

        if targets is not None:
            if mask_left.any():
                loss_l2 += F.cross_entropy(self.router_l2_left(pool_emb[mask_left]), experts[mask_left])
            if mask_right.any():
                loss_l2 += F.cross_entropy(self.router_l2_right(pool_emb[mask_right]), experts[mask_right] - 2)

        # 🚀 FORWARD PASS MELEWATI MULTI-LAYER
        for layer in self.layers:
            out_state = layer(x, experts)
            x = x + out_state.unsqueeze(1).expand(-1, x.size(1), -1) 
            
        final_state = self.final_norm(x[:, -1, :])
        logits = self.lm_head(final_state)
        
        if targets is not None:
            loss_lm = F.cross_entropy(logits, targets)
            return loss_lm + (0.1 * loss_l1) + (0.1 * loss_l2), loss_lm
        return logits

# ==========================================
# 💾 FUNGSI CHECKPOINT & EXPORT ZBRAIN (V6)
# ==========================================
def save_checkpoint_v6(model, optimizer, epoch, best_loss):
    state = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 
        'best_loss': best_loss
    }
    local_pt = os.path.join(LOCAL_MODELS, CHECKPOINT_NAME)
    torch.save(state, local_pt)
    shutil.copy(local_pt, os.path.join(DRIVE_MODELS, CHECKPOINT_NAME))

def load_checkpoint_v6(model, optimizer, device):
    drive_pt = os.path.join(DRIVE_MODELS, CHECKPOINT_NAME)
    if os.path.exists(drive_pt):
        local_pt = os.path.join(LOCAL_MODELS, CHECKPOINT_NAME)
        shutil.copy(drive_pt, local_pt)
        checkpoint = torch.load(local_pt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Memulihkan riwayat V6 ke Epoch {checkpoint['epoch']+1}")
        return checkpoint['epoch'] + 1, checkpoint['best_loss']
    return 1, float('inf')

def export_to_zbrain_v6(model):
    local_zb = os.path.join(LOCAL_MODELS, ZBRAIN_NAME)
    
    weights = [
        model.embeddings.weight.data,
        model.router_l1.weight.data,
        model.router_l2_left.weight.data,
        model.router_l2_right.weight.data
    ]
    
    # 🚀 Dinamis mengekstrak bobot dari setiap Lapis (Layer)
    for layer in model.layers:
        weights.extend([
            layer.norm1.weight.data,
            torch.sigmoid(layer.time_decay_logit).detach(),
            layer.norm2.weight.data,
            layer.exp_calc.weight.data,
            layer.exp_sync.weight.data,
            layer.exp_sci.weight.data,
            layer.exp_story.weight.data
        ])
        
    weights.extend([
        model.final_norm.weight.data,
        model.lm_head.weight.data
    ])
    
    with open(local_zb, "wb") as f:
        for w in weights: f.write(w.cpu().numpy().astype('float32').flatten().tobytes())
    shutil.copy(local_zb, os.path.join(DRIVE_MODELS, ZBRAIN_NAME))
    print(f"   💾 ZBrain V6 berhasil diekspor!")

# ==========================================
# 🚀 KERNEL TRAINING UTAMA
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 === MEMULAI PELATIHAN V6 (DEEP HMoE) {device.type.upper()} ===")
    
    model = CyberBrainV6().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Memuat Checkpoint (Jika colab diskonek, otomatis lanjut)
    start_epoch, best_loss = load_checkpoint_v6(model, optimizer, device)
    
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
        
        # Simpan progress setiap 1 epoch selesai
        save_checkpoint_v6(model, optimizer, epoch, best_loss)
        
        if avg_loss < best_loss:
            print(f"   🏆 REKOR BARU: {best_loss:.4f} -> {avg_loss:.4f}")
            best_loss = avg_loss
            export_to_zbrain_v6(model)