import os
import shutil
from google.colab import drive

# =================================================================
# 🔄 MOUNT GOOGLE DRIVE & SETUP PATHS
# =================================================================
print("🔄 Menghubungkan ke Google Drive...")
drive.mount('/content/drive')

# Path utama di Google Drive
DRIVE_BASE = "/content/drive/MyDrive/Dual-Brain"
DRIVE_DATA = os.path.join(DRIVE_BASE, "data")
DRIVE_MODELS = os.path.join(DRIVE_BASE, "models")

# Buat folder jika belum ada
os.makedirs(DRIVE_DATA, exist_ok=True)
os.makedirs(DRIVE_MODELS, exist_ok=True)

# Path lokal di Colab (untuk kecepatan training I/O)
LOCAL_BASE = "/content/workspace"
LOCAL_DATA = os.path.join(LOCAL_BASE, "data")
LOCAL_MODELS = os.path.join(LOCAL_BASE, "models")
os.makedirs(LOCAL_DATA, exist_ok=True)
os.makedirs(LOCAL_MODELS, exist_ok=True)

print("✅ Google Drive terhubung!")
print(f"📂 Drive Path: {DRIVE_BASE}")