# ═══════════════════════════════════════════════════
# 🔧 GPU RESET & DIAGNOSTIC
# ═══════════════════════════════════════════════════
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    # FIX: total_mem diubah menjadi total_memory di PyTorch baru
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    mem_used = torch.cuda.memory_allocated(0) / 1024**3
    print(f"✅ GPU: {device_name}")
    print(f"   VRAM: {mem_used:.2f}GB / {mem_total:.2f}GB")

    # Test GPU
    try:
        test = torch.randn(10, 10, device='cuda')
        result = test @ test.T
        print("   GPU Test: ✅ BERFUNGSI")
        del test, result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   GPU Test: ❌ GAGAL - {e}")
else:
    print("❌ CUDA tidak tersedia!")