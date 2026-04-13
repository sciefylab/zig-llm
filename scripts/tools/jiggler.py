import pyautogui
import time

# Mengaktifkan fitur fail-safe (gerakkan mouse ke sudut layar untuk mematikan paksa)
pyautogui.FAILSAFE = True

# Jeda waktu dalam detik (misalnya: 60 detik = 1 menit)
JEDA_WAKTU = 60 

print("Mouse Jiggler sedang berjalan...")
print("Tekan [Ctrl + C] di terminal untuk berhenti, atau gerakkan mouse ke sudut layar dengan cepat.")

try:
    while True:
        # Mendapatkan posisi mouse saat ini
        x, y = pyautogui.position()
        
        # Menggerakkan mouse 5 piksel ke kanan
        pyautogui.moveTo(x + 5, y, duration=0.2)
        
        # Memberi jeda kecil agar terlihat lebih natural (opsional)
        time.sleep(0.5)
        
        # Menggerakkan mouse kembali ke posisi awal (5 piksel ke kiri)
        pyautogui.moveTo(x, y, duration=0.2)
        
        # Menunggu selama waktu yang ditentukan sebelum bergerak lagi
        time.sleep(JEDA_WAKTU)

except KeyboardInterrupt:
    print("\nMouse Jiggler dihentikan oleh pengguna.")