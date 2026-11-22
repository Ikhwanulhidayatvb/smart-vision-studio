# Smart Vision Studio

Sistem Aplikasi Desktop Python (**OpenCV + MediaPipe + CustomTkinter + Pygame + Voice-to-Speech**) dengan fitur:

1. **Kamera PTZ Digital & Auto-Tracking**
   - **Auto Face**: Kamera otomatis zoom dan bergerak halus mengikuti posisi wajah Anda.
   - **Auto Hand**: Kamera otomatis mengikuti gerakan tangan Anda.
   - **Auto Hybrid**: Mengikuti tangan saat aktif berinteraksi, atau wajah saat tangan tidak terlihat.
   - **Manual PTZ**: Kontrol geser (Atas/Bawah/Kiri/Kanan), Zoom In/Out, dan Reset via tombol UI, tombol panah keyboard, atau perintah suara.
   - Dilengkapi **PTZ Sensor Minimap** di sudut kanan bawah kamera.

2. **Pembaca Gerakan Tangan & Kontrol Musik Tanpa Sentuh**
   - **Telapak Terbuka (5 Jari)** *(tahan ~0.5 detik)* → Play / Pause Musik
   - **Kepalan Tangan (0 Jari)** → Pause Musik
   - **Dua Jari / Peace (✌️) atau Swipe Kanan** → Lagu Berikutnya (`Next Track`)
   - **Tiga Jari (🤟) atau Swipe Kiri** → Lagu Sebelumnya (`Previous Track`)
   - **Telunjuk ke Atas (☝️)** → Volume Naik (+10%)
   - **Jempol ke Bawah (👎)** → Volume Turun (-10%)
   - **Pinch / Cubit (👌) + Gerak Naik/Turun** → Pengaturan Volume Halus (0–100%)
   - **Jempol ke Atas (👍)** → Umumkan Judul Lagu via Suara
   - **Rock (🤘)** → Mute / Unmute

3. **Pembaca Gerakan & Ekspresi Muka**
   - Mendeteksi ekspresi real-time: **Senyum**, **Senyum Lebar**, **Mulut Terbuka**, **Alis Terangkat**, **Kedip Mata Kiri/Kanan/Kedua Mata**, serta penghitung kedipan.
   - Mendeteksi gerakan kepala 3D: **Menoleh Kiri/Kanan (Yaw)**, **Mendongak/Menunduk (Pitch)**, **Miring Kiri/Kanan (Roll)**, serta **Anggukan (`NOD`)** dan **Gelengan (`SHAKE`)**.

4. **Voice-to-Speech Lengkap (STT + TTS)**
   - **Speech-to-Text (`id-ID` & `en-US`)**: Mode Push-to-Talk (`🎙️ Rekam Suara`) atau `Dengar Otomatis` (Voice Activity Detection).
   - **Perintah Suara**: *"Putar musik"*, *"Pause"*, *"Lagu selanjutnya"*, *"Volume naik"*, *"Ikuti wajah"*, *"Zoom in"*, *"Geser kiri"*, *"Reset kamera"*, *"Katakan <teks>"*.
   - **Mode Echo TTS**: Mengulang setiap ucapan pengguna menjadi suara sistem secara langsung.
   - **Text-to-Speech Box**: Ketik teks dan ucapkan langsung (`Auto gTTS -> Offline SAPI5`), lengkap dengan **Audio Ducking** (volume musik otomatis mengecil sementara saat sistem berbicara).

5. **Pemutar Musik Siap Pakai**
   - Otomatis menyediakan 4 lagu demo sintetis (`Lo-Fi Chill`, `Neon Synthwave`, `Ambient Zen`, `Upbeat Groove`) sehingga langsung bisa dicoba.
   - Mendukung penambahan file **MP3 / WAV / OGG / FLAC** atau pemilihan satu folder musik.

## Cara Menjalankan

Klik dua kali `run.bat`, atau jalankan lewat terminal:

```powershell
cd d:\projects\smart-vision-studio
.\.venv\Scripts\python.exe main.py
```
