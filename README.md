# MNIST Handwritten Digit Recognition (Realtime)

Web application Flask untuk mengenali satu digit tulisan tangan secara realtime memakai CNN TensorFlow/Keras. Arsitektur dan proses training mengikuti spesifikasi proyek: dua `Conv2D`, dua `MaxPooling2D`, `Flatten`, dan layer `Dense`.

## Menjalankan aplikasi

1. Install dependency:

   ```bash
   pip install -r requirements.txt
   ```

2. Train model (sekali saja; proses ini mengunduh dataset MNIST jika belum ada):

   ```bash
   python train_model.py
   ```

3. Jalankan web server:

   ```bash
   python app.py
   ```

4. Buka [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Cara kerja

Frontend mengirim PNG canvas setelah debounce 200 ms. Backend membalik gambar (ink hitam pada canvas menjadi putih seperti MNIST), mendeteksi bounding box tulisan, menambah padding, mengubah ukurannya agar muat sekitar 20×20 sambil menjaga rasio, memusatkannya menggunakan center of mass, lalu menormalisasi hasil menjadi `float32` dengan bentuk `(1, 28, 28, 1)`. Probabilitas yang ditampilkan berasal langsung dari `model.predict()`.

`app.py` tidak pernah melatih model. Jika `mnist_cnn.keras` belum ada, jalankan `train_model.py` terlebih dahulu.
