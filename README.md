# MNIST Handwritten Digit Recognition (Realtime)

Aplikasi pengenalan digit 0–9 yang berjalan sepenuhnya di browser. UI memakai canvas HTML dan TensorFlow.js; tidak ada Flask atau endpoint API saat dipublikasikan, sehingga proyek dapat di-host di GitHub Pages.

## Menyiapkan model web

Training tetap memakai arsitektur CNN Keras dalam `train_model.py` dan hanya dilakukan sekali.

```powershell
pip install -r requirements.txt
python train_model.py
python convert_model.py
```

`convert_model.py` membuat `model/model.json` dan file bobot `.bin`. Commit semua file tersebut ke GitHub karena browser memuatnya untuk prediksi.

## Publish di GitHub Pages

1. Push project termasuk isi `model/`:

   ```powershell
   git add .
   git commit -m "Add TensorFlow.js browser model"
   git push
   ```

2. Di repository GitHub, buka **Settings** → **Pages**.
3. Pilih **Deploy from a branch**, branch `main`, dan folder `/(root)`, lalu klik **Save**.
4. Link biasanya menjadi `https://Juliaan77.github.io/cnn-mnist/`.

## Cara kerja realtime

Setelah jeda 200 ms, JavaScript membalik ink hitam menjadi putih seperti MNIST, crop bounding box, resize proporsional ke sekitar 20×20, memusatkan digit dengan center of mass, lalu membuat tensor `float32` berbentuk `(1, 28, 28, 1)`. Tensor tersebut diberikan langsung ke `model.predict()` TensorFlow.js, dan output softmax mengisi probability bar 0–9.
