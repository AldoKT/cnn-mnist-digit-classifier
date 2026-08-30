"""Export the trained Keras CNN to browser-ready TensorFlow.js assets."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parent
KERAS_MODEL = PROJECT_DIR / "mnist_cnn.keras"
OUTPUT_DIR = PROJECT_DIR / "model"


def main():
    if not KERAS_MODEL.exists():
        raise FileNotFoundError(
            "mnist_cnn.keras tidak ditemukan. Jalankan `python train_model.py` terlebih dahulu."
        )

    model = tf.keras.models.load_model(KERAS_MODEL)
    OUTPUT_DIR.mkdir(exist_ok=True)
    # TensorFlow.js Layers uses the same Keras model topology and tensor layout
    # for these standard Conv2D/Dense layers. Store the weights sequentially in
    # one binary shard and describe them in the standard TFJS manifest format.
    weights = []
    binary_weights = []
    for variable in model.weights:
        values = variable.numpy().astype(np.float32)
        weights.append({
            "name": variable.name.replace(":0", ""),
            "shape": list(values.shape),
            "dtype": "float32",
        })
        binary_weights.append(values.tobytes(order="C"))

    topology = json.loads(model.to_json())
    metadata = {
        "format": "layers-model",
        "generatedBy": "TensorFlow/Keras MNIST exporter",
        "convertedBy": "MNIST GitHub Pages exporter",
        "modelTopology": topology,
        "weightsManifest": [{
            "paths": ["group1-shard1of1.bin"],
            "weights": weights,
        }],
    }
    (OUTPUT_DIR / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
    (OUTPUT_DIR / "group1-shard1of1.bin").write_bytes(b"".join(binary_weights))
    print(f"TensorFlow.js model created in: {OUTPUT_DIR}")
    print("Commit model/model.json dan file .bin yang dibuat ke GitHub.")


if __name__ == "__main__":
    main()
