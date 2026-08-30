"""Flask server for realtime handwritten digit recognition."""
import base64
import io
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_cnn.keras")
model = None
model_error = None


def load_model():
    """Load only a previously trained model; never train from the web server."""
    global model, model_error
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model_error = None
        print(f"Loaded model: {MODEL_PATH}")
    except Exception as exc:  # Keep the page usable and expose a helpful API error.
        model = None
        model_error = str(exc)
        print(f"Could not load MNIST model: {exc}")


def decode_canvas(data_url):
    """Decode a browser PNG data URL into a grayscale Pillow image."""
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Invalid image format. Expected a PNG data URL.")
    try:
        encoded = data_url.split(",", 1)[1]
        raw_image = base64.b64decode(encoded, validate=True)
        return Image.open(io.BytesIO(raw_image)).convert("L")
    except Exception as exc:
        raise ValueError("Unable to read the submitted canvas image.") from exc


def preprocess_image(canvas_image):
    """Convert black-on-white canvas pixels to centered MNIST-like 28x28 input."""
    gray = np.asarray(canvas_image, dtype=np.uint8)

    # The UI has black ink on white. MNIST has white ink on black, hence inversion.
    inverted = 255 - gray
    mask = inverted > 20
    if not np.any(mask):
        raise ValueError("Canvas is empty. Draw a digit before predicting.")

    ys, xs = np.where(mask)
    # Small padding retains the edge of strokes after resampling.
    pad = 4
    left, right = max(0, xs.min() - pad), min(inverted.shape[1], xs.max() + pad + 1)
    top, bottom = max(0, ys.min() - pad), min(inverted.shape[0], ys.max() + pad + 1)
    cropped = inverted[top:bottom, left:right]

    height, width = cropped.shape
    scale = 20.0 / max(height, width)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = np.asarray(
        Image.fromarray(cropped).resize((new_width, new_height), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )

    # Put the resized glyph in the middle first, then align its intensity centroid.
    output = np.zeros((28, 28), dtype=np.float32)
    y0 = (28 - new_height) // 2
    x0 = (28 - new_width) // 2
    output[y0:y0 + new_height, x0:x0 + new_width] = resized

    total = output.sum()
    if total > 0:
        y_coords, x_coords = np.indices(output.shape)
        center_y = float((y_coords * output).sum() / total)
        center_x = float((x_coords * output).sum() / total)
        shift_y, shift_x = int(round(13.5 - center_y)), int(round(13.5 - center_x))
        aligned = np.zeros_like(output)
        src_y0, src_y1 = max(0, -shift_y), min(28, 28 - shift_y)
        src_x0, src_x1 = max(0, -shift_x), min(28, 28 - shift_x)
        dst_y0, dst_y1 = max(0, shift_y), min(28, 28 + shift_y)
        dst_x0, dst_x1 = max(0, shift_x), min(28, 28 + shift_x)
        aligned[dst_y0:dst_y1, dst_x0:dst_x1] = output[src_y0:src_y1, src_x0:src_x1]
        output = aligned

    return (output / 255.0).astype(np.float32).reshape(1, 28, 28, 1)


@app.get("/")
def index():
    return render_template("index.html", model_ready=model is not None)


@app.post("/predict")
def predict():
    if model is None:
        return jsonify(error="Model belum tersedia. Jalankan python train_model.py terlebih dahulu."), 503

    payload = request.get_json(silent=True) or {}
    try:
        image = preprocess_image(decode_canvas(payload.get("image")))
        prediction = model.predict(image, verbose=0)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = [float(value) for value in prediction[0]]
        return jsonify(prediction=predicted_digit, probabilities=confidence)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Server error while processing the image."), 500


load_model()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
