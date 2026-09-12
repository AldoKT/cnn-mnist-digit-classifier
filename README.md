# MNIST Handwritten Digit Recognition

A browser-based handwritten digit recognition application powered by a convolutional neural network and TensorFlow.js. Users draw a digit from 0 to 9 on an HTML canvas and receive a client-side prediction with probabilities for all ten classes.

## Links

- **Live demo:** [aldokt.github.io/cnn-mnist-digit-classifier](https://aldokt.github.io/cnn-mnist-digit-classifier/)
- **Repository:** [github.com/AldoKT/cnn-mnist-digit-classifier](https://github.com/AldoKT/cnn-mnist-digit-classifier)

## Results

The existing trained model was evaluated on all 10,000 MNIST test samples:

| Metric                   | Result          |
| -------------------------| ---------------:|
| Test accuracy            | **99.00%**      |
| Correct predictions      | 9,900           |
| Incorrect predictions    | 100             |
| Test samples             | 10,000          |

The most frequent confusion pairs were:

| Actual → predicted | Errors |
| -------------------| -----: |
|      2 → 7         | 10     |
|      9 → 5         | 9      |
|      9 → 4         | 6      |
|      4 → 9         | 6      |
|      3 → 5         | 5      |
|      9 → 7         | 5      |
|      8 → 9         | 5      |

Some incorrect predictions had very high softmax confidence. This shows that confidence is not a guarantee of correctness when handwritten forms are visually ambiguous.

## Overview

The project covers the full path from model training to browser deployment:

1. Train a CNN on the MNIST dataset with TensorFlow/Keras.
2. Export the trained model to TensorFlow.js artifacts.
3. Load the model in the browser with TensorFlow.js.
4. Preprocess canvas input into the model's expected format.
5. Display the predicted digit and the probability distribution.

Inference runs entirely in the browser. No backend API is required.

## How It Works

The user draws on an HTML canvas. The browser then:

- Inverts the foreground/background polarity.
- Detects the drawing's bounding box.
- Adds padding around the digit.
- Resizes the digit proportionally.
- Centers the digit using its center of mass.
- Normalizes pixel values to the model's expected range.
- Creates a `float32` tensor with shape `(1, 28, 28, 1)`.

The TensorFlow.js model returns a softmax distribution over digits 0 through 9.

## Model Architecture

| Layer | Configuration |
| --- | --- |
| Input | `28 x 28 x 1` grayscale image |
| Conv2D | 16 filters, `3 x 3`, ReLU |
| MaxPooling2D | `2 x 2` |
| Conv2D | 32 filters, `3 x 3`, ReLU |
| MaxPooling2D | `2 x 2` |
| Flatten | 800 values |
| Dense | 128 units, ReLU |
| Output | 10 units, Softmax |

The model has **108,618 parameters** and predicts ten classes: digits 0 through 9.

## Training

Training uses:

- **Dataset:** MNIST
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 20
- **Validation split:** 10%
- **Pixel normalization:** divide by `255.0`
- **Input shape:** `28 x 28 x 1`

The current training script defines a fixed random seed and an explicit batch size for future reproducibility. The existing `mnist_cnn.keras` artifact that produced the reported 99.00% result was trained before those reproducibility settings were added. The reported result should therefore not be interpreted as the result of the newer seed and batch-size configuration.

## Evaluation and Error Analysis

The model was evaluated on the standard MNIST test split without retraining:

- 9,900 correct predictions
- 100 incorrect predictions
- 99.00% overall accuracy

The most common errors involve visually similar handwritten forms, particularly `2 → 7`, `9 → 5`, and `9 → 4`. The high-confidence errors are a useful reminder that softmax probability is a model score, not a calibrated correctness guarantee.

## TensorFlow.js Model Validation

The existing TensorFlow.js weights were compared with the original Keras model weights:

| Check | Result |
| --- | ---: |
| Keras tensors | 8 |
| TensorFlow.js tensors | 8 |
| Exact tensor matches | 8/8 |
| Values compared | 108,618 |
| Mismatches | 0 |
| Maximum absolute difference | 0.0 |
| Maximum relative difference | 0.0 |

This confirms that the exported `float32` weight values exactly match the original Keras model weights. Full Keras-versus-TensorFlow.js inference parity has not been formally validated.

## Browser Inference

The static application loads the TensorFlow.js model from `model/model.json` and its binary weight shard. Inference runs entirely in the browser as the user draws. The interface displays:

- The predicted digit.
- A model loading/prediction status.
- A probability bar for each digit from 0 to 9.
- A brush-size control and clear action.

## Project Structure

```text
.
├── convert_model.py          # Model development script for TensorFlow.js export
├── index.html                # Browser application shell
├── mnist_cnn.keras           # Existing trained Keras model
├── model/
│   ├── model.json            # TensorFlow.js model topology and manifest
│   └── group1-shard1of1.bin  # TensorFlow.js model weights
├── requirements.txt          # Python dependencies
├── static/
│   ├── script.js             # Canvas input, preprocessing, and inference
│   └── style.css             # Application styling
└── train_model.py            # Model development script for CNN training/evaluation
```

## Running Locally

The existing browser demo uses the committed TensorFlow.js files in `model/`. Start a local static server from the project root:

```powershell
python -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000) in a browser. Training and model conversion are not required to run the existing demo.

`train_model.py` and `convert_model.py` are model development scripts for training and exporting model artifacts. They are separate from the steps required to run the committed browser demo.

## Limitations

- The model is trained and evaluated on MNIST, so performance on arbitrary real-world handwriting is not represented by the 99.00% test accuracy.
- Browser canvas preprocessing approximates the MNIST input format; it is not the original MNIST data-generation process.
- Softmax confidence is not a formal uncertainty estimate.
- The current documentation reports exact weight equivalence between the Keras and TensorFlow.js artifacts, but not full inference parity across runtimes.
- The dependency versions are not pinned in `requirements.txt`.

## What I Learned

- A compact CNN can provide strong MNIST performance while remaining practical for browser inference.
- Preprocessing is an important part of deploying a model beyond its original dataset format.
- Export validation should check both tensor values and runtime prediction behavior.
- High classification accuracy can still hide systematic errors between visually similar classes.
