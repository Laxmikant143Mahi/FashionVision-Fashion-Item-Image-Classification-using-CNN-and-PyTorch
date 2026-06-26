# FashionVision – Fashion Item Image Classification using CNN and PyTorch

FashionVision is a beginner-friendly computer vision project designed to classify clothing items from the popular **FashionMNIST** dataset. Using **PyTorch**, this project builds, trains, tests, and evaluates a simple 2-layer Convolutional Neural Network (CNN).

This project is tailored specifically for beginner AI/ML engineers. The code is kept clean, highly structured, well-commented, and avoids complex object-oriented design or advanced abstract wrappers.

---

## 📂 Folder Structure

```text
FashionVision/
│
├── dataset/                  # Automatically downloaded FashionMNIST dataset
├── saved_models/             # Holds the saved training weights
│      └── best_model.pth     # Best model weights based on test accuracy
│
├── utils/
│      └── class_names.py     # Class labels dictionary mapping integers to names
│
├── model.py                  # Simple CNN architecture using nn.Module
├── train.py                  # Training pipeline with validation checks
├── test.py                   # Testing script that measures overall accuracy
├── predict.py                # Single-image inference and plotting script
├── requirements.txt          # File containing list of required packages
├── README.md                 # Project documentation
└── screenshots/              # Folder to save output charts and prediction samples
```

---

## 🛠️ Technologies Used

* **Python**: Core programming language.
* **PyTorch**: Deep learning framework used for network building, autograd, and dataset handling.
* **Torchvision**: Package containing popular datasets, model architectures, and image transformations.
* **Matplotlib**: Plotting library used to visualize classification results.

---

## 🧠 CNN Architecture

The model architecture is built using a sequential-like structure inside PyTorch's `nn.Module`. Below are the details of each layer:

| Layer | Input Size | Details | Output Size | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Input Image** | `(1, 28, 28)` | Grayscale pixel values | `(1, 28, 28)` | Grayscale 28x28 FashionMNIST image. |
| **Conv2D (1)** | `(1, 28, 28)` | 32 filters, Kernel 3x3, stride 1 | `(32, 26, 26)` | Calculates features, size = `(28 - 3)/1 + 1 = 26`. |
| **ReLU (1)** | `(32, 26, 26)` | Rectified Linear Unit activation | `(32, 26, 26)` | Introduces non-linearity. |
| **MaxPool2D (1)**| `(32, 26, 26)` | Filter size 2x2, stride 2 | `(32, 13, 13)` | Downsamples image height and width by half. |
| **Conv2D (2)** | `(32, 13, 13)` | 64 filters, Kernel 3x3, stride 1 | `(64, 11, 11)` | Learns higher-level shapes, size = `(13 - 3)/1 + 1 = 11`. |
| **ReLU (2)** | `(64, 11, 11)` | Rectified Linear Unit activation | `(64, 11, 11)` | Introduces non-linearity. |
| **MaxPool2D (2)**| `(64, 11, 11)` | Filter size 2x2, stride 2 | `(64, 5, 5)` | Downsamples to 5x5, size = `floor((11 - 2)/2) + 1 = 5`. |
| **Flatten** | `(64, 5, 5)` | Reshapes image to 1D vector | `(1600)` | Converts 3D grid to 1D array (`64 * 5 * 5 = 1600`). |
| **Linear (1)** | `(1600)` | 128 output neurons | `(128)` | Fully connected layer summarizing representations. |
| **ReLU (3)** | `(128)` | Activation function | `(128)` | Non-linearity before classification. |
| **Linear (2)** | `(128)` | 10 output classes | `(10)` | Computes logits for the 10 clothing categories. |

---

## ⚡ Installation

1. Clone or download this project workspace.
2. Open your terminal in the `FashionVision` folder.
3. Install the required libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Step 1: Train the Model
Run `train.py` to download the dataset, train the network for 10 epochs, and automatically save the best model weights.
```bash
python train.py
```

### Step 2: Test the Model
Run `test.py` to evaluate the saved model's overall accuracy on the 10,000 unseen test images.
```bash
python test.py
```

### Step 3: Run Single Image Prediction
Run `predict.py` to pick a random image from the testing set, display prediction results in the console, and open a Matplotlib visualization window.
```bash
python predict.py
```

---

## 📈 Results

* **Optimizer**: Adam (Learning Rate = `0.001`)
* **Loss Function**: CrossEntropyLoss
* **Training Epochs**: 10
* **Expected Test Accuracy**: ~90.00% to 92.00% on FashionMNIST test set.

When running `predict.py`, you will see output like this:
```text
=========================================
FashionVision Prediction
=========================================
Actual Label      : Sneaker
Predicted Label   : Sneaker
Confidence        : 98.76%
=========================================
Saved visualization to 'screenshots/prediction_sample.png'
```

---

## 🔮 Future Improvements

1. **Data Augmentation**: Implement random horizontal flips or rotations using `torchvision.transforms` to improve generalization.
2. **Batch Normalization**: Add `nn.BatchNorm2d` after conv layers to stabilize training and speed up convergence.
3. **Dropout Layers**: Introduce regularizing dropout layers (`nn.Dropout`) to reduce model overfitting.
4. **Learning Rate Decay**: Implement dynamic learning rate adjustments to refine weights as training progresses.
