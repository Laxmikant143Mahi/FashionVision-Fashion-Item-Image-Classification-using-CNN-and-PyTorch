# FashionVision – Interview Preparation & Running Guide

This guide is designed to help you open, run, and confidently explain the **FashionVision** project in your technical interviews. It contains step-by-step running instructions, a conceptual overview, and sample interview questions.

---

## 🚀 Part 1: How to Open & Run the Project

The most common error is running python commands from the wrong folder. Follow these steps exactly:

### Step 1: Open Terminal or Command Prompt
Open your terminal (PowerShell, Command Prompt, or Git Bash).

### Step 2: Navigate to the Project Root Folder
Navigate to the directory where the code is stored. On your system, run this command:
```bash
cd "c:\Users\mahin\Downloads\pytorch project\FashionVision"
```
*(Note: Do NOT run commands in `pytorch project` folder directly. You must change directory into the `FashionVision` subfolder).*

### Step 3: Install Dependencies
Install all the required Python libraries using pip:
```bash
pip install -r requirements.txt
```

### Step 4: Run Training Script
Train the Convolutional Neural Network (CNN) for 10 epochs. It will download the dataset automatically to the `dataset/` folder and save the best model weights as `saved_models/best_model.pth`:
```bash
python train.py
```

### Step 5: Run Evaluation Script
Test the trained model weights on 10,000 unseen test images and verify the final classification accuracy:
```bash
python test.py
```

### Step 6: Run Single Image Inference (Prediction)
Pick a random image from the test set, output labels and confidence to the console, and open a Matplotlib visualizer:
```bash
python predict.py
```
*(If running on a server or automated environment with no monitor, you can run `python predict.py --headless` to save the plot without popping up a GUI window).*

---

## 🔍 Part 2: What the Project Does

### 1. The Core Objective
The project classifies small $28\times28$ grayscale images of fashion items into one of 10 clothing categories. This is a **supervised multi-class classification task**.

### 2. The Dataset (FashionMNIST)
* **What is it?** A modern replacement for the traditional digits MNIST dataset, consisting of Zalando's article images.
* **Size**: 70,000 grayscale images (60,000 for training, 10,000 for testing).
* **Categories**: 10 classes (T-shirt/Top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot).
* **Format**: Grayscale images with a single channel (1x28x28).

### 3. The Model (SimpleCNN)
* A **Convolutional Neural Network (CNN)** is a type of deep neural network designed to extract spatial features from visual inputs (like images).
* Instead of looking at pixels individually (like a standard feedforward neural network), a CNN uses a **sliding filter (Kernel)** to detect shapes, edges, and texture patterns.

---

## 🧠 Part 3: Deep Learning Concepts Explained Simply

| Concept | Simple Definition | How it is used in our project |
| :--- | :--- | :--- |
| **Tensor** | A multi-dimensional array (like a list of lists) that PyTorch uses to store data and calculate gradients on GPUs/CPUs. | Images are converted from PIL images into PyTorch Tensors using `transforms.ToTensor()`. |
| **DataLoader** | A helper that groups images into mini-batches, shuffles them, and feeds them to the model during training. | We load images in batches of `64`. Shuffling is `True` for training and `False` for testing. |
| **Convolution (Conv2D)** | A matrix sliding over an image, multiplying values, and summing them to detect features like lines, edges, or texture. | `Conv2d(1, 32, kernel_size=3)` extracts 32 distinct visual features from our grayscale input. |
| **ReLU (Activation)** | An activation function that outputs input directly if positive; otherwise, it outputs zero ($f(x) = \max(0, x)$). | Introduces non-linearity, allowing the model to learn complex shapes rather than basic linear equations. |
| **MaxPool2D** | A pooling layer that slides a grid (e.g. 2x2) over the image and keeps only the maximum pixel value. | Reduces height/width by half (downsampling), saving memory and making the model invariant to minor translations. |
| **Logits** | The raw, unnormalized scores outputted by the final layer of the neural network. | The outputs of `model(images)` are logits (10 numbers). |
| **Softmax** | A function that converts logits into probability percentages that sum up to 100%. | Used in `predict.py` to calculate the prediction confidence score. |
| **CrossEntropyLoss** | The formula that measures the difference between the model's predicted logits and actual correct labels. | Calculates our training error. It penalizes the network heavily if it makes incorrect predictions with high confidence. |
| **Adam Optimizer** | An algorithm that dynamically updates model weights to reduce the training loss. | Automatically adjusts learning rates for different parameters as training proceeds. |

---

## 💬 Part 4: Top 15 Interview Q&As for Freshers

### Q1: What is the folder structure of your project, and why did you divide it this way?
**Answer**: 
Our project is modularized into:
* `model.py`: Neural network structure definition.
* `train.py`: Training loop script.
* `test.py`: Evaluating accuracy script.
* `predict.py`: Runs a single image prediction.
* `utils/class_names.py`: Holds label mapping.
Dividing files this way follows standard AI engineering practices (Separation of Concerns). It makes the codebase clean, reusable, easy to debug, and simple to test.

### Q2: Why did you choose a Convolutional Neural Network (CNN) instead of a simple Linear Multi-Layer Perceptron (MLP)?
**Answer**: 
MLPs flatten images into a single vector, which discards spatial relationship information (which pixels are next to each other). CNNs use kernels to scan pixels locally, preserving spatial hierarchy (textures, edges, patterns), and use far fewer parameters due to weight sharing.

### Q3: How did you compute the input size (`1600` features) for the first fully connected layer `fc1`?
**Answer**:
1. Input image shape is `(1, 28, 28)`.
2. First layer is `Conv2d(1, 32, kernel_size=3)` (no padding, stride 1). Output is `(32, 26, 26)` because $(28 - 3)/1 + 1 = 26$.
3. MaxPool1 reduces size by half: `(32, 13, 13)`.
4. Second layer is `Conv2d(32, 64, kernel_size=3)`. Output is `(64, 11, 11)` because $(13 - 3)/1 + 1 = 11$.
5. MaxPool2 reduces size using a 2x2 filter: $\lfloor (11 - 2)/2 \rfloor + 1 = 5$, giving `(64, 5, 5)`.
6. Flattening the `(64, 5, 5)` tensor yields a 1D vector of size $64 \times 5 \times 5 = 1600$ elements.

### Q4: What does `transforms.ToTensor()` do to the images?
**Answer**:
It does two main tasks:
1. Converts input images (PIL or NumPy arrays) of shape `(H, W, C)` to PyTorch tensors of shape `(C, H, W)`.
2. Scales pixel integer values from range `[0, 255]` to floating-point range `[0.0, 1.0]`.

### Q5: Why do we write `optimizer.zero_grad()` in the training loop?
**Answer**:
In PyTorch, gradients are accumulated (added together) by default on every `loss.backward()` call. If we don't reset them to zero at the start of each training step, the gradients from previous steps will pile up, leading to incorrect weight updates.

### Q6: What is the difference between `model.train()` and `model.eval()`?
**Answer**:
They set the model state. `model.train()` enables training features like Dropout or Batch Normalization. `model.eval()` disables these behaviors to ensure that model outputs are consistent, deterministic, and fast during testing or inference.

### Q7: Why do we use `torch.no_grad()` during evaluation?
**Answer**:
It tells PyTorch not to build the computational graph or calculate gradients. This reduces memory footprint (avoiding storing intermediate gradients) and accelerates execution speed during evaluation.

### Q8: What loss function did you use, and why?
**Answer**:
We used `nn.CrossEntropyLoss()`. It is the standard loss function for multi-class classification. It applies Softmax to the model's logits internally and calculates the negative log-likelihood loss, measuring the error between predictions and target distributions.

### Q9: What optimizer did you use, and what was the learning rate?
**Answer**:
We used the `Adam` optimizer with a learning rate of `0.001`. Adam is an adaptive learning rate optimizer that combines Momentum and RMSProp, making it robust and fast for training deep learning models.

### Q10: How did you calculate confidence scores in `predict.py`?
**Answer**:
The model outputs raw logits (scores) for the 10 classes. In `predict.py`, we apply the `torch.softmax` function over these logits, which converts them to a probability distribution (values between 0 and 1 that sum up to 1). We then extract the highest value as the confidence score.

### Q11: What does `torch.save(model.state_dict(), path)` do, and why save `state_dict` instead of the whole model?
**Answer**:
`state_dict()` is a Python dictionary that maps each layer's parameter name to its weights tensor. Saving the `state_dict` is the recommended best practice in PyTorch because it only saves the trained weights, keeping file sizes small and making it highly flexible to load weights onto modified model architectures.

### Q12: Why is batch size important, and why did you choose 64?
**Answer**:
Batch size dictates how many samples are processed before updating model parameters. Processing one image at a time (Stochastic Gradient Descent) is noisy and slow; processing the entire dataset at once (Batch Gradient Descent) is memory-intensive. A batch size of 64 is a standard power-of-two setting that balances training speed, memory consumption, and gradient updates.

### Q13: What is "Overfitting", and does your model suffer from it?
**Answer**:
Overfitting occurs when a model learns the training data *too* well (including its noise) and fails to generalize to unseen test data. If overfitting occurred, training loss would continue to drop while test accuracy decreased. In our run, test accuracy stabilized around **91.08%** with stable loss, showing healthy training behavior.

### Q14: How does the model run prediction on CPU if it was trained on GPU?
**Answer**:
We use the `map_location` argument when loading weights: `torch.load(model_path, map_location=device)`. If `device` is set to `cpu`, PyTorch automatically remaps the GPU tensors to CPU tensors, preventing runtime crashes.

### Q15: What improvements would you suggest to increase test accuracy further?
**Answer**:
1. **Data Augmentation**: Introduce random flips, rotations, or crops to make the training set diverse.
2. **Batch Normalization**: Add `nn.BatchNorm2d` layers to stabilize training and accelerate convergence.
3. **Dropout**: Add `nn.Dropout` to randomly turn off neurons, preventing overfitting.
4. **Learning Rate Scheduler**: Reduce the learning rate gradually as loss plateaus to fine-tune weights.
