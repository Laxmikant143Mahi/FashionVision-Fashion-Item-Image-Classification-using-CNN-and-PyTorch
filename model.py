# model.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Why it exists:
        To define the neural network architecture (SimpleCNN) used to classify 
        FashionMNIST images. It represents the brain of our AI project.

    What it does:
        This class sets up the layers of the Convolutional Neural Network (CNN) 
        and defines how an input image passes through these layers (forward pass).

    Input (for instantiating):
        None

    Output (when instantiated):
        An object of SimpleCNN which is a PyTorch neural network module.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 1. First Convolutional Layer
        # Input: 1 channel (grayscale image), Output: 32 channels, Kernel size: 3x3
        # Output height and width calculation: (28 - 3) / 1 + 1 = 26.
        # Output shape for a single image: (32, 26, 26)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        
        # Activation function: Introduces non-linearity to help the model learn complex patterns
        self.relu1 = nn.ReLU()
        
        # Max Pooling Layer: Reduces height and width by half using a 2x2 grid and stride of 2
        # Output shape for a single image: (32, 13, 13)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. Second Convolutional Layer
        # Input: 32 channels, Output: 64 channels, Kernel size: 3x3
        # Output height and width calculation: (13 - 3) / 1 + 1 = 11.
        # Output shape for a single image: (64, 11, 11)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        # Activation function
        self.relu2 = nn.ReLU()
        
        # Max Pooling Layer: Reduces height and width using a 2x2 grid and stride of 2
        # Formula: floor((11 - 2) / 2) + 1 = floor(4.5) + 1 = 5
        # Output shape for a single image: (64, 5, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3. Fully Connected (Linear) Layers
        # Before sending features to a Linear layer, we flatten the 3D features (64 channels of 5x5)
        # into a 1D vector. Total elements = 64 * 5 * 5 = 1600.
        self.fc1 = nn.Linear(in_features=1600, out_features=128)
        
        # Activation function
        self.relu3 = nn.ReLU()
        
        # Final output layer: Maps 128 features to 10 classes (one for each clothing type)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        """
        Why it exists:
            To define the forward pass of the model, which describes how the input
            data flows through the layers to generate prediction logits.

        What it does:
            Passes the input tensor x through convolutional, activation, pooling,
            flattening, and linear layers.

        Input:
            x (torch.Tensor): Input batch of images of shape (batch_size, 1, 28, 28).

        Output:
            torch.Tensor: Logits for the 10 classes of shape (batch_size, 10).
        """
        # Pass through first conv -> relu -> pool block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Pass through second conv -> relu -> pool block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten the features starting from the channel dimension (dim 1)
        # keeping the batch size dimension (dim 0) intact.
        # This converts shape (batch_size, 64, 5, 5) to (batch_size, 1600).
        x = torch.flatten(x, start_dim=1)
        
        # Pass through first fully connected layer and activation
        x = self.fc1(x)
        x = self.relu3(x)
        
        # Pass through final fully connected layer to get class logits
        x = self.fc2(x)
        
        return x
