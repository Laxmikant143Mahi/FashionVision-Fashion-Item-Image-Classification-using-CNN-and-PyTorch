# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

def get_data_loaders():
    """
    Why it exists:
        To load and prepare the FashionMNIST dataset for training and testing.
        DataLoaders load data in batches and shuffle training images.

    What it does:
        1. Automatically downloads the FashionMNIST training and testing datasets if not present.
        2. Converts the PIL images to PyTorch Tensors using transforms.ToTensor().
        3. Creates and returns two DataLoader objects: one for training and one for testing.

    Input:
        None

    Output:
        tuple: (train_loader, test_loader)
            - train_loader (DataLoader): Loader for training data (batch_size=64, shuffle=True)
            - test_loader (DataLoader): Loader for testing data (batch_size=64, shuffle=False)
    """
    # Define image transformations (convert image to PyTorch FloatTensor and scale values to [0.0, 1.0])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Download training dataset (automatically downloaded to 'dataset' folder)
    train_dataset = datasets.FashionMNIST(
        root="./dataset",
        train=True,
        download=True,
        transform=transform
    )

    # Download testing dataset (automatically downloaded to 'dataset' folder)
    test_dataset = datasets.FashionMNIST(
        root="./dataset",
        train=False,
        download=True,
        transform=transform
    )

    # Create training dataloader: shuffle data to ensure the model doesn't learn dependencies on data order
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )

    # Create testing dataloader: shuffle is False since we just want to evaluate the model in its order
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False
    )

    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Why it exists:
        To perform one full pass (epoch) of training over the entire training dataset.

    What it does:
        Puts the model in training mode, loops over the training batches,
        performs forward pass, calculates loss, updates gradients, and prints batch progress.

    Input:
        model (nn.Module): The SimpleCNN neural network model.
        train_loader (DataLoader): The loader containing training batches.
        criterion (nn.Module): The loss function (CrossEntropyLoss).
        optimizer (Optimizer): The optimization algorithm (Adam).
        device (torch.device): The device (CPU or GPU/CUDA) to run calculations on.

    Output:
        float: The average training loss for this epoch.
    """
    # Put model in training mode (activates Dropout, Batch Normalization if present)
    model.train()
    
    running_loss = 0.0
    
    # Loop over all batches of images and labels in the training loader
    for images, labels in train_loader:
        # Move images and labels to the active device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)
        
        # 1. Reset gradients to zero to prevent accumulation from previous batches
        optimizer.zero_grad()
        
        # 2. Forward pass: compute predicted outputs by passing images to the model
        outputs = model(images)
        
        # 3. Calculate loss: compare model outputs with actual labels
        loss = criterion(outputs, labels)
        
        # 4. Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # 5. Optimization step: update model parameters based on computed gradients
        optimizer.step()
        
        # Accumulate the loss for loss calculation at the end of epoch
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate_accuracy(model, test_loader, device):
    """
    Why it exists:
        To calculate how accurately the model classifies unseen testing images.

    What it does:
        Puts the model in evaluation mode, disables gradient computation,
        makes predictions on test images, and calculates accuracy percentage.

    Input:
        model (nn.Module): The SimpleCNN neural network model.
        test_loader (DataLoader): The loader containing testing batches.
        device (torch.device): The device (CPU or GPU/CUDA) to run calculations on.

    Output:
        float: The test accuracy percentage (e.g., 85.50).
    """
    # Put model in evaluation mode (deactivates Dropout, Batch Normalization if present)
    model.eval()
    
    correct_predictions = 0
    total_images = len(test_loader.dataset)
    
    # Disable gradient tracking since we are only doing inference (saves memory and speed)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass: get model outputs
            outputs = model(images)
            
            # Get the index of the highest logit value (predicted class)
            _, predictions = torch.max(outputs, 1)
            
            # Sum up correct predictions
            correct_predictions += (predictions == labels).sum().item()
            
    # Calculate percentage
    accuracy = (correct_predictions / total_images) * 100.0
    return accuracy


def main():
    """
    Why it exists:
        To orchestrate the entire training pipeline, run all epochs, and save
        the best performing model.

    What it does:
        Sets the device, creates saving folders, loads the model, optimizer,
        and loss function, then loops 10 times to train and evaluate, saving 
        the best state_dict.

    Input:
        None

    Output:
        None
    """
    # Use GPU (cuda) if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the 'saved_models' directory if it does not exist
    os.makedirs("saved_models", exist_ok=True)

    # Initialize the model and move it to the device
    model = SimpleCNN().to(device)

    # Define loss function: CrossEntropyLoss is ideal for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Define optimizer: Adam optimizer with learning rate 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load training and testing data
    train_loader, test_loader = get_data_loaders()

    epochs = 10
    best_accuracy = 0.0

    print("Starting training...")
    
    # Run training for 10 epochs
    for epoch in range(1, epochs + 1):
        # Train model for one epoch and get the average training loss
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Calculate accuracy on the test set
        accuracy = evaluate_accuracy(model, test_loader, device)
        
        # Print progress summary for this epoch
        print(f"Epoch [{epoch}/{epochs}] - Loss: {loss:.4f} - Test Accuracy: {accuracy:.2f}%")
        
        # If this epoch's accuracy is the highest so far, save this model state
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save the parameters of the model (state_dict) to path
            torch.save(model.state_dict(), "saved_models/best_model.pth")
            print(f"--> Saved new best model with accuracy: {best_accuracy:.2f}%")
            
    print("\nTraining completed successfully!")
    print(f"Best Accuracy Achieved: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
