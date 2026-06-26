# test.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

def get_test_loader():
    """
    Why it exists:
        To load the testing dataset for the evaluation run.

    What it does:
        Loads the test set of FashionMNIST, converts images to tensors,
        and returns a DataLoader (batch_size=64, shuffle=False).

    Input:
        None

    Output:
        DataLoader: Loader for testing images.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    test_dataset = datasets.FashionMNIST(
        root="./dataset",
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False
    )
    
    return test_loader


def test_model():
    """
    Why it exists:
        To load the trained best model and evaluate its overall classification 
        accuracy on the testing dataset.

    What it does:
        Checks if model file exists, initializes model, loads state_dict, 
        evaluates classification accuracy, and prints in the designated terminal style.

    Input:
        None

    Output:
        None
    """
    model_path = "saved_models/best_model.pth"
    
    # Check if the trained model file exists before trying to load it
    if not os.path.exists(model_path):
        print(f"Error: Trained model file '{model_path}' not found.")
        print("Please run 'train.py' first to train and save the model.")
        return

    # Check device availability (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model structures
    model = SimpleCNN()
    
    # Load model weights (map_location handles loading GPU-trained model on CPU if needed)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move model to selected device and set it to evaluation mode
    model.to(device)
    model.eval()

    test_loader = get_test_loader()
    
    correct_predictions = 0
    total_images = len(test_loader.dataset)

    # Disable gradient tracking for testing/inference
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass: get outputs
            outputs = model(images)
            
            # Get predictions
            _, predictions = torch.max(outputs, 1)
            
            # Sum up correct predictions
            correct_predictions += (predictions == labels).sum().item()

    # Calculate final accuracy percentage
    accuracy = (correct_predictions / total_images) * 100.0

    # Print the output in the required format
    print("=====================================")
    print("Testing Completed")
    print("=====================================")
    print(f"Accuracy : {accuracy:.2f} %")
    print("=====================================")


if __name__ == "__main__":
    test_model()
