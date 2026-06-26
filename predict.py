# predict.py
import os
import sys
import random
import torch

# Check if we should run in headless mode (e.g. during automated/background testing)
if "--headless" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import SimpleCNN
from utils.class_names import get_class_names


def load_trained_model(model_path, device):
    """
    Why it exists:
        To load the trained SimpleCNN model and its learned weights from disk.

    What it does:
        Creates the SimpleCNN architecture, loads the state dictionary from the
        given file path, sets it to evaluation mode, and moves it to the device.

    Input:
        model_path (str): Path to the saved '.pth' model file.
        device (torch.device): The CPU or GPU device to run on.

    Output:
        nn.Module: Loaded model ready for inference.
    """
    model = SimpleCNN()
    
    # Load model weights, handling device mapping
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()  # Put model in evaluation mode
    return model


def predict_random_sample():
    """
    Why it exists:
        To run a test prediction on a single random clothing image and visualize 
        both text predictions and matplotlib results.

    What it does:
        1. Selects a random image from the FashionMNIST test set.
        2. Feeds it through the trained network.
        3. Computes class probabilities using Softmax.
        4. Prints actual, predicted, and confidence score in the required style.
        5. Visualizes the sample using matplotlib and saves it to 'screenshots/'.

    Input:
        None

    Output:
        None
    """
    model_path = "saved_models/best_model.pth"
    
    # Check if the trained model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train the model first by running 'train.py'.")
        return

    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model
    model = load_trained_model(model_path, device)

    # Prepare dataset (uses standard transform.ToTensor)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    test_dataset = datasets.FashionMNIST(
        root="./dataset",
        train=False,
        download=True,
        transform=transform
    )

    # Get class names from utils mapping
    class_mapping = get_class_names()

    # Pick a random image index from the test set
    random_idx = random.randint(0, len(test_dataset) - 1)
    
    # Get the image tensor and actual integer label
    image_tensor, actual_label_idx = test_dataset[random_idx]

    # Model expects batch dimension: reshape image from (1, 28, 28) to (1, 1, 28, 28)
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # Perform inference (forward pass) with gradients disabled
    with torch.no_grad():
        output = model(input_tensor)
        
        # Calculate class probabilities using softmax function
        probabilities = torch.softmax(output, dim=1)

    # Get index of highest probability and its value
    confidence_tensor, predicted_label_idx = torch.max(probabilities, 1)
    
    # Convert PyTorch values to standard Python types
    predicted_idx = predicted_label_idx.item()
    confidence = confidence_tensor.item() * 100.0  # Convert to percentage
    
    # Get label name strings
    actual_label_name = class_mapping[actual_label_idx]
    predicted_label_name = class_mapping[predicted_idx]

    # Print the exact requested output format
    print("=========================================")
    print("FashionVision Prediction")
    print("=========================================")
    print(f"Actual Label      : {actual_label_name}")
    print(f"Predicted Label   : {predicted_label_name}")
    print(f"Confidence        : {confidence:.2f}%")
    print("=========================================")

    # Display image using Matplotlib
    # Squeeze out the channel dimension: (1, 28, 28) -> (28, 28) for matplotlib plotting
    img = image_tensor.squeeze().numpy()
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")
    
    # Set the title according to specifications
    title_text = f"Actual : {actual_label_name}\nPredicted : {predicted_label_name}\nConfidence : {confidence:.2f}%"
    plt.title(title_text, fontsize=12, fontweight="bold", color="darkblue")
    plt.axis("off")  # Remove pixel axes labels for cleaner visual
    
    # Make sure screenshots folder exists
    os.makedirs("screenshots", exist_ok=True)
    
    # Save the screenshot plot inside the screenshots directory
    plt.savefig("screenshots/prediction_sample.png", bbox_inches="tight")
    print("Saved visualization to 'screenshots/prediction_sample.png'")
    
    # Display the window (if running in interactive screen mode)
    plt.show()


if __name__ == "__main__":
    predict_random_sample()
