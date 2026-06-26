# utils/class_names.py

def get_class_names():
    """
    Why it exists:
        To map the numeric labels (0 to 9) from the FashionMNIST dataset 
        to human-readable clothing item names.

    What it does:
        Creates and returns a dictionary where keys are class indices (integers)
        and values are the names of the clothing items (strings).

    Input:
        None

    Output:
        dict: A dictionary mapping integer keys (0-9) to clothing name strings.
    """
    class_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"
    }
    return class_mapping
