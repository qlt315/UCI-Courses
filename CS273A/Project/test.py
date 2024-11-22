import torch
from sklearn.metrics import accuracy_score, classification_report
from train import SimpleNN
import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(file_path):
    # Read the dataset without headers
    data = pd.read_csv(file_path, header=None)

    # Extract features (x) and target (y)
    x = data.iloc[1:, :-1]  # Exclude the first row (header) for features
    y = data.iloc[1:, -1]  # Exclude the first row (header) for labels

    # Reset index for both x and y
    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    return x, y


def test_model(test_file_path, model_path):
    # Load test data
    x_test, y_test = load_data(test_file_path)
    y_test = y_test.astype(int)
    x_test = x_test.astype(np.float32)
    x_test_array = x_test.to_numpy()
    # Convert data to tensors
    x_test_tensor = torch.tensor(x_test_array, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(
        1)  # Add a dimension for binary classification

    # Load the saved model
    input_size = x_test.shape[1]
    model = SimpleNN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    # Perform predictions
    with torch.no_grad():
        y_prob = model(x_test_tensor).squeeze()  # Remove extra dimension
        y_pred = (y_prob > 0.5).int()  # Apply threshold to get binary predictions

    # Compute and print accuracy
    accuracy = accuracy_score(y_test_tensor.numpy(), y_pred.numpy())
    print(f"Test Accuracy: {accuracy:.4f}")


# Entry point for the script
if __name__ == "__main__":
    set_seed(37)
    # Path to the test dataset and saved model
    test_file_path = "dataset/adult/augmented_test_data.csv"
    model_path = "models/simple_nn_model.pth"

    # Test the model
    test_model(test_file_path, model_path)
