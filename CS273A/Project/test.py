import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from train import SimpleNN

# Define the test function
def test_model(test_file_path, model_path):

    # Load the test dataset
    test_data = pd.read_csv(test_file_path, header=None)

    # Separate features (x) and labels (y)
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # Ensure all feature columns are numeric
    x_test = x_test.apply(pd.to_numeric, errors='coerce')
    x_test = x_test.fillna(0)
    y_test = y_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    # Convert to PyTorch tensors
    x_test_tensor = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)  # Ensure labels are integers

    # Create DataLoader for batching
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the saved model
    input_size = x_test.shape[1]
    model = SimpleNN(input_size)

    # Load the saved model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    # Initialize variables for evaluation
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    # Perform testing
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    # Calculate accuracy
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))


# Entry point for the script
if __name__ == "__main__":
    # Path to the test dataset and saved model
    test_file_path = "dataset/adult/augmented_test_data.csv"
    model_path = "models/simple_nn_model.pth"

    # Test the model
    test_model(test_file_path, model_path)
