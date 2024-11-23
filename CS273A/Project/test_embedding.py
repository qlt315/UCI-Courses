import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from train_embedding import EmbeddingNN
import random

# Define categorical columns
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']

# Define embedding dimensions (same as used in training)
embedding_dims = [(10, 4), (16, 6), (5, 3), (8, 5), (5, 4), (5, 3)]  # Example embedding dimensions

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load dataset and separate features
def load_data(file_path, categorical_columns):
    # Load data
    data = pd.read_csv(file_path)

    # Extract features (x) and target (y)
    x = data.drop(columns=['income'])  # Exclude target column
    y = data['income']

    return x, y

def main():
    # Set the seed for reproducibility
    set_seed(37)

    # Load test data
    test_file_path = "dataset/adult/augmented_test_data.csv"
    x_test, y_test = load_data(test_file_path, categorical_columns)

    # Convert labels to float32 (for binary classification)
    y_test = y_test.replace({' <=50K': 0, ' >50K': 1}).astype(np.float32)

    # Convert data to PyTorch tensors
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader for batching
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the trained model
    model = EmbeddingNN(train_data=x_test,
                        categorical_columns=categorical_columns,
                        embedding_dims=embedding_dims)

    # Load the saved model state
    model.load_state_dict(torch.load("models/embedding_nn_model.pth"))
    model.eval()  # Set the model to evaluation mode

    # Evaluate the model on the test data
    y_pred = []
    y_true = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            y_pred.extend((outputs > 0.5).float().numpy().flatten())
            y_true.extend(y_batch.numpy().flatten())

    # Calculate accuracy and classification report
    test_accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report on Test Set:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()
