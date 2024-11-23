import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random
# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Fully connected layer 1
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(128, 64)  # Fully connected layer 2
        self.fc3 = nn.Linear(64, 1)  # Output layer (binary classification)
        self.sigmoid = nn.Sigmoid()  # Sigmoid for probability

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Load dataset
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

# Main function
def main():
    # Load the dataset
    train_file_path = "dataset/adult/augmented_train_data.csv"
    eval_file_path = "dataset/adult/augmented_eval_data.csv"
    x_train, y_train = load_data(train_file_path)
    x_eval, y_eval = load_data(eval_file_path)

    # Convert labels to float32
    y_train = y_train.astype(np.float32)
    x_train = x_train.astype(np.float32)
    x_eval = x_eval.astype(np.float32)
    y_eval = y_eval.astype(np.float32)
    x_eval_array = x_eval.to_numpy()
    x_train_array = x_train.to_numpy()

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train_array, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(
        1)  # Add a dimension for binary classification
    x_val_tensor = torch.tensor(x_eval_array, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_eval, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader for batching
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    input_size = x_train.shape[1]
    model = SimpleNN(input_size)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        y_pred = []
        y_true = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                # Collect predictions for evaluation
                y_pred.extend((outputs > 0.5).float().numpy().flatten())
                y_true.extend(y_batch.numpy().flatten())

        # Calculate accuracy on the validation set
        val_accuracy = accuracy_score(y_true, y_pred)

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(
            f"Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "models/simple_nn_model.pth")
    print("Model saved to 'models/simple_nn_model.pth'")

    # Print final classification report
    print("\nFinal Classification Report on Validation Set:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    set_seed(37)
    main()
