import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import random

# Define categorical columns and original column names
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race']

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define a neural network model with embedding functionality
class EmbeddingNN(nn.Module):
    def __init__(self, train_data, categorical_columns, embedding_dims):
        super(EmbeddingNN, self).__init__()

        # Dynamically identify numerical and categorical columns based on the train data
        self.numerical_columns = [col for col in train_data.columns if col not in categorical_columns]
        self.final_categorical_columns = [col for col in categorical_columns if col in train_data.columns]
        # Calculate the input size for numerical features
        input_size = len(self.numerical_columns)

        # Create embedding layers for each selected categorical column
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_categories, embedding_dim)
            for col, (num_categories, embedding_dim) in zip(self.final_categorical_columns, embedding_dims)
        })

        # Calculate total embedding dimension
        total_embedding_dim = sum(embedding_dim for _, embedding_dim in embedding_dims)

        # Fully connected layers
        self.fc1 = nn.Linear(input_size + total_embedding_dim, 128)  # Input size includes numerical + embeddings
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Separate numerical and categorical features from the input x
        x_num = x[:, :len(self.numerical_columns)]  # Numerical features (first part of x)
        x_cat = {col: x[:, i + len(self.numerical_columns)].long()  # Convert to LongTensor
                 for i, col in enumerate(self.final_categorical_columns)}  # Categorical features (rest of x)

        # Apply embeddings to categorical features
        embedded = [self.embeddings[col](x_cat[col]) for col in x_cat]
        embedded = torch.cat(embedded, dim=1)  # Concatenate along feature dimension

        # Concatenate numerical and embedded features
        x_combined = torch.cat([x_num, embedded], dim=1)
        x_combined = self.fc1(x_combined)
        x_combined = self.relu(x_combined)
        x_combined = self.fc2(x_combined)
        x_combined = self.relu(x_combined)
        x_combined = self.fc3(x_combined)
        x_combined = self.sigmoid(x_combined)
        return x_combined

# Load dataset and split into numerical and categorical features
def load_data(file_path, categorical_columns):
    # Load data
    data = pd.read_csv(file_path)

    # Extract features (x) and target (y)
    x = data.drop(columns=['income'])  # Exclude target column
    y = data['income']

    return x, y

# Main function
def main():
    # Define file paths
    train_file_path = "dataset/adult/augmented_train_data.csv"
    eval_file_path = "dataset/adult/augmented_eval_data.csv"

    # Load data
    x_train, y_train = load_data(train_file_path, categorical_columns)
    x_eval, y_eval = load_data(eval_file_path, categorical_columns)

    # Convert labels to float32
    y_train = y_train.replace({' <=50K': 0, ' >50K': 1}).astype(np.float32)
    y_eval = y_eval.replace({' <=50K': 0, ' >50K': 1}).astype(np.float32)

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    x_eval_tensor = torch.tensor(x_eval.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # Add a dimension for binary classification
    y_eval_tensor = torch.tensor(y_eval.values, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader for batching
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_eval_tensor, y_eval_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define embedding dimensions
    embedding_dims = [(10, 4), (16, 6), (5, 3), (8, 5), (5, 4), (5, 3)]  # Example embedding dimensions

    # Initialize the model with both x_num and x_cat
    model = EmbeddingNN(train_data=x_train,
                        categorical_columns=categorical_columns,
                        embedding_dims=embedding_dims)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            # Forward pass: pass the whole batch of data (both numerical and categorical features)
            outputs = model(x_batch)  # x_batch contains both numerical and categorical features
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

                y_pred.extend((outputs > 0.5).float().numpy().flatten())
                y_true.extend(y_batch.numpy().flatten())

        val_accuracy = accuracy_score(y_true, y_pred)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "models/embedding_nn_model.pth")
    print("Model saved to 'models/embedding_nn_model.pth'")

    # Print final classification report
    print("\nFinal Classification Report on Validation Set:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    set_seed(37)
    main()