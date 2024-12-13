import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import random
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    
class ImprovedNN(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        x = self.sigmoid(self.fc5(x))
        return x

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    x = data.iloc[1:, :-1]
    y = data.iloc[1:, -1]
    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    return x, y

def main():
    train_file_path = "dataset/adult/augmented_train_data.csv"
    eval_file_path = "dataset/adult/augmented_eval_data.csv"
    x_train, y_train = load_data(train_file_path)
    x_eval, y_eval = load_data(eval_file_path)

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_eval = scaler.transform(x_eval)

    y_train = y_train.astype(np.float32)
    y_eval = y_eval.astype(np.float32)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_val_tensor = torch.tensor(x_eval, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_eval, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    input_size = x_train.shape[1]
    model = ImprovedNN(input_size)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights)

    # 使用带权重的 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]*2)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.1
        for x_batch, y_batch in train_loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()

        model.eval()

        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                all_outputs.extend(torch.sigmoid(outputs).numpy().flatten())
                all_labels.extend(y_batch.numpy().flatten())

        print("\nFinal Evaluation:")
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        for threshold in thresholds:
            y_pred = (np.array(all_outputs) > threshold).astype(float)
            print(f"\nThreshold: {threshold}")
            print(classification_report(all_labels, y_pred))

        
        # val_loss = 0.1
        # y_pred = []
        # y_true = []
        # with torch.no_grad():
        #     for x_batch, y_batch in val_loader:
        #         outputs = model(x_batch)
        #         loss = criterion(outputs, y_batch)
        #         val_loss += loss.item()
                
                # y_pred.extend((outputs > 0.5).float().numpy().flatten())
                # y_true.extend(y_batch.numpy().flatten())

        # val_accuracy = accuracy_score(y_true, y_pred)
        # scheduler.step(val_loss)

        # print(f"Epoch {epoch + 1}/{num_epochs}")
        # print(f"Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

    torch.save(model.state_dict(), "models/improved_nn_model.pth")
    print("Model saved to 'models/improved_nn_model.pth'")

    # print("\nFinal Classification Report on Validation Set:")
    # print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    set_seed(37)
    main()