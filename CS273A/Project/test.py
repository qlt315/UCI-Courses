# from sklearn.discriminant_analysis import StandardScaler
# import torch
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# from train import ImprovedNN
# import pandas as pd
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc

# # Set random seed for reproducibility
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def load_data(file_path):
#     # Read the dataset without headers
#     data = pd.read_csv(file_path, header=None)

#     # Extract features (x) and target (y)
#     x = data.iloc[1:, :-1]  # Exclude the first row (header) for features
#     y = data.iloc[1:, -1]  # Exclude the first row (header) for labels

#     # Reset index for both x and y
#     x.reset_index(drop=True, inplace=True)
#     y.reset_index(drop=True, inplace=True)

#     return x, y

# def test_model(test_file_path, model_path):
#     # Load test data
#     x_test, y_test = load_data(test_file_path)
    
#     # Standardize features
#     scaler = StandardScaler()
#     x_test = scaler.fit_transform(x_test)

#     y_test = y_test.astype(np.float32)
#     x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
#     y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

#     # Load the saved model
#     input_size = x_test.shape[1]
#     model = ImprovedNN(input_size)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()  # Set to evaluation mode

#     # Perform predictions
#     with torch.no_grad():
#         y_pred_proba = model(x_test_tensor)
#         y_pred = (y_pred_proba > 0.5).float()

#     # Convert tensors to numpy arrays for sklearn metrics
#     y_true = y_test_tensor.numpy().flatten()
#     y_pred = y_pred.numpy().flatten()

# # new roc curve
#     plt.figure()
#     fpr, tpr, _ = roc_curve(y_test, y_pred)
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, lw=2, label='%s (AUC = %0.2f)' % ('NN', roc_auc))
#     # Plot the chance line
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic Comparison')
#     plt.legend(loc="lower right")
#     plt.show()

# # new roc curve

#     # Compute metrics
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)

#     # Print results
#     print(f"Test Accuracy: {accuracy:.4f}")
#     print(f"Test Precision: {precision:.4f}")
#     print(f"Test Recall: {recall:.4f}")
#     print(f"Test F1-score: {f1:.4f}")

#     # Print detailed classification report
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred))

# # Entry point for the script
# if __name__ == "__main__":
#     set_seed(37)
#     # Path to the test dataset and saved model
#     test_file_path = "dataset/adult/augmented_test_data.csv"
#     model_path = "models/improved_nn_model.pth"

#     # Test the model
#     test_model(test_file_path, model_path)

from sklearn.discriminant_analysis import StandardScaler
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from train import ImprovedNN, SimpleNN  # 确保您的 train.py 中包含 SimpleNN 类
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

def test_models(test_file_path, improved_model_path, simple_model_path):
    # Load test data
    x_test, y_test = load_data(test_file_path)
    
    # Standardize features
    scaler = StandardScaler()
    x_test = scaler.fit_transform(x_test)

    y_test = y_test.astype(np.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Load the saved models
    input_size = x_test.shape[1]
    improved_model = ImprovedNN(input_size)
    improved_model.load_state_dict(torch.load(improved_model_path))
    improved_model.eval()

    simple_model = SimpleNN(input_size)
    simple_model.load_state_dict(torch.load(simple_model_path))
    simple_model.eval()

    # Perform predictions
    with torch.no_grad():
        y_pred_proba_improved = improved_model(x_test_tensor)
        y_pred_improved = (y_pred_proba_improved > 0.5).float()

        y_pred_proba_simple = simple_model(x_test_tensor)
        y_pred_simple = (y_pred_proba_simple > 0.5).float()

    # Convert tensors to numpy arrays for sklearn metrics
    y_true = y_test_tensor.numpy().flatten()
    y_pred_improved = y_pred_improved.numpy().flatten()
    y_pred_simple = y_pred_simple.numpy().flatten()

    # ROC curve
    plt.figure(figsize=(10, 8))
    
    # Improved NN
    fpr_improved, tpr_improved, _ = roc_curve(y_true, y_pred_proba_improved.numpy())
    roc_auc_improved = auc(fpr_improved, tpr_improved)
    plt.plot(fpr_improved, tpr_improved, lw=2, label='Improved NN (AUC = %0.2f)' % roc_auc_improved)
    
    # Simple NN
    fpr_simple, tpr_simple, _ = roc_curve(y_true, y_pred_proba_simple.numpy())
    roc_auc_simple = auc(fpr_simple, tpr_simple)
    plt.plot(fpr_simple, tpr_simple, lw=2, label='Simple NN (AUC = %0.2f)' % roc_auc_simple)

    # Plot the chance line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Comparison')
    plt.legend(loc="lower right")
    plt.show()

    # Compute and print metrics for both models
    for name, y_pred in [("Improved NN", y_pred_improved), ("Simple NN", y_pred_simple)]:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"\n{name} Results:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-score: {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

# Entry point for the script
if __name__ == "__main__":
    set_seed(37)
    # Path to the test dataset and saved models
    test_file_path = "dataset/adult/augmented_test_data.csv"
    improved_model_path = "models/improved_nn_model.pth"
    simple_model_path = "models/simple_nn_model.pth"  # 确保您有这个模型文件

    # Test the models
    test_models(test_file_path, improved_model_path, simple_model_path)