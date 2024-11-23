import dataset_utils as util
import random
import numpy as np
import torch

categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def process_train_data(file_path):

    # 1. Load data
    x, y = util.load_data(file_path)

    # Clean and embed the data
    x, y = util.clean_data(x, y)

    # 3. Feature selection (select top K features based on feature importance)
    x, selected_features = util.feature_selection(x, y, k=12)

    # 3. Data standardization (mean=0, std=1) for numerical features
    x, std_scaler = util.standardize_train_data(x, categorical_columns)

    # 4. Data normalization (scaling features between 0 and 1)
    x, norm_scaler = util.normalize_train_data(x, categorical_columns)

    # 5. Data augmentation (increase training data size via augmentation techniques)
    # x, y = util.data_augmentation(x, y)

    # 6. Split the data into training and evaluation sets (80/20 split)
    x_train, x_eval, y_train, y_eval = util.split_train_data(x, y)

    # Output the shape of the resulting datasets
    print("\nTraining Data Shape:", x_train.shape)
    print("Evaluation Data Shape:", x_eval.shape)

    return x_train, x_eval, y_train, y_eval, std_scaler, norm_scaler, selected_features


def process_test_data(file_path, std_scaler, norm_scaler, selected_features):
    """Process test data with the same transformations as train data."""
    # Load data
    x, y = util.load_data(file_path)
    y = y.str.replace('.', '', regex=False)
    print("\nTesting Data Shape:", x.shape)
    # Clean data
    x, y = util.clean_data(x, y)
    # Export the test data

    # Drop features not selected in the training data
    x = x[selected_features]

    # Standardize/Normalize the data using the scaler fitted on training data
    x = util.standardize_test_data(x, std_scaler,categorical_columns)
    x = util.normalize_test_data(x, norm_scaler,categorical_columns)

    return x, y


if __name__ == "__main__":
    set_seed(37)
    train_file_path = "dataset/adult/train_data.csv"
    x_train, x_eval, y_train, y_eval, std_scaler, norm_scaler, selected_features = process_train_data(train_file_path)
    # Export the train and eval data
    util.export_data(x_train, y_train, "dataset/adult/augmented_train_data.csv")
    util.export_data(x_eval, y_eval, "dataset/adult/augmented_eval_data.csv")

    test_file_path = "dataset/adult/test_data.csv"
    x_test, y_test = process_test_data(test_file_path, std_scaler, norm_scaler, selected_features)
    y_test = y_test.astype(int)  # Ensure integer labels
    # Export the test data
    util.export_data(x_test, y_test, "dataset/adult/augmented_test_data.csv")
