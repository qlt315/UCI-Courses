import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Embedding
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def load_data(file_path):
    # Read the dataset without headers
    data = pd.read_csv(file_path, header=None,
                       names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                              'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                              'hours-per-week', 'native-country', 'income'])

    # Extract features (x) and target (y)
    x = data.iloc[:, :-1]  # All columns except the last one as features
    y = data.iloc[:, -1]  # Last column as target
    return x, y


def check_data_info(x, y):
    # Check basic information of feature data
    print("Feature Data Info:")
    print(x.info())
    print("\nTarget Data Info:")
    print(y.info())

    # Describe the dataset to get basic statistics
    print("\nFeature Data Description:")
    print(x.describe(include='all'))


def clean_data(x, y):
    def embedding_for_categorical(x):
        # Identify all categorical columns (string types)
        categorical_columns = x.select_dtypes(include=['object']).columns

        # Encode each categorical feature with LabelEncoder
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            x[col] = le.fit_transform(x[col])  # Convert categorical values to integers
            label_encoders[col] = le  # Save encoder for later use

        return x, label_encoders

    def apply_embedding(x, embedding_dim=8):
        # Identify encoded categorical columns (now integers)
        categorical_columns = x.select_dtypes(include=['int64']).columns  # Encoded categorical features

        # Create Embedding layer for each categorical feature
        embedding_layers = {}
        for col in categorical_columns:
            n_unique_values = len(x[col].unique())  # Number of unique categories
            embedding_layer = Embedding(input_dim=n_unique_values, output_dim=embedding_dim, input_length=1)
            embedding_layers[col] = embedding_layer

        # Return the dataframe (no actual embedding performed here, done in the model)
        return x, embedding_layers

    # Identify missing values represented as '?'
    missing_symbol = ' ?'
    missing_counts = (x == missing_symbol).sum()
    # print("\n Missing Values Represented by '?':")
    # print(missing_counts)

    # Replace '?' with np.nan for standard missing value handling
    x = x.replace(missing_symbol, np.nan)

    # Filling missing values
    # 1. For numerical columns: Fill with the mean
    num_cols = x.select_dtypes(include=['float64', 'int64']).columns
    x[num_cols] = x[num_cols].fillna(x[num_cols].mean())

    # 2. For categorical columns: Fill with the mode
    cat_cols = x.select_dtypes(include=['object']).columns
    x[cat_cols] = x[cat_cols].apply(lambda col: col.fillna(col.mode()[0]))

    # Check for missing values after filling
    # print("\nMissing values after filling:")
    # print(x.isna().sum())

    label_mapping = {' >50K': 1, ' <=50K': 0}
    y = y.replace(label_mapping)

    # 1. Embed all categorical features
    x, label_encoders = embedding_for_categorical(x)

    # 2. Apply embedding (this step will typically be done in a deep learning model, so here we just prepare the data)
    x, embedding_layers = apply_embedding(x)
    return x, y


def feature_selection(x, y, k=10):
    """Select top K features based on RFE."""
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=k)

    # Fit the model and transform the data
    x_selected = rfe.fit_transform(x, y)

    # Get the names of selected features
    selected_features = x.columns[rfe.support_]

    print(f"Selected Features: {selected_features}")

    # Convert the selected features into a DataFrame and return
    x_selected_df = pd.DataFrame(x_selected, columns=selected_features)

    return x_selected_df, selected_features


# def standardize_train_data(x):
#     std_scaler = StandardScaler()
#     x_standardized = std_scaler.fit_transform(x)
#     return pd.DataFrame(x_standardized, columns=x.columns), std_scaler
#
# def normalize_train_data(x):
#     norm_scaler = MinMaxScaler()
#     x_normalized = norm_scaler.fit_transform(x)
#     return pd.DataFrame(x_normalized, columns=x.columns), norm_scaler

def standardize_train_data(x, categorical_columns):
    non_categorical_columns = [col for col in x.columns if col not in categorical_columns]
    std_scaler = StandardScaler()

    # Standardize only non-categorical columns
    x_standardized = x.copy()
    x_standardized[non_categorical_columns] = std_scaler.fit_transform(x[non_categorical_columns])

    return x_standardized, std_scaler


def normalize_train_data(x, categorical_columns):
    non_categorical_columns = [col for col in x.columns if col not in categorical_columns]
    norm_scaler = MinMaxScaler()

    # Normalize only non-categorical columns
    x_normalized = x.copy()
    x_normalized[non_categorical_columns] = norm_scaler.fit_transform(x[non_categorical_columns])

    return x_normalized, norm_scaler


#
# def standardize_test_data(x, scaler):
#     std_scaler = scaler
#     x_standardized = std_scaler.fit_transform(x)
#     return pd.DataFrame(x_standardized, columns=x.columns)
#
#
# def normalize_test_data(x, scaler):
#     norm_scaler = scaler
#     x_normalized = norm_scaler.fit_transform(x)
#     return pd.DataFrame(x_normalized, columns=x.columns)

def standardize_test_data(x, scaler, categorical_columns):
    non_categorical_columns = [col for col in x.columns if col not in categorical_columns]

    # Standardize only non-categorical columns
    x_standardized = x.copy()
    x_standardized[non_categorical_columns] = scaler.transform(x[non_categorical_columns])

    return x_standardized

def data_augmentation(x, y):
    noise_factor = 0.1
    x_augmented = x + noise_factor * np.random.randn(*x.shape)
    y_augmented = y.copy()
    return x_augmented, y_augmented


def export_data(x, y, output_file):
    # Combine features and target
    data = pd.concat([x, y], axis=1)

    # Export to CSV
    data.to_csv(output_file, index=False)
    print(f"Data exported successfully to {output_file}")


def split_train_data(x_train, y_train, test_size=0.2, seed=42):
    # Split the data into train and evaluation sets (80% for training, 20% for evaluation)
    x_train_split, x_eval, y_train_split, y_eval = train_test_split(x_train, y_train, test_size=test_size,
                                                                    random_state=seed)

    return x_train_split, x_eval, y_train_split, y_eval
