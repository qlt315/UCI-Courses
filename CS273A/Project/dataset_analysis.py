import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load and preprocess dataset
def load_data(file_path):
    # Define column names for the dataset
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    # Read the dataset
    data = pd.read_csv(file_path, header=None, names=column_names, na_values=' ?')

    # Fill missing values
    for column in data.columns:
        if data[column].dtype == 'object':  # Categorical column
            data[column].fillna(data[column].mode()[0], inplace=True)  # Fill with mode
        else:  # Numerical column
            data[column].fillna(data[column].mean(), inplace=True)  # Fill with mean

    return data

# Plot bar chart for a specific feature against income
def plot_feature_income_relationship(data, feature_index):
    # Assuming the last column is the 'income' column
    income_column = data.columns[-1]
    feature_column = data.columns[feature_index]
    figure_size = (10, 4)  # Set a thin and long figure size

    fig, ax = plt.subplots(figsize=figure_size)
    # Group by the feature and income, and count occurrences
    group_data = data.groupby([feature_column, income_column]).size().unstack()
    group_data.plot(kind='bar', stacked=True, ax=ax)  # Use the existing ax

    ax.set_title(f'Income Distribution by {feature_column}')
    ax.set_xlabel(feature_column)
    ax.set_ylabel('Count')

    # Modify x-tick labels to split long labels into multiple lines
    def split_label(label,max_length):
        label = str(label)
        if len(label) > max_length:
            return '\n'.join([label[i:i+max_length] for i in range(0, len(label), max_length)])
        return label

    max_length = 6
    # Determine x-tick label display based on the number of categories
    num_categories = len(group_data.index)
    if num_categories > 15:
        # Show labels at fixed intervals
        interval = max(1, num_categories // 15)
        ax.set_xticks(np.arange(num_categories)[::interval])
        ax.set_xticklabels([split_label(label,max_length) for label in group_data.index[::interval]], rotation=0)
    else:
        ax.set_xticklabels([split_label(label,max_length) for label in group_data.index], rotation=0)

    ax.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.show()

def main():
    set_seed(37)

    # Load the dataset
    file_path = "dataset/adult/train_data.csv"  # Path to the original dataset
    data = load_data(file_path)

    # Feature index to plot
    feature_index = 13  # Do not set to 2, 10, 11, 13

    # Plot the feature-income relationship for the specified feature
    plot_feature_income_relationship(data, feature_index)

if __name__ == "__main__":
    main()