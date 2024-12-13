import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_data(file_path):
    """Load and preprocess dataset"""
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status',
        'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    data = pd.read_csv(file_path, sep=",\s", header=None, names=column_names, engine='python')

    # For test set, remove trailing periods in income column
    if 'income' in data.columns:
        data['income'].replace(regex=True, inplace=True, to_replace=r'\.', value=r'')

    return data

def preprocess_data(data):
    """Preprocess the dataset by encoding categorical features and scaling"""
    # Separate features and target
    features = data.drop(columns=['income'])
    labels = data['income']

    # One-hot encoding for categorical features
    features_encoded = pd.get_dummies(features.select_dtypes(include=['category', 'object']))
    features_numeric = features.select_dtypes(exclude=['category', 'object'])
    features_final = pd.concat([features_numeric, features_encoded], axis=1)

    return features_final, labels

def train_model(train_data, train_labels, test_data, test_labels, model):
    """Train a model and evaluate on test data"""
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)

    # Evaluation metrics
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    confusion = confusion_matrix(test_labels, predictions)

    return accuracy, report, confusion

def main():
    set_seed(37)

    # Load and preprocess data
    train_data = load_data('dataset/adult/augmented_train_data.csv')
    test_data = load_data('dataset/adult/augmented_test_data.csv')
    data = pd.concat([train_data, test_data]).reset_index(drop=True)
    features, labels = preprocess_data(data)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train and evaluate Decision Tree (Gini)
    dt_gini = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=0.05, min_samples_leaf=0.001)
    dt_gini_accuracy, dt_gini_report, dt_gini_confusion = train_model(dt_gini, x_train, y_train, x_test, y_test)
    print(f"Decision Tree (Gini) Accuracy: {dt_gini_accuracy:.2f}")
    print(dt_gini_report)

    # Train and evaluate Decision Tree (Entropy)
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=0.05, min_samples_leaf=0.001)
    dt_entropy_accuracy, dt_entropy_report, dt_entropy_confusion = train_model(dt_entropy, x_train, y_train, x_test, y_test)
    print(f"Decision Tree (Entropy) Accuracy: {dt_entropy_accuracy:.2f}")
    print(dt_entropy_report)

    # Train and evaluate Random Forest (Gini)
    rf_gini = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=0.05, min_samples_leaf=0.001)
    rf_gini_accuracy, rf_gini_report, rf_gini_confusion = train_model(rf_gini, x_train, y_train, x_test, y_test)
    print(f"Random Forest (Gini) Accuracy: {rf_gini_accuracy:.2f}")
    print(rf_gini_report)

    # Train and evaluate Random Forest (Entropy)
    rf_entropy = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_split=0.05, min_samples_leaf=0.001)
    rf_entropy_accuracy, rf_entropy_report, rf_entropy_confusion = train_model(rf_entropy, x_train, y_train, x_test, y_test)
    print(f"Random Forest (Entropy) Accuracy: {rf_entropy_accuracy:.2f}")
    print(rf_entropy_report)

    # Train and evaluate AdaBoost
    ada = AdaBoostClassifier(n_estimators=100)
    ada_accuracy, ada_report, ada_confusion = train_model(ada, x_train, y_train, x_test, y_test)
    print(f"AdaBoost Accuracy: {ada_accuracy:.2f}")
    print(ada_report)

if __name__ == "__main__":
    main()
