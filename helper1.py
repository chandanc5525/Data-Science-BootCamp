# Prepared By: Chandan Chaudhari
# All Copy Rights Reserved With Author

import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import matplotlib.pyplot as plt
import seaborn as sns

def download_csv(url, save_path):
    """
    Downloads a CSV file from the given URL and saves it to the specified path.

    Parameters:
    - url (str): The URL of the CSV file.
    - save_path (str): The local path (including filename) where the file should be saved.

    Returns:
    - str: The file path if the download was successful.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"File successfully downloaded and saved to {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    except IOError as io_error:
        print(f"File write error: {io_error}")
        return None

def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using Seaborn heatmap.

    Parameters:
    - cm (array): Confusion matrix.
    - class_names (list): List of class names.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def train_regression_model(X, y, model_type='linear', test_size=0.2, random_state=42):
    """
    Trains a regression model and evaluates its performance.

    Parameters:
    - X: Features (DataFrame or array).
    - y: Target (array).
    - model_type (str): Type of regression model ('linear', 'random_forest', 'svm').
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - model: Trained regression model.
    - float: Mean Squared Error on the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=random_state)
    elif model_type == 'svm':
        model = SVR()
    else:
        raise ValueError("Unsupported model_type. Choose 'linear', 'random_forest', or 'svm'.")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error ({model_type}): {mse}")
    return model, mse

def train_classification_model(X, y, model_type='logistic', test_size=0.2, random_state=42, class_names=None):
    """
    Trains a classification model and evaluates its performance.

    Parameters:
    - X: Features (DataFrame or array).
    - y: Target (array).
    - model_type (str): Type of classification model ('logistic', 'random_forest', 'svm').
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.
    - class_names (list): List of class names for confusion matrix.

    Returns:
    - model: Trained classification model.
    - dict: Evaluation metrics including accuracy, confusion matrix, and classification report.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_type == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state)
    elif model_type == 'svm':
        model = SVC()
    else:
        raise ValueError("Unsupported model_type. Choose 'logistic', 'random_forest', or 'svm'.")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=class_names)

    print(f"Accuracy ({model_type}): {accuracy}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    if class_names:
        plot_confusion_matrix(cm, class_names)

    return model, {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }

# Example usage for regression
# model, mse = train_regression_model(X, y, model_type='random_forest')

# Example usage for classification
# model, metrics = train_classification_model(X, y, model_type='svm', class_names=['Class 0', 'Class 1'])
