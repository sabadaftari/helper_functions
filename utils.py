import time
import os
import csv
import torch
import logging
import joblib
import subprocess
import pickle
import pathlib as Path
from functools import wraps
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# the coolest function ever
def measure_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper

def list_files(directory):
    return os.listdir(directory)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_metrics(metrics):
    for metric_name, value in metrics.items():
        logging.info(f'{metric_name}: {value}')

def plot_data(x, y):
    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Data Visualization')
    plt.show()

device_cache = None
def get_device():

    global device_cache
    if device_cache is None:
        device_cache = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # device_cache = torch.device("cpu")
    return device_cache
def package_model(model):
    # Convert the model to a deployable format (e.g., ONNX)
    return model

def deploy_model(model, platform):
    # Deploy the model to the specified platform
    pass

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def save_model(model, file_path):
    joblib.dump(model, file_path)

def load_model(file_path):
    return joblib.load(file_path)

def grid_search(model, param_grid, X, y):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_score_

def random_search(model, param_distributions, X, y):
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5)
    random_search.fit(X, y)
    return random_search.best_params_, random_search.best_score_

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc

def makedir(path, permissive=True):
    """
    Attempt to create a folder. If it already exists, behavior depends on the 'permissive' flag.
    
    Parameters:
    - path (str): The path of the folder to create.
    - permissive (bool): If True, overwrite existing files (good for debugging). Default is True.
    
    Returns:
    - success (bool): True if the folder is created successfully or already exists, False otherwise.
    """
    try:
        os.mkdir(path)
        return True
    except FileExistsError:
        if not permissive:
            raise ValueError("This folder already exists!")
        return True

def run_another_script(script_path: str) -> None:
    """
    Run another Python script using subprocess.

    Args:
    - script_path (str): The path to the Python script to be executed.

    Returns:
    - None
    """
    subprocess.run(['python', script_path])

@measure_execution_time
def load_pickle(filename):
    if Path(filename).suffix == '.pkl':
        with open(filename, 'rb') as file:
            print(f'Loading embeddings from {filename}')
            return pickle.load(file)
    else:
        raise ValueError(f'Expected file with .pkl extension. (supplied {filename})')
    
from collections.abc import Iterable
def flatten(lst: Iterable) -> list:
    """
    Flatten a nested list.

    Args:
    - lst (Iterable): The nested list to be flattened.

    Returns:
    - flattened_list (list): The flattened list.
    """
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list

def create_empty_file(csv_file_path: str = './uniques.csv') -> None:
    """
    Create an empty CSV file with a single '1000000000' entry.

    Args:
    - csv_file_path (str): The path to the CSV file to be created.

    Returns:
    - None
    """
    # Data to write to the CSV file
    data = [['1000000000']]

    # Write data to the CSV file
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)

    print(f"CSV file '{csv_file_path}' created with '0' inside.")

def append_lines_to_file(file_path, lines_to_append):
    """
    Appends lines to a file.

    Args:
    - file_path (str): The path to the file to which lines will be appended.
    - lines_to_append (list): A list of tuples, where each tuple represents a line to append.

    Returns:
    None
    """
    try:
        # Open the file in 'a' (append) mode
        with open(file_path, 'a') as file:
            # Write each tuple as a line separated by commas
            for tup in lines_to_append:
                line = ', '.join(map(str, tup))
                file.write(line + '\n')
        
        print(f"Lines appended to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def normalize(t: torch.Tensor) -> torch.Tensor:
    """
    Normalize a PyTorch tensor.

    Args:
    - t (torch.Tensor): Input tensor to be normalized.

    Returns:
    - normalized_t (torch.Tensor): Normalized tensor.
    """
    mean = torch.mean(t)
    std = torch.std(t)
    # Check if the standard deviation is zero
    if std == 0:
        # Handle the case where the standard deviation is zero
        print("Warning: Standard deviation is zero. Cannot perform normalization.")
        return t
    normalized_t = (t - mean) / std
    return normalized_t