import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
from utils import load_config
import joblib

# Function to map categorical 'Stay' or 'Age' ranges to numerical values
def range_mapping(categorical_value):
    # Maps categorical 'Stay' or 'Age' ranges to numerical values.
    # Args: categorical_value (str): Categorical range, e.g., '0-10', '11-20'.
    # Returns: int: Corresponding numerical value.
    
    mapping = {
        '0-10': 5, '11-20': 15, '21-30': 25, '31-40': 35,
        '41-50': 45, '51-60': 55, '61-70': 65, '71-80': 75,
        '81-90': 85, '91-100': 95, 'More than 100 Days': 110
    }
    return mapping.get(categorical_value, -1)  # Default to -1 if the category is not found

# Function to reverse the mapping from numerical values to categorical ranges
def reverse_range_mapping(numerical_value):
    # Maps numerical values back to categorical ranges.
    # Args: numerical_value (float): Numerical value, e.g., 5, 15, 25.
    # Returns: str: Corresponding categorical range.
    
    ranges = {
        (0, 10): '0-10', (11, 20): '11-20', (21, 30): '21-30', (31, 40): '31-40',
        (41, 50): '41-50', (51, 60): '51-60', (61, 70): '61-70', (71, 80): '71-80',
        (81, 90): '81-90', (91, 100): '91-100', (101, float('inf')): 'More than 100 Days'
    }
    
    for (lower, upper), label in ranges.items():
        if lower <= numerical_value <= upper:
            return label
    return "Unknown"  # Fallback in case of unexpected values

# Function to apply range_mapping to a pandas series (Stay or Age)
def apply_range_mapping(series):
    # Applies range_mapping to a pandas Series to convert categorical values to numerical.
    # Args: series (pd.Series): Pandas series containing categorical values (e.g., 'Stay' or 'Age').
    # Returns: pd.Series: Series with numerical values.
    
    return series.apply(range_mapping)

# Function to encode categorical features using LabelEncoder
def encode_categorical_features(df, categorical_cols):
    # Encodes categorical columns using LabelEncoder.
    # Args: df (pd.DataFrame): DataFrame containing the dataset.
    #        categorical_cols (list): List of columns to encode.
    # Returns: pd.DataFrame: DataFrame with encoded categorical columns.
    #          dict: Dictionary of label encoders used for each column.
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            raise ValueError(f"Categorical column '{col}' not found in dataset.")
    
    return df, label_encoders

# Function to load data and return DataLoaders for training, validation, and testing
def load_data(config) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    # Loads dataset and returns DataLoaders for training, validation, and testing.
    # Args: config (dict): Configuration dictionary containing dataset details.
    # Returns: Tuple: Train, validation, test DataLoaders and input dimension (number of features)
    
    # Load dataset
    try:
        df = pd.read_csv(config["paths"]["data_path"])
        print(f"The file at {config['paths']['data_path']} is loaded.")
    except FileNotFoundError:
        print(f"Error: The file at {config['paths']['data_path']} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the CSV file.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col].fillna(mode_val[0], inplace=True)
    
    # Convert 'Stay' and 'Age' columns from categorical to numerical using range_mapping
    if 'Stay' in df.columns:
        df['Stay'] = apply_range_mapping(df['Stay'])
    
    if 'Age' in df.columns:
        df['Age'] = apply_range_mapping(df['Age'])
    

    # Load or create label encoders
    label_encoder_save_path = config["paths"].get("label_encoder_save_path", None)
    
    if label_encoder_save_path and os.path.isfile(label_encoder_save_path):
        # If the encoder file exists, load it
        label_encoders = joblib.load(label_encoder_save_path)
        # Update df using the existing label encoders
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))  # Transform using the loaded encoder
    else:
        # If the encoder file does not exist, perform encoding and save the encoders
        df, label_encoders = encode_categorical_features(df, config["data"]["categorical_features"])
        if label_encoder_save_path:
            # Save the newly created label encoders
            joblib.dump(label_encoders, config["paths"]["model_home_path"])

    
    # Standardize numerical features
    numerical_cols = config["data"]["numerical_features"]
    scaler = StandardScaler()
    if not all(col in df.columns for col in numerical_cols):
        raise ValueError(f"One or more numerical columns not found in dataset: {numerical_cols}")
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Drop specified columns mentioned in config before splitting into features and target
    if 'drop_columns' in config["data"]:
        drop_columns = config["data"]["drop_columns"]
        df.drop(columns=drop_columns, errors='ignore', inplace=True)
        
        
    # Split features and target
    X = df.drop(columns=['Stay']).values
    y = df['Stay'].values

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Split dataset into training, validation, and test sets
    train_size = int(len(df) * config["data"]["train_split"])
    valid_size = int(len(df) * config["data"]["valid_split"])
    test_size = len(df) - train_size - valid_size

    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    valid_dataset = TensorDataset(X_tensor[train_size:train_size+valid_size], y_tensor[train_size:train_size+valid_size])
    test_dataset = TensorDataset(X_tensor[train_size+valid_size:], y_tensor[train_size+valid_size:])

    # Create DataLoaders
    batch_size = config["data"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get the input dimension (number of features for the model)
    input_dim = X.shape[1]

    return train_loader, valid_loader, test_loader, input_dim, label_encoders

# Function to decode categorical features (Inverse of label encoding)
def decode_categorical_features(input_data, label_encoders, categorical_cols):
    # Decodes the categorical features from numeric values back to their original categorical values.
    # Args: input_data (list): List of numeric values corresponding to categorical features.
    #        label_encoders (dict): Dictionary of label encoders for categorical columns.
    #        categorical_cols (list): List of categorical columns.
    # Returns: dict: Decoded categorical feature names and values.
    
    decoded_data = {}
    
    for i, col in enumerate(categorical_cols):
        if col in label_encoders:
            decoded_data[col] = label_encoders[col].inverse_transform([input_data[i]])[0]
        else:
            decoded_data[col] = input_data[i]  # For numerical columns, just retain the value
    
    return decoded_data

