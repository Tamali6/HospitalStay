import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.util import load_config

def load_data(config):
    # Loads dataset and returns DataLoaders.
    
    # Load dataset
    df = pd.read_csv(config["paths"]["data_path"])
    
    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Convert categorical ranges
    stay_mapping = { '0-10': 5, '11-20': 15, '21-30': 25, '31-40': 35, '41-50': 45, '51-60': 55, '61-70': 65,
                     '71-80': 75, '81-90': 85, '91-100': 95, 'More than 100 Days': 110 }
    df['Stay'] = df['Stay'].map(stay_mapping)

    # Encode categorical features
    categorical_cols = config["data"]["categorical_features"]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Standardize numerical features
    numerical_cols = config["data"]["numerical_features"]
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Split features and target
    X = df.drop(columns=['Stay']).values
    y = df['Stay'].values

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Split dataset
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

    return train_loader, valid_loader, test_loader, X.shape[1]

