import torch
import torch.nn as nn
from utils import load_config

class HospitalStayPredictor(nn.Module):
    def __init__(self, input_dim):
        super(HospitalStayPredictor, self).__init__()

        # Load hyperparameters from config
        config = load_config()
        hidden_layers = config["model"]["hidden_layers"]
        dropout_rate = config["model"]["dropout"]
        activation_func = config["model"]["activation"]

        # Define activation function
        if activation_func == "relu":
            self.activation = nn.ReLU()
        elif activation_func == "leakyrelu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation_func == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_func}")

        self.dropout = nn.Dropout(dropout_rate)
        self.softplus = nn.Softplus()  # Ensures non-negative output

        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        batch_norms = []

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            batch_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.batch_norms = nn.ModuleList(batch_norms)

        # Final output layer
        self.out = nn.Linear(hidden_layers[-1], 1)

        # Skip connection (first → last hidden layer)
        self.shortcut = nn.Linear(hidden_layers[0], hidden_layers[-1]) if hidden_layers[0] != hidden_layers[-1] else nn.Identity()

    def forward(self, x):
        skip = None  # Store first-layer output for skip connection

        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
            if i == 0:
                skip = x  # Capture first layer output

        # Apply skip connection
        x += self.shortcut(skip)

        # Output layer with Softplus
        return self.softplus(self.out(x))

