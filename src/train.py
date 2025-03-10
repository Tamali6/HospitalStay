import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import load_data
from model import HospitalStayPredictor
from util import load_config
import joblib


# Argument parser to allow custom config file location
parser = argparse.ArgumentParser(description="Train Hospital Stay Predictor Model")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Set device from config
if config["device"] == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(config["device"])

print(f"\n Using device: {device}")

# Load dataset
train_loader, valid_loader,  _, input_dim, label_encoders = load_data(config)

# Initialize model & move to device
model = HospitalStayPredictor(input_dim).to(device)

# Define loss function
loss_fn = nn.SmoothL1Loss() if config["training"]["loss_function"] == "smoothl1" else nn.MSELoss()

# Define optimizer dynamically from config
if config["training"]["optimizer"] == "adam":
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
elif config["training"]["optimizer"] == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=config["training"]["learning_rate"], momentum=0.9)
else:
    raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")

# Define learning rate scheduler
if not all(key in config["training"]["scheduler"] for key in ["base_lr", "max_lr", "step_size_up", "mode"]):
    raise ValueError("Learning rate scheduler configuration is missing required parameters.")
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer, 
    base_lr=config["training"]["scheduler"]["base_lr"], 
    max_lr=config["training"]["scheduler"]["max_lr"], 
    step_size_up=config["training"]["scheduler"]["step_size_up"], 
    mode=config["training"]["scheduler"]["mode"]
)

# Mixed precision training (optional, only on CUDA)
scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

# Training loop
num_epochs = config["training"]["num_epochs"]
patience = config["training"]["patience"]
best_val_loss = float("inf")
epochs_no_improve = 0

print(f"\n Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move to specified device
        optimizer.zero_grad()

        # Forward pass with mixed precision (if enabled)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        # Backward pass with scaling (if enabled)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()  # Update scheduler per batch
        train_loss += loss.item()

    # Validation
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            valid_loss += loss.item()

    # Compute average losses
    avg_train_loss = train_loss / len(train_loader)
    avg_valid_loss = valid_loss / len(valid_loader)

    # Print epoch summary
    print(f" Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")

    # Early stopping & best model saving
    if avg_valid_loss < best_val_loss:
        best_val_loss = avg_valid_loss
        torch.save(model.state_dict(), config["paths"]["best_model_path"])
        epochs_no_improve = 0
        print(f" Best model updated & saved to {config['paths']['best_model_path']}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(" Early stopping triggered.")
            break

# Save final model
torch.save(model, config["paths"]["model_save_path"])
# Save label encoders
joblib.dump(label_encoders, config["paths"]["label_encoder_save_path"])  # Save the label encoders after training
print(f"\n Training complete. Final model saved at {config['paths']['model_save_path']}")

