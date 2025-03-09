import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import load_data
from src.model import HospitalStayPredictor
from src.util import load_config

# Load configuration
config = load_config()

# Load dataset
train_loader, valid_loader, test_loader, input_dim = load_data(config)

# Initialize model
model = HospitalStayPredictor(input_dim, config["model"]["hidden_layers"], config["model"]["dropout"], config["model"]["activation"])

# Define loss function
loss_fn = nn.SmoothL1Loss() if config["training"]["loss_function"] == "smoothl1" else nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

# Define learning rate scheduler
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer, 
    base_lr=config["training"]["scheduler"]["base_lr"], 
    max_lr=config["training"]["scheduler"]["max_lr"], 
    step_size_up=config["training"]["scheduler"]["step_size_up"], 
    mode=config["training"]["scheduler"]["mode"]
)

# Training loop
num_epochs = config["training"]["num_epochs"]
patience = config["training"]["patience"]
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            valid_loss += loss.item()
    
    # Print epoch summary
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    # Learning rate adjustment
    scheduler.step(valid_loss)

    # Early stopping
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(model.state_dict(), config["paths"]["best_model_path"])
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Save final model
torch.save(model.state_dict(), config["paths"]["model_save_path"])

