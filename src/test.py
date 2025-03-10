import argparse
import torch
import torch.nn as nn
from model import HospitalStayPredictor
from dataset import load_data, reverse_range_mapping, decode_categorical_features
from util import load_config
import joblib


# Argument parser for config file
parser = argparse.ArgumentParser(description="Test Hospital Stay Predictor Model")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Set device
device = torch.device(config["device"] if config["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"\n Using device: {device}")

# Load dataset (only test set)
_, _, test_loader, input_dim, label_encoders = load_data(config)

# Initialize model & load weights
model = HospitalStayPredictor(input_dim).to(device)
model.load_state_dict(torch.load(config["paths"]["model_save_path"], map_location=device))


model.eval()

# Define loss function
loss_fn = nn.SmoothL1Loss() if config["training"]["loss_function"] == "smoothl1" else nn.MSELoss()

# Testing loop
test_loss = 0
example_inputs = []
example_predictions = []
example_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move to device
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        test_loss += loss.item()

        # Collecting example inputs, predictions, and targets for analysis
        example_inputs.extend(inputs.cpu().numpy())  # Move to CPU for analysis
        example_predictions.extend(outputs.cpu().numpy())
        example_targets.extend(targets.cpu().numpy())

# Compute average test loss
avg_test_loss = test_loss / len(test_loader)
print(f"\n Test Loss: {avg_test_loss:.4f}")

# Output Analysis (if enabled in config)
if config.get("analyse_output", False):
    print("\n Analyzing output...")

    # Count correct predictions
    correct_predictions = 0
    total_predictions = len(example_inputs)

    print("\nExample Predictions with Accuracy Check:")
    for i in range(total_predictions):
        # Decode the inputs using the label encoders
        decoded_input = decode_categorical_features(example_inputs[i], label_encoders, config["data"]["categorical_features"])

        # Map the actual and predicted stay values to categorical ranges
        actual_range = reverse_range_mapping(float(example_targets[i]))
        predicted_range = reverse_range_mapping(float(example_predictions[i]))

        # Check if prediction is correct
        is_correct = actual_range == predicted_range  # True if ranges match
        if is_correct:
            correct_predictions += 1

        # Print the results for each example
        print(f"Decoded Input: {decoded_input}")
        print(f"Actual Stay: {actual_range} ({float(example_targets[i]):.2f} days)")
        print(f"Predicted Stay: {predicted_range} ({float(example_predictions[i]):.2f} days)")
        print(f"Match: {'✅' if is_correct else '❌'}\n")

    # Calculate and print accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Prediction Accuracy: {accuracy:.2f}%")

