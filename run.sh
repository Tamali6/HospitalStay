#!/bin/bash

echo "Starting Hospital Stay Prediction..."

# Train the model using config.yaml
echo "Training the model..."
python src/train.py --config config.yaml

# Test the model
echo "Evaluating the model..."
python src/test.py --config config.yaml

echo "Execution completed."

