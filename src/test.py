import torch
import torch.nn as nn
from model import HospitalStayPredictor
from dataset import get_dataloaders
from utils import load_model

_, _, test_loader = get_dataloaders("data/hospital_stay_data.csv")

input_dim = next(iter(test_loader))[0].shape[1]
model = HospitalStayPredictor(input_dim)
load_model(model, "model/hospital_stay_regression.pth")
model.eval()

criterion = nn.SmoothL1Loss()
test_loss = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_loader):.4f}")

