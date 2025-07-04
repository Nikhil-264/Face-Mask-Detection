import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture exactly like during training
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

# Load saved weights (but strip weird keys manually)
loaded_state_dict = torch.load("best_model.pth", map_location=device)
model_state_dict = model.state_dict()
# Remove unexpected keys like "fc.out_features.weight"
filtered_dict = {k: v for k, v in loaded_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

# Load cleaned weights
model.load_state_dict(filtered_dict, strict=False)
model.eval()

from torchvision import transforms
from PIL import Image

# Define the class labels (must match training)
class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']

# Load and preprocess image
image_path = "maksssksksss432.png"  # 🔁 Replace with your own image path
img = Image.open(image_path).convert("RGB")

# Transform — must match training time transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

input_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dim

# Predict
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    label = class_names[predicted.item()]
    print("✅ Prediction:", label)

