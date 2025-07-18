import torch
from torchvision import transforms
from PIL import Image
from resnet_model import get_resnet18_model
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transform as training
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Load model
class_names = sorted(os.listdir("dataset/data/train"))
model = get_resnet18_model(num_classes=len(class_names))
model.load_state_dict(torch.load("plant_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Predict
image_path = input("Enter the path to the image: ").strip()
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

print("Predicted Class:", class_names[predicted.item()])
