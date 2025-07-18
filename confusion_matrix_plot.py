import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from resnet_model import get_resnet18_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
val_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)
val_dataset = datasets.ImageFolder("dataset/data/val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load model
model = get_resnet18_model(num_classes=len(val_dataset.classes)).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Get predictions
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=val_dataset.classes,
    yticklabels=val_dataset.classes,
)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
