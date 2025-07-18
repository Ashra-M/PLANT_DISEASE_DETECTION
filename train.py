import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from resnet_model import get_resnet18_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Load datasets
train_dataset = datasets.ImageFolder("dataset/data/train", transform=train_transform)
val_dataset = datasets.ImageFolder("dataset/data/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Class weights
class_counts = Counter([label for _, label in train_dataset.samples])
class_weights = torch.tensor(
    [1.0 / class_counts[i] for i in range(len(train_dataset.classes))],
    dtype=torch.float,
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Get model (already has dropout + custom fc inside get_resnet18_model)
model = get_resnet18_model(num_classes=len(train_dataset.classes)).to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_val_loss = float("inf")

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    scheduler.step()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    print(
        f"Epoch [{epoch+1}/10] | Train Loss: {running_loss/len(train_loader):.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Saved new best model with Val Loss: {best_val_loss:.4f}")
