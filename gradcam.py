import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from resnet_model import get_resnet18_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your model
model = get_resnet18_model(num_classes=15).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Load one image to test
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
img_path = "dataset/data/val/Pepper__bell___Bacterial_spot/0169b9ac-07b9-4be1-8b85-da94481f05a4___NREC_B.Spot 9169.JPG"
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# Hook to capture gradients & activations
gradients = []
activations = []


def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])


def forward_hook(module, input, output):
    activations.append(output)


# Register on last conv layer
target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Forward pass
output = model(input_tensor)
class_idx = output.argmax(dim=1).item()

# Backward pass
model.zero_grad()
one_hot = torch.zeros_like(output)
one_hot[0][class_idx] = 1
output.backward(gradient=one_hot)

# Grad-CAM
grads = gradients[0].cpu().detach().numpy()[0]
acts = activations[0].cpu().detach().numpy()[0]

weights = np.mean(grads, axis=(1, 2))
cam = np.zeros(acts.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * acts[i]

cam = np.maximum(cam, 0)
cam = cam / np.max(cam)
cam = np.uint8(255 * cam)
cam = Image.fromarray(cam).resize((224, 224), resample=Image.BILINEAR)
cam = np.array(cam)

# Overlay
img_np = np.array(img.resize((224, 224)))
plt.imshow(img_np)
plt.imshow(cam, cmap="jet", alpha=0.5)
plt.title(f"Grad-CAM for class: {class_idx}")
plt.axis("off")
plt.tight_layout()
plt.savefig("gradcam_result.png")
plt.show()
