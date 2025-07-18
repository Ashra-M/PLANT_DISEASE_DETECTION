import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import tempfile
import os

from resnet_model import get_resnet18_model

# =================== Load model ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = sorted(os.listdir("dataset/data/train"))
num_classes = len(class_names)

model = get_resnet18_model(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# =================== Grad-CAM Setup ===================
gradients = []
activations = []


def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])


def forward_hook(module, input, output):
    activations.append(output)


target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)


def generate_gradcam(input_tensor, class_idx):
    model.zero_grad()
    output = model(input_tensor)
    one_hot = torch.zeros_like(output)
    one_hot[0][class_idx] = 1
    output.backward(gradient=one_hot)

    grads = gradients[-1].cpu().detach().numpy()[0]
    acts = activations[-1].cpu().detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam /= np.max(cam) if np.max(cam) != 0 else 1
    cam = np.uint8(255 * cam)
    cam = cv2.resize(cam, (224, 224))
    return cam


# =================== Streamlit UI ===================
st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a leaf image to predict disease. Optionally generate Grad-CAM & PDF report."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    gradients.clear()
    activations.clear()

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).cpu().squeeze()

    top_index = torch.argmax(probs).item()
    top_prediction = class_names[top_index]
    top_probability = probs[top_index].item() * 100

    st.subheader(f"✅ Result: {top_prediction} ({top_probability:.2f}%)")
    st.subheader("🔍 Class Probabilities:")
    for cls, prob in zip(class_names, probs):
        st.write(f"{cls}: {prob*100:.2f}%")

    # Grad-CAM
    heatmap = generate_gradcam(input_tensor, top_index)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    orig_image = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(orig_image, 0.6, heatmap_img, 0.4, 0)

    st.subheader("🔥 Grad-CAM Heatmap")
    st.image(overlay, use_container_width=True)

    # PDF Download
    if st.button("Download PDF Report"):
        # Save images temporarily
        orig_path = tempfile.mktemp(suffix=".jpg")
        heatmap_path = tempfile.mktemp(suffix=".jpg")

        image.resize((224, 224)).save(orig_path)
        cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Generate PDF
        pdf = FPDF()
        pdf.set_font("Helvetica", size=12)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Plant Disease Detection Report", ln=True)

        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, f"Result: {top_prediction} ({top_probability:.2f}%)", ln=True)
        pdf.ln(5)
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 10, "Class Probabilities:", ln=True)
        for cls, prob in zip(class_names, probs):
            pdf.cell(0, 8, f"{cls}: {prob*100:.2f}%", ln=True)

        # Side-by-side images
        y_pos = pdf.get_y() + 10
        pdf.image(orig_path, x=10, y=y_pos, w=90)
        pdf.image(heatmap_path, x=110, y=y_pos, w=90)
        pdf.ln(75)

        # Output PDF to bytes
        pdf_bytes = pdf.output(dest="S").encode("latin1")

        st.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name="plant_disease_report.pdf",
            mime="application/pdf",
        )

        os.remove(orig_path)
        os.remove(heatmap_path)
