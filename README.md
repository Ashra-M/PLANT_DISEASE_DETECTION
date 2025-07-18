# 🌿 PLANT_DISEASE_DETECTION

This project predicts plant leaf diseases using deep learning and Grad-CAM visualization.

> ⚠️ Currently supports: **Tomato**, **Potato**, and **Pepper** leaf diseases only.

## Features

- Upload leaf image via a web interface (Streamlit)
- Predict top 3 diseases using ResNet18
- Generate Grad-CAM heatmaps
- Download a PDF report with prediction and visual explanation

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Ashra-M/PLANT_DISEASE_DETECTION.git
cd PLANT_DISEASE_DETECTION

# 2. Set up virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
