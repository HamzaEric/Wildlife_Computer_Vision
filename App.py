from pathlib import Path
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import plotly.express as px
import pandas as pd
import requests
from io import BytesIO

st.set_page_config(layout="wide")

# ==========================================
# 1. DEFINE MODEL ARCHITECTURE
# ==========================================
class DeepNet(nn.Module):
    def __init__(self, num_classes):
        super(DeepNet, self).__init__()

        # Load the ResNet50 architecture
        self.model = models.resnet50(weights=None)

        # Recreate the exact 2-step custom head used during training
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=500, out_features=num_classes)
        )

    def forward(self, x):
        return self.model(x)


WILDLIFE_CLASSES = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent']


# ==========================================
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model(weights_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepNet(num_classes=num_classes)

    weights_file = Path(weights_path)

    if weights_file.exists():
        state_dict = torch.load(weights_file, map_location=device, weights_only=True)

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = f"model.{key}" if not key.startswith('model.') else key
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, device
    else:
        st.error(f"Model file not found at '{weights_file}'. Ensure the file is in the correct directory.")
        return None, device


# ==========================================
# 3. STATE MANAGEMENT FOR SAMPLE IMAGES
# ==========================================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "sample_url" not in st.session_state:
    st.session_state.sample_url = None


def set_sample_image(url):
    """Callback function to set sample image and reset file uploader."""
    st.session_state.sample_url = url
    st.session_state.uploader_key += 1  # Changing the key forces the uploader to reset


# ==========================================
# 4. TRANSFORMS & UI
# ==========================================
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


st.title("ConserVision: Wildlife Camera Trap Classification")
st.write("Upload an image of wildlife or try one of the samples below to run inference.")

# Hardcoded to Wildlife Paths
weights_path = Path("Models/wild_life_deepnet5.pth")
model, device = load_model(weights_path=weights_path, num_classes=len(WILDLIFE_CLASSES))

# --- SAMPLE IMAGES SECTION ---
st.subheader("1. Try a Sample Image")
st.caption("Click one of our samples below to quickly test the model.")

# Using three wildlife samples
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

sample_leopard = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTtAAgCKkQCQvD6_r9ng8qZMLm1J_y5R_XPJ0sdnyc6Ow&s=10"
sample_bird = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXmvf45Rb163jZQVgHwNlA5zEpa-4spQQ8pWqWGFlZEw&s=10"
sample_antelope = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQErg10uCjxpxe-_7YOYZHxzo6QV2Js5fnwrVZfCPLymQ&s=10"

with col_s1:
    st.image(sample_leopard, use_container_width=True)
    # Added unique keys so Streamlit doesn't throw a Duplicate Widget error
    st.button("Try This", key="btn_leo", on_click=set_sample_image, args=(sample_leopard,))
with col_s2:
    st.image(sample_bird, use_container_width=True)
    st.button("Try This", key="btn_bird", on_click=set_sample_image, args=(sample_bird,))
with col_s3:
    st.image(sample_antelope, use_container_width=True)
    st.button("Try This", key="btn_antelope", on_click=set_sample_image, args=(sample_antelope,))

# --- UPLOAD SECTION ---
st.subheader("2. Upload An Image ")
st.warning("Upload Restricted to the following classes:\n WILDLIFE_CLASSES = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent']")
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}"
)

# Determine which image to process
image_to_process = None

if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert("RGB")
    st.session_state.sample_url = None
elif st.session_state.sample_url is not None:
    try:
        # Added a User-Agent header so the image hosts don't block the request
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(st.session_state.sample_url, headers=headers)
        image_to_process = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load sample image from the internet. Error: {e}")

# --- INFERENCE SECTION ---
if image_to_process is not None and model is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image_to_process, caption="Selected Image", use_container_width=True)

    with col2:
        st.subheader("Prediction")

        if st.button("🔍 Classify Image", type="primary", use_container_width=True):
            with st.spinner("Processing image..."):
                input_tensor = image_transforms(image_to_process).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)

                confidence, predicted_idx = torch.max(probabilities, 0)
                predicted_class = WILDLIFE_CLASSES[predicted_idx.item()]
                confidence_percent = confidence.item() * 100

                st.success(f"**Result:** {predicted_class}")
                st.metric(label="Confidence", value=f"{confidence_percent:.2f}%")

                st.write("---")
                st.write("**All Probabilities:**")

                # Pandas DataFrame for Plotly
                df_probs = pd.DataFrame({
                    "Class": WILDLIFE_CLASSES,
                    "Probability": [float(p) for p in probabilities]
                })

                df_probs = df_probs.sort_values(by="Probability", ascending=True)

                fig = px.bar(
                    df_probs,
                    x="Probability",
                    y="Class",
                    orientation='h',
                    text_auto='.1%',
                    labels={"Probability": "Confidence", "Class": ""}
                )

                fig.update_layout(
                    xaxis_tickformat='.0%',
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=350
                )

                st.plotly_chart(fig, use_container_width=True)