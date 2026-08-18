import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Leaf Classifier",
    layout="wide"
)

# Reduce spacing
st.markdown("""
<style>

.block-container{
    padding-top:1rem;
    padding-bottom:0rem;
}

[data-testid="stFileUploader"]{
    padding:0.2rem;
}

[data-testid="stFileUploaderDropzone"]{
    padding:0.8rem;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Encoder
# =========================

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128*28*28,256),
            nn.ReLU()
        )

    def forward(self,x):
        return self.features(x)

# =========================
# Classes
# =========================

classes = [
    "ashok leaves",
    "banana leaves",
    "blackboard leaves",
    "gulmohar leaves",
    "jamun leaves",
    "lily leaves",
    "neem leaves",
    "paper flower leaves",
    "sadabahar (madagascar) leaves"
]

# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder()

    classifier = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, len(classes))
    )

    model = nn.Sequential(encoder, classifier)

    # Model is stored in the repository root
    model_path = Path(__file__).resolve().parent.parent / "leaf_classifier.pth"

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()
# =========================
# Transform
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# =========================
# UI
# =========================

st.title("🌿 Leaf Species Classifier")
st.caption("Self-Supervised Learning (SimCLR) based model")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs,1)

    prediction = classes[pred.item()]
    confidence = confidence.item()*100

    # Wider results section
    col1, col2 = st.columns([1,2])

    # LEFT
    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # RIGHT
    with col2:

        info1, info2 = st.columns(2)

        with info1:
            st.success(f"Prediction\n\n**{prediction}**")

        with info2:
            st.info(f"Confidence\n\n**{confidence:.2f}%**")

        st.subheader("Class Probabilities")

        prob_dict = {
            classes[i]: float(probs[0][i]*100)
            for i in range(len(classes))
        }

        st.bar_chart(
            prob_dict,
            height=320
        )
    
