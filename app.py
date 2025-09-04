import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==============================================================================
# App Configuration and Model Loading
# ==============================================================================

st.set_page_config(
    page_title="Snake Species Classifier",
    page_icon="🐍",
    layout="centered"
)

# --- Model Constants and Class Mappings ---
NUM_CLASSES = 13
MODEL_PATH = 'resnet50_snake_classifier3.pth' # Make sure this path is correct

# Mapping from index to class name
idx_to_class = {
    0: 'Checkered Keelback',
    1: 'Common Bronzesback Tree Snake',
    2: 'Cobra',
    3: 'Fake Viper',
    4: 'Flying Snake',
    5: 'Green Vine Snake',
    6: 'INDIAN_BOA_NON-VENOMOUS',
    7: 'King Cobra',
    8: 'Python',
    9: 'Red-necked Keelback',
    10: 'Russell Vipper',
    11: 'Sea Krait',
    12: 'Striped Keelback'
}

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Cached Model Loading Function ---
# Using st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """
    Loads and prepares the ResNet50 model.
    """
    # Create the model architecture
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Load the trained weights
    # The map_location ensures the model loads correctly onto the specified device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Move model to the device and set it to evaluation mode
    model.to(device)
    model.eval()
    return model

# --- Image Transformation ---
# Define the same transformations used during validation
prediction_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model
model = load_model()


# ==============================================================================
# Prediction Function
# ==============================================================================

def predict(image: Image.Image, model, idx_to_class, device):
    """
    Takes a PIL image, preprocesses it, and returns the predicted class and confidence.
    """
    # Ensure the image is in RGB format
    image = image.convert("RGB")
    
    # Apply transformations and add a batch dimension
    image_tensor = prediction_transforms(image).unsqueeze(0)
    
    # Move tensor to the correct device
    image_tensor = image_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get the top prediction
    confidence, predicted_idx = torch.max(probabilities, 1)
    predicted_class = idx_to_class[predicted_idx.item()]
    
    return predicted_class, confidence.item()


# ==============================================================================
# Streamlit User Interface
# ==============================================================================

st.title("🐍 Snake Species Classifier")
st.write("Upload an image of a snake, and the model will try to identify its species.")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("") # Add a little space

    # --- Classification Button ---
    if st.button('Classify Snake'):
        # Show a spinner while the model is running
        with st.spinner('Analyzing the image...'):
            predicted_species, confidence = predict(image, model, idx_to_class, device)
            
            # Display the result
            st.success(f"**Prediction: {predicted_species}**")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
            
            # Display a progress bar for confidence
            st.progress(confidence)

st.markdown("---")
st.write("Built with PyTorch and Streamlit. Model based on ResNet50.")