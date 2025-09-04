import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
import io

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
# IMPORTANT: Replace with the direct download link to your .pth file
MODEL_URL = 'https://github.com/gajanand-1/Indian-snake-classifier/releases/download/v1.0/resnet50_snake_classifier3.pth' 

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
def load_model(url):
    """
    Downloads the model from a URL and prepares it for inference.
    """
    # Create the model architecture
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    try:
        # Download the model file from the URL
        st.info(f"Downloading model from URL... this may take a moment.")
        response = requests.get(url)
        response.raise_for_status()  # Check if the download was successful
        
        # Load the model state dictionary from the downloaded content
        buffer = io.BytesIO(response.content)
        model.load_state_dict(torch.load(buffer, map_location=device))
        st.success("Model loaded successfully!")
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading the model from URL: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}. Make sure the URL points to a valid .pth file.")
        return None
    
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

# Load the model and stop if it fails
if 'YOUR_MODEL_DOWNLOAD_URL_HERE' in MODEL_URL:
    st.warning("Please replace 'YOUR_MODEL_DOWNLOAD_URL_HERE.pth' with the actual URL to your model file.")
    st.stop()
else:
    model = load_model(MODEL_URL)
    if model is None:
        st.stop()


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

st.title("Indian 🐍 Snake Species Classifier")
st.write("Upload an image or provide a URL, and the model will try to identify the snake's species.")

# --- Create two tabs for input methods ---
tab1, tab2 = st.tabs(["📁 Upload an Image", "🔗 Provide Image URL"])

image_to_process = None

# --- Tab 1: File Uploader ---
with tab1:
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)

# --- Tab 2: URL Input ---
with tab2:
    url = st.text_input("Enter the image URL here:", "")
    if url:
        try:
            # Send a GET request to the URL
            response = requests.get(url)
            # Raise an exception if the request was unsuccessful
            response.raise_for_status() 
            # Open the image from the response content
            image_to_process = Image.open(io.BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching image from URL: {e}")
        except Exception as e:
            st.error(f"An error occurred: Please check if the URL points to a valid image. Error: {e}")

# --- Common Area for Image Display and Classification ---
if image_to_process is not None:
    st.write("") # Add a little space
    st.image(image_to_process, caption='Image to Classify', use_column_width=True)
    
    # --- Classification Button ---
    if st.button('Classify Snake'):
        # Show a spinner while the model is running
        with st.spinner('Analyzing the image...'):
            predicted_species, confidence = predict(image_to_process, model, idx_to_class, device)
            
            # Display the result
            st.success(f"**Prediction: {predicted_species}**")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
            
            # Display a progress bar for confidence
            st.progress(confidence)

st.markdown("---")
st.write("Built with PyTorch and Streamlit. Model based on ResNet50.")



