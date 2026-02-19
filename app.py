import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Fire Detection AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #FF6B35;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 40px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .fire-box {
        background-color: #FFE8E0;
        color: #FF6B35;
        border: 3px solid #FF6B35;
    }
    .nofire-box {
        background-color: #E0F7FA;
        color: #00838F;
        border: 3px solid #00838F;
    }
    .confidence-meter {
        width: 100%;
        height: 30px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'transform' not in st.session_state:
    st.session_state.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model"""
    num_classes = 2  # Default to 2 classes
    
    # Create model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Try to load trained weights if they exist
    model_path = "fire_detection_model.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=st.session_state.device))
        except Exception as e:
            pass  # Use untrained model if load fails
    
    model = model.to(st.session_state.device)
    model.eval()
    return model

# Header
st.markdown('<div class="main-title">🔥 Fire Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning-based Image Classification System</div>', unsafe_allow_html=True)

# Sidebar - Information and Settings
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    ### About This App
    This application uses a **ResNet18** neural network trained to detect whether an image contains fire or not.
    
    ### Classes
    - **0**: No Fire (nofire) 🌙
    - **1**: Fire (fire) 🔥
    
    ### Model Details
    - Architecture: ResNet18
    - Input Size: 224×224 RGB
    - Accuracy: ~74%
    
    ### How to Use
    1. Upload an image (JPG, PNG)
    2. View the prediction result
    3. Check confidence score
    """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    device_info = f"Device: {st.session_state.device}"
    st.info(device_info)
    
    if st.button("🔄 Reload Model", use_container_width=True):
        st.session_state.model = None
        st.cache_resource.clear()
        st.rerun()

# Main content area
col1, col2 = st.columns([1.5, 1], gap="medium")

# Load model
if st.session_state.model is None:
    st.session_state.model = load_model()

# Left column - Upload and Image
with col1:
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Right column - Prediction Results
        with col2:
            st.header("🎯 Prediction Results")
            
            try:
                if st.session_state.model is not None:
                    # Preprocess image
                    img_tensor = st.session_state.transform(image).unsqueeze(0).to(st.session_state.device)
                    
                    # Make prediction
                    with torch.no_grad():
                        output = st.session_state.model(img_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probs, 1)
                    
                    # Get prediction label
                    class_names = ["🌙 No Fire", "🔥 Fire"]
                    prediction = predicted.item()
                    confidence_score = confidence.item() * 100
                    
                    # Display prediction
                    if prediction == 0:
                        st.markdown(
                            f'<div class="prediction-box nofire-box">{class_names[prediction]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box fire-box">{class_names[prediction]}</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Display confidence
                    st.metric("Confidence Level", f"{confidence_score:.2f}%")
                    
                    # Confidence bar
                    st.progress(confidence_score / 100)
                    
                    # Detailed probabilities
                    st.subheader("📊 Probability Distribution")
                    prob_data = {
                        "No Fire": probs[0, 0].item() * 100,
                        "Fire": probs[0, 1].item() * 100
                    }
                    
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("No Fire Probability", f"{prob_data['No Fire']:.2f}%")
                    with col_prob2:
                        st.metric("Fire Probability", f"{prob_data['Fire']:.2f}%")
                    
                    # Chart
                    st.bar_chart(prob_data)
                    
                    if not os.path.exists("fire_detection_model.pth"):
                        st.warning("⚠️ Note: Using untrained model. Train the model in the notebook for better accuracy!")
                else:
                    st.error("❌ Model failed to load.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Try refreshing the page or checking the model file.")
    else:
        # Show placeholder when no image uploaded
        col2.header("🎯 Prediction Results")
        col2.info("👆 Upload an image to see predictions")

# Bottom section - Recent examples or instructions
st.divider()

st.header("📋 Instructions")
cols = st.columns(3)

with cols[0]:
    st.markdown("""
    ### Step 1: Upload
    Click the upload button to select an image file
    """)

with cols[1]:
    st.markdown("""
    ### Step 2: Analyze
    The model will process your image automatically
    """)

with cols[2]:
    st.markdown("""
    ### Step 3: View Results
    Get instant predictions with confidence scores
    """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #999; margin-top: 30px;'>
    <p>🔥 Fire Detection AI v1.0 | Powered by PyTorch & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
