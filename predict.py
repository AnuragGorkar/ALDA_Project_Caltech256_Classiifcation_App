import streamlit as st
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="Image Classifier", page_icon="🔍")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None

def load_model():
    """Load the YOLO model"""
    if st.session_state.model is None:
        with st.spinner('Loading model...'):
            st.session_state.model = YOLO(r'C:\Users\Admin\Desktop\ALDA Project\best.pt')
    return st.session_state.model

def process_image(uploaded_file):
    """Process the image and return top 5 predictions"""
    model = load_model()
    
    # Read the uploaded file
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Make prediction
    results = model(image)
    
    # Get probabilities and class names
    probs = results[0].probs.data.tolist()
    names = results[0].names
    
    # Create a list of (probability, class_name) tuples
    class_probs = [(prob, names[i]) for i, prob in enumerate(probs)]
    
    # Sort by probability (descending) and get top 5
    top5 = sorted(class_probs, reverse=True)[:5]
    
    return top5

def main():
    st.title("Image Classification App")
    st.write("Upload an image to get the top 5 predictions")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width=300)
            
            # Reset the file pointer to the beginning
            uploaded_file.seek(0)
            
            # Make prediction when user clicks the button
            if st.button('Predict'):
                with st.spinner('Analyzing image...'):
                    try:
                        # Get predictions
                        predictions = process_image(uploaded_file)
                        
                        # Display results
                        st.subheader("Top 5 Predictions:")
                        for prob, class_name in predictions:
                            # Create a progress bar for each prediction
                            st.write(f"{class_name}: {prob:.2%}")
                            st.progress(prob)
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
        except Exception as e:
            st.error(f"Error opening image: {str(e)}")

if __name__ == '__main__':
    main()