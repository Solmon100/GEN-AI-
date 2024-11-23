import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pyttsx3
import cv2
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from Gemini import Gemini_API_KEY
import numpy as np

# Set up API key for Gemini
key = Gemini_API_KEY
genai.configure(api_key=key)

# Initialize models
gen_model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
yolo_model = YOLO('yolov8n.pt')

# Streamlit app
st.title("🤖AI-Powered Visual & Text Assistant")
#st.subheader("Leverage Generative AI and Object Detection for Image Understanding")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    try:
        # Read uploaded file as bytes for compatibility with OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            st.error("Failed to decode the uploaded image. Please ensure it is a valid image file.")
        else:
            # Display image using PIL for Streamlit
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Generate Image Description
            st.subheader("Image Description")
            try:
                prompt = "Generate a brief text description of this image."
                config = genai.GenerationConfig(temperature=0.2)
                description = gen_model.generate_content([prompt, image], generation_config=config).text
                st.write(description)

                # Text-to-Speech Option
                tts_option = st.checkbox("Enable Text-to-Speech")
                if tts_option:
                    engine = pyttsx3.init()
                    engine.say(description)
                    engine.runAndWait()
            except Exception as e:
                st.error(f"Failed to generate image description: {e}")

            # Perform Object Detection
            st.subheader("Object Detection")
            try:
                results = yolo_model(img_cv2)
                for result in results:
                    annotated_image = result.plot()
                st.image(annotated_image, caption="Detected Objects", use_column_width=True)
            except Exception as e:
                st.error(f"Failed to perform object detection: {e}")
    except Exception as e:
        st.error(f"An error occurred while processing the uploaded image: {e}")
else:
    st.info("Please upload an image to proceed.")
