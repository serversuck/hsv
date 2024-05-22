import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to apply color mask
def apply_color_mask(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img

# Streamlit app
def app():
    st.sidebar.title("HSV Color Mask Application")

    # Upload image
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        img = Image.open(uploaded_file)
        img = np.array(img)


        # Get lower and upper HSV bounds
        
        hue_low = st.sidebar.slider("Low Hue", 0, 179, 0)
        hue_high = st.sidebar.slider("High Hue", 0, 179, 180)

        sat_low = st.sidebar.slider("Low Saturation", 0, 255, 0)
        sat_high = st.sidebar.slider("High Saturation", 0, 255, 255)

        val_low = st.sidebar.slider("Low Value", 0, 255, 0)
        val_high = st.sidebar.slider("High Value", 0, 255, 255)

    
        # Create lower and upper HSV bounds
        lower = np.array([hue_low, sat_low, val_low])
        upper = np.array([hue_high, sat_high, val_high])

        # Apply color mask
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masked_img = cv2.inRange(hsv, lower, upper)
        

        # Display the original and masked images
        st.title("Original and Masked Images")
        st.header("Original Image")
        st.image(img, width=400)
        st.header("Masked Image")
        st.image(masked_img, width=400)


if __name__ == "__main__":
    app()
