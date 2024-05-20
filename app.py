import streamlit as st
from PIL import Image
import numpy as np
import os
from backend import process_image, apply_blur, adjust_contrast, apply_morphology

def edge_detection_app():
    st.set_page_config(page_title="Edge Detection App", page_icon="🤖")

    st.header("Edge Detection App")
    uploaded_file = st.file_uploader("Upload the image", type=["bmp"])

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        image = np.array(Image.open(uploaded_file))

        # Creating tabs for different preprocessing techniques
        tab1, tab2, tab3 = st.tabs(["Blurring", "Image Contrast", "Morphological Operations"])

        with tab1:
            st.header("Blurring Options")
            blur_method = st.selectbox("Select Blur Method", ["None", "Gaussian Blur", "Median Blur", "Bilateral Filter"])
            with st.sidebar:
                if blur_method == "Gaussian Blur":
                    ksize = st.number_input("Kernel Size", min_value=1, max_value=31, value=5, step=2)
                    image = apply_blur(image, method="gaussian", ksize=ksize)
                elif blur_method == "Median Blur":
                    ksize = st.number_input("Kernel Size", min_value=1, max_value=31, value=5, step=2)
                    image = apply_blur(image, method="median", ksize=ksize)
                elif blur_method == "Bilateral Filter":
                    d = st.number_input("Diameter", min_value=1, max_value=20, value=9)
                    sigma_color = st.number_input("Sigma Color", min_value=1, max_value=100, value=75)
                    sigma_space = st.number_input("Sigma Space", min_value=1, max_value=100, value=75)
                    image = apply_blur(image, method="bilateral", d=d, sigma_color=sigma_color, sigma_space=sigma_space)

        with tab2:
            st.header("Image Contrast Options")
            contrast_method = st.selectbox("Select Contrast Method", ["None", "Histogram Equalization", "CLAHE", "Alpha-Beta"])
            with st.sidebar:
                if contrast_method == "Histogram Equalization":
                    image = adjust_contrast(image, method="hist_eq")
                elif contrast_method == "CLAHE":
                    clip_limit = st.number_input("Clip Limit", min_value=1.0, max_value=40.0, value=2.0)
                    tile_grid_size = st.number_input("Tile Grid Size", min_value=1, max_value=20, value=8)
                    image = adjust_contrast(image, method="clahe", clip_limit=clip_limit, tile_grid_size=tile_grid_size)
                elif contrast_method == "Alpha-Beta":
                    alpha = st.number_input("Alpha (contrast)", min_value=0.0, max_value=3.0, value=1.5)
                    beta = st.number_input("Beta (brightness)", min_value=-100, max_value=100, value=0)
                    image = adjust_contrast(image, method="alpha_beta", alpha=alpha, beta=beta)

        with tab3:
            st.header("Morphological Operations Options")
            morph_method = st.selectbox("Select Morphological Operation", ["None", "Erosion", "Dilation", "Opening", "Closing"])
            with st.sidebar:
                if morph_method != "None":
                    kernel_size = st.number_input("Kernel Size", min_value=1, max_value=20, value=5)
                    image = apply_morphology(image, method=morph_method, kernel_size=kernel_size)

        # Common processing step: thresholding and contour detection
        thresh_value = st.number_input("Adjust Threshold Value", min_value=0, max_value=255, value=114, step=1)
        contour_image = process_image(image, thresh_value)

        # Display the image with contours
        st.image(contour_image, caption='Detected Edges', use_column_width=True)
        save_name = st.text_input("Enter the name to be saved")

        if st.button("Save Processed Image"):
            save_dir = "processed_images"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = f"{save_dir}/{save_name}.bmp"
            Image.fromarray(contour_image).save(filename)
            st.success(f"Processed image saved as {filename}")

# Run the app
if __name__ == "__main__":
    edge_detection_app()
