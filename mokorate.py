import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("Manual Plant Sample ExG Analyzer 🌿")

uploaded_file = st.file_uploader("Upload top-view plant image", type=["jpg", "jpeg", "png"])

def calculate_exg(image):
    B, G, R = cv2.split(image.astype(np.float32))
    exg = 2 * G - R - B
    return exg

def calculate_vari(image):
    B, G, R = cv2.split(image.astype(np.float32))
    denominator = (G + R - B)
    denominator[denominator == 0] = 1  # Prevent division by zero
    vari = (G - R) / denominator
    return vari

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Background removal using ExG threshold ---
    exg = calculate_exg(img_rgb)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Morphological opening to clean small noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply mask to RGB image, set background pixels to white
    masked_img = img_rgb.copy()
    masked_img[mask == 0] = [255, 255, 255]

    # --- Resize masked image for canvas display, but only shrink if too large ---
    orig_height, orig_width = masked_img.shape[:2]
    # Let user pick display width, default 700, max is image width or 1000
    display_width = st.slider(
        "Canvas display width (px)",
        min_value=300,
        max_value=min(1000, orig_width),
        value=min(700, orig_width),
        step=10
    )
    scale_ratio = display_width / orig_width
    display_height = int(orig_height * scale_ratio)
    if scale_ratio != 1.0:
        masked_img_resized = cv2.resize(masked_img, (display_width, display_height))
        mask_resized = cv2.resize(mask, (display_width, display_height), interpolation=cv2.INTER_NEAREST)
    else:
        masked_img_resized = masked_img
        mask_resized = mask

    img_pil = Image.fromarray(masked_img_resized)

    # Add horizontal scroll for the canvas
    st.markdown(
        """
        <style>
        .scrollable-canvas {overflow-x: auto;}
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.container():
        st.markdown('<div class="scrollable-canvas">', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.3)",
            stroke_width=3,
            stroke_color="#0000FF",
            background_image=img_pil,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key="canvas",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    data = []
    annotated_image = masked_img.copy()  # Use background-removed image for annotation

    if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
        for i, shape in enumerate(canvas_result.json_data["objects"], start=1):
            if shape["type"] == "rect":
                left = int(shape["left"] / scale_ratio)
                top = int(shape["top"] / scale_ratio)
                width = int(shape["width"] / scale_ratio)
                height = int(shape["height"] / scale_ratio)

                # Ensure bounds don't exceed original image size
                right = min(left + width, masked_img.shape[1])
                bottom = min(top + height, masked_img.shape[0])

                # Extract sample and corresponding mask
                sample = masked_img[top:bottom, left:right]
                sample_mask = mask[top:bottom, left:right]
                exg_sample = calculate_exg(sample)
                vari_sample = calculate_vari(sample)
                # Only consider plant pixels (mask > 0)
                green_pixel_count = int(np.sum(sample_mask > 0))
                if green_pixel_count > 0:
                    mean_exg = np.mean(exg_sample[sample_mask > 0])
                    mean_vari = np.mean(vari_sample[sample_mask > 0])
                else:
                    mean_exg = 0
                    mean_vari = 0

                true_green_value = mean_vari * green_pixel_count

                data.append({
                    "Sample": f"Plant {i}",
                    "Mean ExG": round(float(mean_exg), 2),
                    "Green Pixels": green_pixel_count,
                    "True Green Value": round(float(true_green_value), 2)
                })

                # Draw rectangle and label on annotated image
                cv2.rectangle(annotated_image, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(annotated_image, f"Plant {i}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if data:
        st.image(annotated_image, caption="Labeled Plant Samples", use_column_width=True)
        df = pd.DataFrame(data)
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "exg_results.csv", "text/csv")
    else:
        st.info("Draw rectangles on the image to analyze plant samples.")