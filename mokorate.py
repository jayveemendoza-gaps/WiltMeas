import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("Wilt Measure 🌿")

uploaded_file = st.file_uploader("Upload top-view plant image", type=["jpg", "jpeg", "png"])

@st.cache_data
def calculate_exg(img):
    img = img.astype(np.float32)
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    exg = 2 * g - r - b
    return exg

@st.cache_data
def calculate_vari(img):
    img = img.astype(np.float32)
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    denominator = (g + r - b)
    denominator[denominator == 0] = 1e-6  # Avoid division by zero
    vari = (g - r) / denominator
    return vari

def downscale_image(image, max_dim=800):
    """Downscale the image to a maximum dimension for faster processing."""
    w, h = image.size
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.LANCZOS)
    return image

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = downscale_image(image)  # Downscale immediately
        img_rgb = np.array(image)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # --- Background removal using ExG threshold ---
    exg = calculate_exg(img_rgb)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Morphological opening to clean small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply mask to RGB image, set background pixels to white
    masked_img = img_rgb.copy()
    masked_img[mask == 0] = [255, 255, 255]

    # --- Resize masked image for canvas display ---
    orig_height, orig_width = masked_img.shape[:2]
    display_width = min(700, orig_width)
    scale_ratio = display_width / orig_width
    display_height = int(orig_height * scale_ratio)
    if scale_ratio != 1.0:
        masked_img_resized = cv2.resize(masked_img, (display_width, display_height))
        mask_resized = cv2.resize(mask, (display_width, display_height), interpolation=cv2.INTER_NEAREST)
    else:
        masked_img_resized = masked_img
        mask_resized = mask

    img_pil = Image.fromarray(masked_img_resized)

    # --- Display Original Image on Top ---
    st.markdown("### Original Image")
    st.image(img_rgb, caption="Original Image", use_column_width=True)

    # --- Display Drawable Canvas Below ---
    st.markdown("### Annotate the Background-Removed Image")
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

    data = []
    annotated_image = masked_img.copy()

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
                green_pixel_count = int(np.sum(sample_mask > 0))
                if green_pixel_count > 0:
                    mean_exg = np.mean(exg_sample[sample_mask > 0])
                    mean_vari = np.mean(vari_sample[sample_mask > 0])
                else:
                    mean_exg = 0
                    mean_vari = 0

                total_vari = mean_vari * green_pixel_count

                data.append({
                    "Sample": f"Plant {i}",
                    "Mean ExG": round(float(mean_exg), 2),
                    "Green Pixels": green_pixel_count,
                    "Total VARI": round(float(total_vari), 2)
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

    # Add institute and contact info as a footer
    st.markdown(
        """
        <hr>
        <div style='text-align: center; font-size: 15px;'>
            <b>PPL, Institute of Plant Breeding, UPLB</b><br>
            Contact: <a href="mailto:jsmendoza5@up.edu.ph">jsmendoza5@up.edu.ph</a>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Please upload an image to begin.")
    st.stop()