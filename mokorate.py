import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import gc

st.set_page_config(page_title="Wilt Measure 🌿", layout="wide")
st.title("Wilt Measure 🌿")

# Memory optimization settings
MAX_IMAGE_SIZE = 1024  # Reduced from 800
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB limit

uploaded_file = st.file_uploader("Upload top-view plant image", type=["jpg", "jpeg", "png"])

@st.cache_data(max_entries=3, ttl=300)  # Cache for 5 minutes, max 3 entries
def calculate_exg(img):
    try:
        img = img.astype(np.float32)
        r = img[..., 0]
        g = img[..., 1]
        b = img[..., 2]
        exg = 2 * g - r - b
        return exg
    except Exception:
        st.error("Error calculating ExG")
        return None

@st.cache_data(max_entries=3, ttl=300)
def calculate_vari(img):
    try:
        img = img.astype(np.float32)
        r = img[..., 0]
        g = img[..., 1]
        b = img[..., 2]
        denominator = (g + r - b)
        denominator[denominator == 0] = 1e-6
        vari = (g - r) / denominator
        return vari
    except Exception:
        st.error("Error calculating VARI")
        return None

@st.cache_data(max_entries=3, ttl=300)
def downscale_image(image_bytes, max_dim=MAX_IMAGE_SIZE):
    """Downscale image with memory optimization."""
    try:
        image = Image.open(image_bytes).convert("RGB")
        w, h = image.size
        
        # Check if image is too large
        if w * h > 2000000:  # 2MP limit
            max_dim = min(max_dim, 800)
            
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            return image.resize(new_size, Image.LANCZOS)
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

@st.cache_data(max_entries=3, ttl=300)
def process_background_removal(img_array):
    """Process background removal with error handling."""
    try:
        # Force garbage collection before processing
        gc.collect()
        
        exg = calculate_exg(img_array)
        if exg is None:
            return None, None
            
        exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Smaller kernel for memory efficiency
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        masked_img = img_array.copy()
        masked_img[mask == 0] = [255, 255, 255]
        
        return masked_img, mask
    except Exception as e:
        st.error(f"Error in background removal: {e}")
        return None, None

if uploaded_file is not None:
    # Check file size
    if uploaded_file.size > MAX_UPLOAD_SIZE:
        st.error(f"File too large. Maximum size: {MAX_UPLOAD_SIZE/1024/1024:.1f}MB")
        st.stop()
    
    # Progress bar for user feedback
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Loading image...")
        progress_bar.progress(20)
        
        image = downscale_image(uploaded_file)
        if image is None:
            st.stop()
            
        img_rgb = np.array(image)
        progress_bar.progress(40)
        
        status_text.text("Processing background removal...")
        masked_img, mask = process_background_removal(img_rgb)
        if masked_img is None:
            st.stop()
            
        progress_bar.progress(60)
        
        # Resize for display with memory optimization
        orig_height, orig_width = masked_img.shape[:2]
        display_width = min(600, orig_width)  # Reduced from 700
        scale_ratio = display_width / orig_width
        display_height = int(orig_height * scale_ratio)
        
        if scale_ratio != 1.0:
            masked_img_resized = cv2.resize(masked_img, (display_width, display_height))
            mask_resized = cv2.resize(mask, (display_width, display_height), interpolation=cv2.INTER_NEAREST)
        else:
            masked_img_resized = masked_img
            mask_resized = mask
            
        progress_bar.progress(80)
        img_pil = Image.fromarray(masked_img_resized)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Use columns for better layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Original Image")
            st.image(img_rgb, caption="Original Image", use_column_width=True)
        
        with col2:
            st.markdown("### Background Removed")
            st.image(masked_img_resized, caption="Background Removed", use_column_width=True)
        
        st.markdown("### Annotate Plant Samples")
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.3)",
            stroke_width=2,  # Reduced stroke width
            stroke_color="#0000FF",
            background_image=img_pil,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key="canvas",
        )

        data = []
        
        # Use session state to prevent reprocessing
        if 'last_processed' not in st.session_state:
            st.session_state.last_processed = None
            
        if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
            try:
                annotated_image = masked_img.copy()
                
                for i, shape in enumerate(canvas_result.json_data["objects"], start=1):
                    if shape["type"] == "rect":
                        left = max(0, int(shape["left"] / scale_ratio))
                        top = max(0, int(shape["top"] / scale_ratio))
                        width = int(shape["width"] / scale_ratio)
                        height = int(shape["height"] / scale_ratio)

                        right = min(left + width, masked_img.shape[1])
                        bottom = min(top + height, masked_img.shape[0])

                        # Skip if area is too small
                        if (right - left) * (bottom - top) < 100:
                            continue

                        sample = masked_img[top:bottom, left:right]
                        sample_mask = mask[top:bottom, left:right]
                        
                        exg_sample = calculate_exg(sample)
                        vari_sample = calculate_vari(sample)
                        
                        if exg_sample is None or vari_sample is None:
                            continue
                            
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

                        cv2.rectangle(annotated_image, (left, top), (right, bottom), (255, 0, 0), 2)
                        cv2.putText(annotated_image, f"Plant {i}", (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                # Force garbage collection after processing
                gc.collect()
                
            except Exception as e:
                st.error(f"Error processing annotations: {e}")
                data = []

        if data:
            st.image(annotated_image, caption="Labeled Plant Samples", use_column_width=True)
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "exg_results.csv", "text/csv")
        else:
            st.info("Draw rectangles on the image to analyze plant samples.")

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"An error occurred: {e}")
        st.info("Please try uploading a smaller image or refresh the page.")

else:
    st.info("Please upload an image to begin.")
    st.markdown("**Tips for better performance:**")
    st.markdown("- Use images smaller than 10MB")
    st.markdown("- Prefer JPG format for faster loading")
    st.markdown("- Close other browser tabs to free up memory")