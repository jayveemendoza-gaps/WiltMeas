
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import gc
import io


st.set_page_config(page_title="Wilt Measure 🌿", layout="wide")
st.title("Wilt Measure 🌿")


# Cloud-optimized memory settings
MAX_IMAGE_SIZE = 800  # Lowered for cloud
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB limit for cloud

# Sidebar for tuning parameters
with st.sidebar:
    st.markdown("### 🔧 Background Removal Tuning")
    
    # ExG Threshold with +/- buttons
    st.markdown("**ExG Threshold**")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("−", key="exg_minus", use_container_width=True):
            st.session_state.exg_val = max(1, st.session_state.get('exg_val', 50) - 5)
            st.experimental_rerun()
    with col2:
        threshold_value = st.slider("", min_value=1, max_value=100, value=st.session_state.get('exg_val', 50), 
                                   label_visibility="collapsed",
                                   help="Lower = more sensitive to green. Higher = stricter green detection")
        st.session_state.exg_val = threshold_value
    with col3:
        if st.button("✚", key="exg_plus", use_container_width=True):
            st.session_state.exg_val = min(100, st.session_state.get('exg_val', 50) + 5)
            st.experimental_rerun()
    
    # Morphological Kernel with +/- buttons
    st.markdown("**Morphological Kernel Size**")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("−", key="kernel_minus", use_container_width=True):
            st.session_state.kernel_val = max(1, st.session_state.get('kernel_val', 2) - 1)
            st.experimental_rerun()
    with col2:
        morpho_kernel = st.slider("", min_value=1, max_value=5, value=st.session_state.get('kernel_val', 2),
                                 label_visibility="collapsed",
                                 help="Larger kernel removes more small noise")
        st.session_state.kernel_val = morpho_kernel
    with col3:
        if st.button("✚", key="kernel_plus", use_container_width=True):
            st.session_state.kernel_val = min(5, st.session_state.get('kernel_val', 2) + 1)
            st.experimental_rerun()
    
    # Morphological Iterations with +/- buttons
    st.markdown("**Morphological Iterations**")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("−", key="iter_minus", use_container_width=True):
            st.session_state.iter_val = max(1, st.session_state.get('iter_val', 1) - 1)
            st.experimental_rerun()
    with col2:
        morpho_iter = st.slider("", min_value=1, max_value=3, value=st.session_state.get('iter_val', 1),
                               label_visibility="collapsed",
                               help="More iterations = more aggressive noise removal")
        st.session_state.iter_val = morpho_iter
    with col3:
        if st.button("✚", key="iter_plus", use_container_width=True):
            st.session_state.iter_val = min(3, st.session_state.get('iter_val', 1) + 1)
            st.experimental_rerun()
    
    use_vari_filter = st.checkbox("Use VARI filtering", value=False,
                                  help="Add additional VARI index filtering for better green detection")

uploaded_file = st.file_uploader("Upload top-view plant image (JPG preferred, <5MB)", type=["jpg", "jpeg", "png"])

# Store uploaded file in session state to persist across reruns
if uploaded_file is not None:
    st.session_state.uploaded_file_data = uploaded_file.getvalue()
    st.session_state.uploaded_file_name = uploaded_file.name

# Use stored file if available
if 'uploaded_file_data' in st.session_state:
    uploaded_file = io.BytesIO(st.session_state.uploaded_file_data)
    uploaded_file.name = st.session_state.uploaded_file_name

@st.cache_data(max_entries=2, ttl=300)
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

@st.cache_data(max_entries=2, ttl=300)
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

def downscale_image(image_bytes, max_dim=MAX_IMAGE_SIZE):
    """Downscale image for cloud with memory optimization."""
    try:
        # Ensure we're at the beginning of the file
        if hasattr(image_bytes, 'seek'):
            image_bytes.seek(0)
        image = Image.open(image_bytes).convert("RGB")
        w, h = image.size
        # Aggressive downscale for cloud
        if w * h > 1_000_000:
            max_dim = min(max_dim, 600)
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            return image.resize(new_size, Image.LANCZOS)
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

@st.cache_data(max_entries=2, ttl=300)
def process_background_removal(img_array, threshold_val=50, kernel_size=2, iterations=1, use_vari=False):
    """Process background removal with tunable parameters."""
    try:
        gc.collect()
        exg = calculate_exg(img_array)
        if exg is None:
            return None, None
        
        exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Use manual threshold instead of OTSU for better control
        # Scale threshold_val (1-100) to actual threshold range (1-255)
        scaled_threshold = int((threshold_val / 100.0) * 255)
        _, mask = cv2.threshold(exg_norm, scaled_threshold, 255, cv2.THRESH_BINARY)
        
        # Optional: Add VARI filtering for better green detection
        if use_vari:
            vari = calculate_vari(img_array)
            if vari is not None:
                vari_norm = cv2.normalize(vari, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, vari_mask = cv2.threshold(vari_norm, int(threshold_val * 0.8), 255, cv2.THRESH_BINARY)
                # Combine both masks (AND operation - pixel must pass both tests)
                mask = cv2.bitwise_and(mask, vari_mask)
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        masked_img = img_array.copy()
        masked_img[mask == 0] = [255, 255, 255]
        gc.collect()
        return masked_img, mask
    except Exception as e:
        st.error(f"Error in background removal: {e}")
        return None, None

if uploaded_file is not None:
    # Get file size (handle both original upload and session state BytesIO)
    if hasattr(uploaded_file, 'size'):
        file_size = uploaded_file.size
    else:
        file_size = len(st.session_state.uploaded_file_data)
    
    if file_size > MAX_UPLOAD_SIZE:
        st.error(f"File too large. Maximum size: {MAX_UPLOAD_SIZE/1024/1024:.1f}MB")
        st.stop()

    with st.spinner("Processing image..."):
        try:
            image = downscale_image(uploaded_file)
            if image is None:
                st.stop()
            img_rgb = np.array(image)
            masked_img, mask = process_background_removal(img_rgb, threshold_val=threshold_value, 
                                                          kernel_size=morpho_kernel, 
                                                          iterations=morpho_iter,
                                                          use_vari=use_vari_filter)
            if masked_img is None:
                st.stop()

            orig_height, orig_width = masked_img.shape[:2]
            display_width = min(500, orig_width)
            scale_ratio = display_width / orig_width
            display_height = int(orig_height * scale_ratio)

            if scale_ratio != 1.0:
                masked_img_resized = cv2.resize(masked_img, (display_width, display_height))
                mask_resized = cv2.resize(mask, (display_width, display_height), interpolation=cv2.INTER_NEAREST)
            else:
                masked_img_resized = masked_img
                mask_resized = mask

            img_pil = Image.fromarray(masked_img_resized)

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
                stroke_width=2,
                stroke_color="#0000FF",
                background_image=img_pil,
                update_streamlit=True,
                height=display_height,
                width=display_width,
                drawing_mode="rect",
                key="canvas",
            )

            data = []
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
            st.error(f"An error occurred: {e}")
            st.info("Please try uploading a smaller image or refresh the page.")
else:
    st.info("Please upload an image to begin.")
    st.markdown("**Cloud tips:**")
    st.markdown("- Use images smaller than 5MB")
    st.markdown("- Prefer JPG format for faster loading")
    st.markdown("- Avoid uploading very large or high-res images")
    
    st.divider()
    st.markdown("### 📁 Resources")
    st.markdown("[📸 Sample Pictures](https://drive.google.com/drive/folders/1bXmGzeGQUnW7WsKPiUoYEkyfeDChnIj-)")
    
    st.divider()
    st.markdown("### 📧 Contact")
    st.markdown("For inquiries: **jsmendoza5@up.edu.ph**")