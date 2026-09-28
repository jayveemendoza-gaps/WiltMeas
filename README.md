# Wilt Measure 🌿

A Streamlit-based web application for measuring and analyzing plant wilting using computer vision techniques. Designed for plant health monitoring through the analysis of green pixel indices (ExG and VARI).

## Features

- **Image Upload & Processing**: Upload top-view plant images (JPG/PNG, <5MB)
- **Background Removal**: Automatically removes background using ExG (Excess Green) index
- **Fine-tuning Controls**: Adjust background removal sensitivity in real-time with +/− buttons:
  - **ExG Threshold** (1-100): Controls green pixel sensitivity
  - **Morphological Kernel Size** (1-5): Adjusts noise removal aggressiveness
  - **Morphological Iterations** (1-3): Fine-tunes morphological filtering
  - **VARI Filtering**: Optional additional filtering using VARI index
- **Plant Annotation**: Draw rectangles on the image to mark plant samples
- **Analysis Metrics**:
  - Mean ExG (Excess Green Index)
  - Green Pixel Count
  - Total VARI (Visible Atmospherically Resistant Index)
- **Data Export**: Download results as CSV for further analysis

## Installation

### Requirements
- Python 3.9+
- Streamlit 1.25.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/WiltMeas.git
cd WiltMeas
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run mokorate.py
```

The app will open at `http://localhost:8501`

### Workflow

1. **Upload Image**: Click the file uploader to select your plant image
2. **Fine-tune Detection**: Use the sidebar controls to adjust:
   - Lower ExG Threshold to capture lighter green areas
   - Increase Kernel Size to remove more noise
   - Enable VARI filtering for improved green detection
3. **Annotate Samples**: Draw rectangles around plant samples on the canvas
4. **View Results**: See analyzed metrics for each sample
5. **Export Data**: Download CSV with the analysis results

## Tuning Tips

- **Too much background remaining?** → Lower ExG Threshold or increase Morphological Iterations
- **Losing fine plant details?** → Lower Morphological Kernel Size
- **Capturing unwanted colors?** → Raise ExG Threshold or enable VARI Filtering
- **Large noise particles?** → Increase Morphological Kernel Size

## Project Structure

```
WiltMeas/
├── mokorate.py          # Main Streamlit application
├── requirements.txt     # Python dependencies
├── packages.txt         # System dependencies (for Docker)
├── README.md           # This file
└── venv/               # Virtual environment (not in git)
```

## Dependencies

- **streamlit**: Web framework
- **streamlit-drawable-canvas**: Canvas drawing functionality
- **opencv-python-headless**: Image processing
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **Pillow**: Image manipulation

## Performance Optimization

The app includes cloud-optimized settings:
- Max image size: 800px (configurable)
- Max upload size: 5MB
- Aggressive image downscaling for cloud environments
- Caching for frequently used calculations
- Memory management with garbage collection

## API Reference

### calculate_exg(img)
Calculates Excess Green index for an image array.

### calculate_vari(img)
Calculates Visible Atmospherically Resistant Index.

### process_background_removal(img_array, threshold_val, kernel_size, iterations, use_vari)
Processes background removal with tunable parameters.

### downscale_image(image_bytes, max_dim)
Downscales image for optimization.

## Notes

- Use JPG format for optimal performance
- Best results with top-view plant images
- Ensure good lighting for accurate green detection
- Adjust thresholds based on your specific plant and lighting conditions

## License

[Add your license here]

## Contributing

[Add contribution guidelines]

## Resources

**Sample Images**: Access sample plant images for testing and reference:
- [Google Drive Folder - Sample Pictures](https://drive.google.com/drive/folders/1bXmGzeGQUnW7WsKPiUoYEkyfeDChnIj-)

## Support

For issues and feature requests, please open an issue on GitHub.

## Contact

For inquiries, please contact: **jsmendoza5@up.edu.ph**
