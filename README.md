# Project Title

Brief description of the project.
# Handwritten Text Recognition (HTR) System

A complete Handwritten Text Recognition (HTR) system built with a Streamlit web interface. This application uses TrOCR (Transformer-based Optical Character Recognition) to accurately recognize handwritten text from images.

## Features

- **Modern UI**: Clean and intuitive Streamlit interface
- **High-Quality Text Recognition**: Powered by Microsoft's TrOCR models
- **Advanced Image Preprocessing**: 
  - Denoising
  - Contrast enhancement
  - Text deskewing
- **Multiple Model Support**: 
  - TrOCR Base (fast)
  - TrOCR Large (more accurate)
- **Confidence Controls**: Set threshold for acceptable recognition results
- **Sample Image Gallery**: Test with provided examples
- **Docker Support**: Easy deployment with Docker and docker-compose
- **Modular Design**: Well-organized codebase for easy extension

## Project Structure

```
handwritten-text-recognition/
├── app/
│   ├── __init__.py
│   ├── htr_engine.py            # Core HTR functionality
│   ├── utils.py                 # Utility functions
│   └── image_preprocessing.py   # Image preprocessing functions
├── data/
│   └── sample_images/           # Sample images for testing
├── models/
│   └── .gitkeep                 # Directory to store cached models
├── static/
│   ├── css/
│   │   └── style.css            # Custom CSS
│   └── images/
│       └── logo.png             # Project logo
├── tests/
│   ├── __init__.py
│   └── test_htr_engine.py       # Tests for the HTR engine
├── .gitignore                   # Git ignore file
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker compose configuration
└── app.py                       # Main Streamlit application
```

## Quick Start

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/handwritten-text-recognition.git
   cd handwritten-text-recognition
   ```

2. Build and run the Docker container:
   ```bash
   docker-compose up --build
   ```

3. Access the application at `http://localhost:8501`

### Manual Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/handwritten-text-recognition.git
   cd handwritten-text-recognition
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Access the application at `http://localhost:8501`

## How It Works

### TrOCR Model

This application uses Microsoft's TrOCR (Transformer-based Optical Character Recognition), which combines a Vision Transformer (ViT) encoder with a text decoder to recognize text directly from images. TrOCR is particularly effective for handwritten text recognition.

### Image Preprocessing Pipeline

1. **Grayscale Conversion**: Converts color images to grayscale
2. **Denoising**: Removes noise using Non-Local Means denoising
3. **Deskewing**: Automatically straightens text lines for better recognition
4. **Contrast Enhancement**: Improves text-background separation

### Recognition Process

1. User uploads an image or selects a sample
2. Selected preprocessing steps are applied
3. The image is passed through the TrOCR model
4. Recognition results are displayed with confidence scores
5. Users can download the recognized text

## Configuration Options

### Model Selection
- **TrOCR Base**: Balanced between speed and accuracy
- **TrOCR Large**: Higher accuracy but slower processing

### Preprocessing Options
- **Denoising**: Remove noise from the image
- **Contrast Enhancement**: Improve text visibility
- **Deskewing**: Correct text orientation

### Other Settings
- **Confidence Threshold**: Filter out low-confidence predictions