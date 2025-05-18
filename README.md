# Handwritten Text Recognition (HTR) System

A complete Handwritten Text Recognition (HTR) system built with a Streamlit web interface. This application uses TrOCR (Transformer-based Optical Character Recognition) to accurately recognize handwritten text from images.

## Features

* **Modern UI**: Clean and intuitive Streamlit interface
* **High-Quality Text Recognition**: Powered by Microsoft's TrOCR models
* **Advanced Image Preprocessing**:

  * Denoising
  * Contrast enhancement
* **Multiple Model Support**:

  * TrOCR Base (fast)
  * TrOCR Large (more accurate)
* **Sample Image Gallery**: Test with provided examples
  ➤ Sample images available in [`data/sample_images`](data/sample_images)
* **Research Report**: Detailed system analysis and findings
  ➤ See [`research/`](research/)
* **Docker Support**: Easy deployment with Docker and docker-compose

## Project Structure

```
DOC_DIGI_HTR/
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
├── research/
│   └── research_report.md       # Research report
├── .gitignore                   # Git ignore file
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker compose configuration
└── app.py                       # Main Streamlit application
```

## Quick Start

### Manual Installation (Recommended)

1. Clone this repository:

   ```bash
   git clone https://github.com/harshsaini88/DOC_DIGI_HTR.git
   cd DOC_DIGI_HTR
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

### Using Docker 

1. Clone this repository:

   ```bash
   git clone https://github.com/harshsaini88/DOC_DIGI_HTR.git
   cd DOC_DIGI_HTR
   ```

2. Build and run the Docker container:

   ```bash
   docker-compose up --build
   ```

3. Access the application at `http://localhost:8501`
