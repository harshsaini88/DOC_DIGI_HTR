# utils.py
import os
import glob
from PIL import Image

def get_sample_images():
    """
    Get sample images from the data directory.
    
    Returns:
        dict: Dictionary mapping file paths to PIL Image objects
    """
    sample_dir = os.path.join("data", "sample_images")
    
    # Create directory if it doesn't exist
    os.makedirs(sample_dir, exist_ok=True)
    
    # If no sample images exist, create a placeholder
    sample_files = glob.glob(os.path.join(sample_dir, "*.png")) + \
                  glob.glob(os.path.join(sample_dir, "*.jpg")) + \
                  glob.glob(os.path.join(sample_dir, "*.jpeg"))
    
    if not sample_files:
        # Create a very simple sample image with text
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (400, 200), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        
        try:
            # Try to use a suitable font
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            # Fallback to default font
            font = ImageFont.load_default()
            
        d.text((50, 80), "Sample Handwritten Text", fill=(0, 0, 0), font=font)
        
        # Save the sample image
        sample_path = os.path.join(sample_dir, "sample_text.png")
        img.save(sample_path)
        sample_files = [sample_path]
    
    # Load all sample images
    samples = {}
    for file_path in sample_files:
        try:
            img = Image.open(file_path)
            samples[file_path] = img
        except Exception as e:
            print(f"Error loading sample image {file_path}: {e}")
    
    return samples


def format_confidence(confidence):
    """
    Format confidence score as a percentage.
    
    Args:
        confidence (float): Confidence score between 0 and 1
        
    Returns:
        str: Formatted confidence percentage
    """
    return f"{confidence * 100:.2f}%"

def save_upload_file_temporarily(uploaded_file):
    """
    Save an uploaded file to a temporary location.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        str: Path to the temporary file
    """
    import tempfile
    
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Write content to the temporary file
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return temp_path