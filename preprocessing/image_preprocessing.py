"""Image preprocessing utilities for handwritten text recognition."""

import cv2
import numpy as np
from model.config import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS


def preprocess_image(image):
    """Preprocess image for handwritten text recognition.
    
    Args:
        image: Input image
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Remove noise with morphological operations
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Normalize
    normalized = binary / 255.0
    
    return normalized


def resize_to_fixed_size(image):
    """Resize image to fixed size.
    
    Args:
        image: Input image
        
    Returns:
        Resized image
    """
    # Resize to fixed height while preserving aspect ratio
    scale = IMAGE_HEIGHT / image.shape[0]
    new_width = int(image.shape[1] * scale)
    
    # Ensure width is not too large
    if new_width > IMAGE_WIDTH:
        new_width = IMAGE_WIDTH
    
    resized = cv2.resize(image, (new_width, IMAGE_HEIGHT))
    
    # Pad to fixed width
    if new_width < IMAGE_WIDTH:
        if len(resized.shape) == 3:
            padded = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, resized.shape[2]), dtype=resized.dtype)
            padded[:, :new_width, :] = resized
        else:
            padded = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=resized.dtype)
            padded[:, :new_width] = resized
    else:
        padded = resized[:, :IMAGE_WIDTH]
    
    # Add channel dimension if needed
    if len(padded.shape) == 2 and CHANNELS == 1:
        padded = np.expand_dims(padded, axis=-1)
    
    return padded


def enhance_contrast(image):
    """Enhance contrast in image.
    
    Args:
        image: Input image
        
    Returns:
        Contrast enhanced image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced


def deskew_image(image):
    """Deskew image to correct rotation.
    
    Args:
        image: Input image
        
    Returns:
        Deskewed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate skew angle
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def prepare_image_for_model(image):
    """Prepare image for model input.
    
    Args:
        image: Input image
        
    Returns:
        Preprocessed image ready for model input
    """
    # Deskew image
    deskewed = deskew_image(image)
    
    # Enhance contrast
    enhanced = enhance_contrast(deskewed)
    
    # Preprocess image
    preprocessed = preprocess_image(enhanced)
    
    # Resize to fixed size
    resized = resize_to_fixed_size(preprocessed)
    
    # Expand dimensions for batch
    expanded = np.expand_dims(resized, axis=0)
    
    return expanded
