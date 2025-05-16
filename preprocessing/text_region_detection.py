"""Text region detection module for focusing processing on text regions only."""

import cv2
import numpy as np
from model.config import IMAGE_HEIGHT, IMAGE_WIDTH


def detect_text_regions(image):
    """Detect text regions in image.
    
    Args:
        image: Input image
        
    Returns:
        List of bounding boxes for text regions
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to enhance text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to get only text regions
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out very small contours
        if w < 20 or h < 20:
            continue
        
        # Filter based on aspect ratio (text regions typically have width > height)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.5:
            continue
        
        # Add to text regions
        text_regions.append((x, y, w, h))
    
    return text_regions


def extract_text_lines(image, text_regions):
    """Extract text lines from text regions.
    
    Args:
        image: Input image
        text_regions: List of bounding boxes for text regions
        
    Returns:
        List of text line images
    """
    text_lines = []
    
    for region in text_regions:
        x, y, w, h = region
        
        # Extract region from image
        region_img = image[y:y+h, x:x+w]
        
        # Convert to grayscale if needed
        if len(region_img.shape) == 3:
            region_gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        else:
            region_gray = region_img
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(region_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply horizontal projection profile to detect text lines
        h_proj = np.sum(binary, axis=1)
        h_proj_smooth = np.convolve(h_proj, np.ones(10) / 10, mode='same')
        
        # Find peaks in projection profile (text lines)
        peaks = []
        in_peak = False
        start = 0
        
        for i in range(len(h_proj_smooth)):
            if h_proj_smooth[i] > 0 and not in_peak:
                in_peak = True
                start = i
            elif h_proj_smooth[i] == 0 and in_peak:
                in_peak = False
                peaks.append((start, i))
        
        # Extract text lines
        for peak in peaks:
            start, end = peak
            
            # Ignore very short lines
            if end - start < 10:
                continue
            
            # Add some padding
            start = max(0, start - 5)
            end = min(region_img.shape[0], end + 5)
            
            line_img = region_img[start:end, :]
            
            # Resize to fixed height
            scale = IMAGE_HEIGHT / line_img.shape[0]
            new_width = int(line_img.shape[1] * scale)
            
            # Ensure width is not too large
            if new_width > IMAGE_WIDTH:
                new_width = IMAGE_WIDTH
            
            resized = cv2.resize(line_img, (new_width, IMAGE_HEIGHT))
            
            # Pad to fixed width
            if new_width < IMAGE_WIDTH:
                if len(resized.shape) == 3:
                    padded = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, resized.shape[2]), dtype=resized.dtype)
                    padded[:, :new_width, :] = resized
                else:
                    padded = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=resized.dtype)
                    padded[:, :new_width] = resized
            else:
                padded = resized
            
            text_lines.append(padded)
    
    return text_lines


def batch_process_lines(text_lines, batch_size):
    """Process text lines in batches.
    
    Args:
        text_lines: List of text line images
        batch_size: Batch size
        
    Returns:
        List of batches
    """
    batches = []
    
    for i in range(0, len(text_lines), batch_size):
        batch = text_lines[i:i+batch_size]
        
        # Convert to numpy array
        batch_array = np.array(batch)
        
        # Add channel dimension if needed
        if len(batch_array.shape) == 3:
            batch_array = np.expand_dims(batch_array, axis=-1)
        
        batches.append(batch_array)
    
    return batches
