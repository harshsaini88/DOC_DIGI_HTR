# image_preprocessing.py
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_image(image, denoise=True, contrast_enhance=True, deskew=True):
    """
    Apply preprocessing steps to improve image quality for HTR.
    
    Args:
        image (PIL.Image): Input image
        denoise (bool): Apply denoising
        contrast_enhance (bool): Enhance image contrast
        deskew (bool): Apply deskewing to straighten text
        
    Returns:
        PIL.Image: Processed image
    """
    # Convert PIL Image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply denoising if selected
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Deskew image if selected
    if deskew:
        gray = deskew_image(gray)
    
    # Convert back to PIL
    processed_img = Image.fromarray(gray)
    
    # Enhance contrast if selected
    if contrast_enhance:
        enhancer = ImageEnhance.Contrast(processed_img)
        processed_img = enhancer.enhance(2.0)  # Enhance contrast by factor of 2
    
    return processed_img

def deskew_image(image):
    """
    Deskew the image to straighten text lines.
    
    Args:
        image (numpy.ndarray): Grayscale image
        
    Returns:
        numpy.ndarray: Deskewed image
    """
    # Threshold the image to get binary image
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # If no significant contours found, return original image
    if len(contours) == 0:
        return image
    
    # Get rotated rectangle from the largest contour
    rect = cv2.minAreaRect(contours[0])
    angle = rect[2]
    
    # Determine the angle to rotate (to make text horizontal)
    if angle < -45:
        angle = 90 + angle
    else:
        angle = angle
    
    # If angle is very small, no need to rotate
    if abs(angle) < 1:
        return image
    
    # Get image dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    rotated = cv2.warpAffine(
        image, 
        rotation_matrix, 
        (w, h),
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated

def binarize_image(image):
    """
    Convert image to binary using adaptive thresholding.
    
    Args:
        image (numpy.ndarray): Grayscale image
        
    Returns:
        numpy.ndarray: Binary image
    """
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        image, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Perform morphological operations to clean up the binary image
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def segment_lines(image):
    """
    Segment text lines in the image.
    
    Args:
        image (numpy.ndarray): Binary image
        
    Returns:
        list: List of line images
    """
    # Horizontal projection profile
    h_proj = np.sum(image, axis=1)
    
    # Find line boundaries
    line_boundaries = []
    line_start = None
    
    for i, count in enumerate(h_proj):
        if count > 0 and line_start is None:
            line_start = i
        elif count == 0 and line_start is not None:
            line_boundaries.append((line_start, i))
            line_start = None
            
    # Handle case where the last line extends to the bottom of the image
    if line_start is not None:
        line_boundaries.append((line_start, len(h_proj)))
    
    # Extract line images
    line_images = []
    for start, end in line_boundaries:
        if end - start > 5:  # Ignore very small segments
            line_images.append(image[start:end, :])
    
    return line_images