"""Visualization tools for handwritten OCR."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from model.config import CHAR_VECTOR


def visualize_text_regions(image, text_regions):
    """Visualize text regions on image.
    
    Args:
        image: Input image
        text_regions: List of bounding boxes for text regions
        
    Returns:
        Image with text regions visualized
    """
    # Make a copy of image
    vis_image = image.copy()
    
    # Convert to color if grayscale
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Draw bounding boxes
    for region in text_regions:
        x, y, w, h = region
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return vis_image


def visualize_attention(image, attention_weights, text):
    """Visualize attention weights on image.
    
    Args:
        image: Input image
        attention_weights: Attention weights
        text: Predicted text
        
    Returns:
        Image with attention weights visualized
    """
    # Make a copy of image
    vis_image = image.copy()
    
    # Convert to color if grayscale
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Remove axis
    ax.axis('off')
    
    # Display image
    ax.imshow(vis_image)
    
    # Overlay attention weights
    attention_map = np.zeros((vis_image.shape[0], vis_image.shape[1]), dtype=np.float32)
    
    # Reshape attention weights to image dimensions
    h, w = vis_image.shape[:2]
    reshaped_weights = cv2.resize(attention_weights, (w, h))
    
    # Normalize attention weights
    reshaped_weights = (reshaped_weights - np.min(reshaped_weights)) / (np.max(reshaped_weights) - np.min(reshaped_weights))
    
    # Apply colormap
    heatmap = cv2.applyColorMap((reshaped_weights * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay heatmap
    overlay = cv2.addWeighted(vis_image, 0.7, heatmap, 0.3, 0)
    
    # Display overlay
    ax.imshow(overlay)
    
    # Display text
    ax.text(0, h + 20, f"Predicted: {text}", fontsize=12)
    
    # Convert figure to image
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    # Convert canvas to image
    result = np.array(canvas.renderer.buffer_rgba())
    result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    
    # Close figure
    plt.close(fig)
    
    return result


def visualize_confidence(text, char_probs):
    """Visualize confidence of prediction.
    
    Args:
        text: Predicted text
        char_probs: Character probabilities
        
    Returns:
        Image with confidence visualization
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Remove axis
    ax.axis('off')
    
    # Display text with confidence
    for i, (char, prob) in enumerate(zip(text, char_probs)):
        max_prob = np.max(prob)
        
        # Calculate color based on confidence
        r = 1.0 - max_prob
        g = max_prob
        b = 0.0
        
        # Display character with color
        ax.text(i * 20, 20, char, fontsize=16, color=(r, g, b))
    
    # Display legend
    ax.text(0, 60, "Color indicates confidence: Green = High, Red = Low", fontsize=12)
    
    # Convert figure to image
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    # Convert canvas to image
    result = np.array(canvas.renderer.buffer_rgba())
    result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    
    # Close figure
    plt.close(fig)
    
    return result


def visualize_confusion_matrix(true_texts, pred_texts):
    """Visualize confusion matrix for character predictions.
    
    Args:
        true_texts: List of true texts
        pred_texts: List of predicted texts
        
    Returns:
        Confusion matrix image
    """
    # Create character-level confusion matrix
    confusion_matrix = np.zeros((len(CHAR_VECTOR) + 1, len(CHAR_VECTOR) + 1), dtype=np.int32)
    
    # Count character occurrences
    for true_text, pred_text in zip(true_texts, pred_texts):
        # Ensure texts are of same length
        min_len = min(len(true_text), len(pred_text))
        
        for i in range(min_len):
            true_char = true_text[i]
            pred_char = pred_text[i]
            
            # Get indices
            try:
                true_idx = CHAR_VECTOR.index(true_char)
            except ValueError:
                true_idx = len(CHAR_VECTOR)
            
            try:
                pred_idx = CHAR_VECTOR.index(pred_char)
            except ValueError:
                pred_idx = len(CHAR_VECTOR)
            
            # Update confusion matrix
            confusion_matrix[true_idx, pred_idx] += 1
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display confusion matrix
    cax = ax.matshow(confusion_matrix, cmap='Blues')
    
    # Add colorbar
    fig.colorbar(cax)
    
    # Set labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    # Set ticks
    ax.set_xticks(np.arange(len(CHAR_VECTOR) + 1))
    ax.set_yticks(np.arange(len(CHAR_VECTOR) + 1))
    
    # Set tick labels
    tick_labels = list(CHAR_VECTOR) + ['<UNK>']
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add grid
    ax.grid(False)
    
    # Add title
    ax.set_title('Character-level Confusion Matrix')
    
    # Convert figure to image
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    # Convert canvas to image
    result = np.array(canvas.renderer.buffer_rgba())
    result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    
    # Close figure
    plt.close(fig)
    
    return result
