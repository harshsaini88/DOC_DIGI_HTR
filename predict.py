"""Model inference script for handwritten OCR."""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from model.cnn_lstm_model import create_ctc_model
from preprocessing.text_region_detection import detect_text_regions, extract_text_lines, batch_process_lines
from preprocessing.image_preprocessing import prepare_image_for_model
from postprocessing.language_model import LanguageModel
from postprocessing.confidence_verification import verify_prediction, calculate_confidence
from utils.visualization import visualize_text_regions
from model.config import BATCH_SIZE, CHAR_VECTOR


def predict_text(image_path, model_path, batch_size=BATCH_SIZE, visualize=False, use_language_model=True):
    """Predict text in image.
    
    Args:
        image_path: Path to image
        model_path: Path to model
        batch_size: Batch size
        visualize: Whether to visualize results
        use_language_model: Whether to use language model for error correction
        
    Returns:
        Predicted text
    """
    # Load image
    image = cv2.imread(image_path)
    
    # Detect text regions
    text_regions = detect_text_regions(image)
    
    # Extract text lines
    text_lines = extract_text_lines(image, text_regions)
    
    # Process text lines in batches
    batches = batch_process_lines(text_lines, batch_size)
    
    # Load model
    _, pred_model = create_ctc_model()
    pred_model.load_weights(model_path)
    
    # Initialize language model if requested
    if use_language_model:
        lm = LanguageModel()
    
    # Initialize results
    predicted_texts = []
    
    # Process each batch
    for batch in batches:
        # Predict
        char_probs = pred_model.predict(batch)
        
        # Process each sample in batch
        for i in range(char_probs.shape[0]):
            # Get probabilities for this sample
            probs = char_probs[i]
            
            # Get predicted indices (greedy decoding)
            pred_indices = np.argmax(probs, axis=1)
            
            # Group repeated characters
            grouped_indices = []
            prev_idx = -1
            for idx in pred_indices:
                if idx != prev_idx:
                    grouped_indices.append(idx)
                    prev_idx = idx
            
            # Remove blank character
            pred_indices = [idx for idx in grouped_indices if idx < len(CHAR_VECTOR)]
            
            # Convert to text
            pred_text = ''.join([CHAR_VECTOR[idx] for idx in pred_indices])
            
            # Apply language model correction if requested
            if use_language_model:
                pred_text = lm.correct_text(pred_text)
            
            # Calculate confidence
            confidence = calculate_confidence(probs)
            
            # Add to results
            predicted_texts.append((pred_text, confidence))
    
    # Visualize results if requested
    if visualize:
        # Visualize text regions
        vis_image = visualize_text_regions(image, text_regions)
        
        # Display predicted text
        for i, (text, confidence) in enumerate(predicted_texts):
            print(f"Text line {i+1}: {text} (confidence: {confidence:.2f})")
            
            # Draw text on image
            y_pos = i * 30 + 30
            cv2.putText(vis_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display image
        cv2.imshow('Detected Text', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Return predicted texts
    return predicted_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict text in image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize results')
    parser.add_argument('--no_language_model', action='store_true', help='Disable language model correction')
    
    args = parser.parse_args()
    
    predict_text(args.image_path, args.model_path, args.batch_size, args.visualize, not args.no_language_model)