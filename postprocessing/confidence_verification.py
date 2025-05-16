"""Confidence-based verification for OCR outputs."""

import numpy as np
from model.config import CONFIDENCE_THRESHOLD, CHAR_VECTOR


def calculate_confidence(char_probs):
    """Calculate confidence score for prediction.
    
    Args:
        char_probs: Character probabilities from model output
        
    Returns:
        Confidence score
    """
    # Calculate average of maximum probabilities
    confidence = np.mean([np.max(probs) for probs in char_probs])
    
    return confidence


def find_uncertain_regions(char_probs):
    """Find uncertain regions in prediction.
    
    Args:
        char_probs: Character probabilities from model output
        
    Returns:
        List of indices of uncertain characters
    """
    uncertain_indices = []
    
    for i, probs in enumerate(char_probs):
        max_prob = np.max(probs)
        
        # Mark as uncertain if below threshold
        if max_prob < CONFIDENCE_THRESHOLD:
            uncertain_indices.append(i)
    
    return uncertain_indices


def verify_prediction(text, char_probs):
    """Verify prediction using confidence scores.
    
    Args:
        text: Predicted text
        char_probs: Character probabilities from model output
        
    Returns:
        Verified text with uncertain regions marked
    """
    # Calculate confidence
    confidence = calculate_confidence(char_probs)
    
    # Find uncertain regions
    uncertain_indices = find_uncertain_regions(char_probs)
    
    # Mark uncertain regions in text
    verified_text = list(text)
    for i in uncertain_indices:
        if i < len(verified_text):
            # Add marker for uncertain characters
            verified_text[i] = f"[{verified_text[i]}?]"
    
    verified_text = ''.join(verified_text)
    
    return verified_text, confidence


def get_top_alternatives(char_probs, index, num_alternatives=3):
    """Get top alternative characters for uncertain prediction.
    
    Args:
        char_probs: Character probabilities from model output
        index: Index of character to get alternatives for
        num_alternatives: Number of alternatives to return
        
    Returns:
        List of (character, probability) tuples
    """
    if index >= len(char_probs):
        return []
    
    # Get probabilities for this character
    probs = char_probs[index]
    
    # Get indices of top probabilities
    top_indices = np.argsort(probs)[-num_alternatives:][::-1]
    
    # Convert to characters and probabilities
    alternatives = []
    for idx in top_indices:
        if idx < len(CHAR_VECTOR):
            char = CHAR_VECTOR[idx]
            prob = probs[idx]
            alternatives.append((char, prob))
    
    return alternatives


def highlight_uncertain_regions(text, char_probs, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Highlight uncertain regions in text.
    
    Args:
        text: Predicted text
        char_probs: Character probabilities from model output
        confidence_threshold: Threshold for uncertain regions
        
    Returns:
        Text with uncertain regions highlighted
    """
    if len(text) != len(char_probs):
        return text
    
    highlighted_text = ""
    uncertain_region = False
    
    for i, (char, probs) in enumerate(zip(text, char_probs)):
        max_prob = np.max(probs)
        
        if max_prob < confidence_threshold:
            # Start uncertain region
            if not uncertain_region:
                highlighted_text += "{"
                uncertain_region = True
            highlighted_text += char
        else:
            # End uncertain region
            if uncertain_region:
                highlighted_text += "}"
                uncertain_region = False
            highlighted_text += char
    
    # Close any open uncertain region
    if uncertain_region:
        highlighted_text += "}"
    
    return highlighted_text
