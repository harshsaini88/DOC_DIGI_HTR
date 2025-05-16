"""Language model for error correction in OCR output."""

import os
import re
import symspellpy
from symspellpy import SymSpell, Verbosity
from model.config import (
    LANGUAGE_MODEL_WEIGHT,
    MAX_EDIT_DISTANCE,
    LANGUAGE_MODEL_DICT_PATH,
    CHAR_VECTOR
)


class LanguageModel:
    """Language model for error correction in OCR output."""
    
    def __init__(self):
        """Initialize language model."""
        # Initialize SymSpell for spell checking
        self.sym_spell = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE, prefix_length=7)
        
        # Load dictionary if it exists, otherwise create it
        if os.path.exists(LANGUAGE_MODEL_DICT_PATH):
            self.sym_spell.load_dictionary(LANGUAGE_MODEL_DICT_PATH, term_index=0, count_index=1)
        else:
            # Create a simple dictionary from CHAR_VECTOR
            with open(LANGUAGE_MODEL_DICT_PATH, 'w') as f:
                for char in CHAR_VECTOR:
                    if char.isalpha():
                        f.write(f"{char} 1\n")
            
            self.sym_spell.load_dictionary(LANGUAGE_MODEL_DICT_PATH, term_index=0, count_index=1)
    
    def _tokenize(self, text):
        """Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        # Remove special characters and split by whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        return words
    
    def correct_text(self, text, char_probs=None):
        """Correct text using language model.
        
        Args:
            text: Input text
            char_probs: Character probabilities from model output
            
        Returns:
            Corrected text
        """
        words = self._tokenize(text)
        corrected_words = []
        
        for word in words:
            # Skip correction for short words and non-alphabetic words
            if len(word) <= 2 or not word.isalpha():
                corrected_words.append(word)
                continue
            
            # Get suggestions from SymSpell
            suggestions = self.sym_spell.lookup(word.lower(), Verbosity.CLOSEST, 
                                               max_edit_distance=MAX_EDIT_DISTANCE)
            
            if suggestions:
                best_suggestion = suggestions[0].term
                
                # Preserve original capitalization
                if word[0].isupper():
                    best_suggestion = best_suggestion.capitalize()
                
                corrected_words.append(best_suggestion)
            else:
                corrected_words.append(word)
        
        # Join words back into text
        corrected_text = ' '.join(corrected_words)
        
        # Fix common punctuation issues
        corrected_text = self._fix_punctuation(corrected_text)
        
        return corrected_text
    
    def _fix_punctuation(self, text):
        """Fix common punctuation issues in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with fixed punctuation
        """
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        text = re.sub(r'([({])\s+', r'\1', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix missing space after punctuation
        text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def combine_with_model_output(self, text, char_probs):
        """Combine language model correction with model output probabilities.
        
        Args:
            text: Input text
            char_probs: Character probabilities from model output
            
        Returns:
            Corrected text
        """
        # Get language model correction
        lm_corrected = self.correct_text(text)
        
        # If we don't have character probabilities, just return language model correction
        if char_probs is None:
            return lm_corrected
        
        # Calculate confidence score for original text
        confidence = sum(max(probs) for probs in char_probs) / len(char_probs)
        
        # If confidence is high, trust the model output more
        if confidence > 0.9:
            weight = 1.0 - LANGUAGE_MODEL_WEIGHT
        else:
            weight = LANGUAGE_MODEL_WEIGHT
        
        # Simple weighting scheme: if confidence is low, favor language model
        # For now, just return the language model correction
        # In a more sophisticated system, we could blend the outputs based on confidence
        return lm_corrected