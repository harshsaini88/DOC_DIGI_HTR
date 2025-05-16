"""Data loading utilities for handwritten OCR."""

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from model.config import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, CHAR_VECTOR, NUM_CLASSES


class DataLoader:
    """Data loader for handwritten OCR."""
    
    def __init__(self, data_dir, batch_size=32, validation_split=0.2):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing data
            batch_size: Batch size
            validation_split: Fraction of data to use for validation
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Load data
        self.images, self.texts = self._load_data()
        
        # Split data
        self._split_data()
    
    def _load_data(self):
        """Load data from directory.
        
        Returns:
            Tuple of (images, texts)
        """
        images = []
        texts = []
        
        # Look for CSV file with image paths and corresponding text
        csv_path = os.path.join(self.data_dir, 'labels.csv')
        
        if os.path.exists(csv_path):
            # Load from CSV
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                image_path = os.path.join(self.data_dir, row['image_path'])
                text = row['text']
                
                if os.path.exists(image_path):
                    # Load image
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Preprocess image
                    image = self._preprocess_image(image)
                    
                    images.append(image)
                    texts.append(text)
        
        return np.array(images), texts
    
    def _preprocess_image(self, image):
        """Preprocess image.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Resize to fixed height
        scale = IMAGE_HEIGHT / image.shape[0]
        new_width = int(image.shape[1] * scale)
        
        # Ensure width is not too large
        if new_width > IMAGE_WIDTH:
            new_width = IMAGE_WIDTH
        
        resized = cv2.resize(image, (new_width, IMAGE_HEIGHT))
        
        # Pad to fixed width
        padded = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=resized.dtype)
        padded[:, :new_width] = resized
        
        # Normalize
        normalized = padded / 255.0
        
        # Add channel dimension
        if CHANNELS == 1:
            normalized = np.expand_dims(normalized, axis=-1)
        
        return normalized
    
    def _split_data(self):
        """Split data into training and validation sets."""
        # Get number of samples
        num_samples = len(self.images)
        
        # Get number of validation samples
        num_val_samples = int(num_samples * self.validation_split)
        
        # Shuffle indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        # Split indices
        val_indices = indices[:num_val_samples]
        train_indices = indices[num_val_samples:]
        
        # Split data
        self.train_images = self.images[train_indices]
        self.train_texts = [self.texts[i] for i in train_indices]
        
        self.val_images = self.images[val_indices]
        self.val_texts = [self.texts[i] for i in val_indices]
    
    def _text_to_labels(self, text):
        """Convert text to labels.
        
        Args:
            text: Input text
            
        Returns:
            Labels as numpy array
        """
        labels = []
        
        for char in text:
            # Find index of character in CHAR_VECTOR
            try:
                index = CHAR_VECTOR.index(char)
                labels.append(index)
            except ValueError:
                # Use blank character for unknown characters
                labels.append(len(CHAR_VECTOR))
        
        return np.array(labels)
    
    def _labels_to_text(self, labels):
        """Convert labels to text.
        
        Args:
            labels: Input labels
            
        Returns:
            Text as string
        """
        text = ""
        
        for label in labels:
            if label < len(CHAR_VECTOR):
                text += CHAR_VECTOR[label]
            else:
                # Skip blank character
                pass
        
        return text
    
    def _generate_ctc_inputs(self, images, texts):
        """Generate inputs for CTC loss.
        
        Args:
            images: Input images
            texts: Input texts
            
        Returns:
            Tuple of (inputs, outputs)
        """
        # Prepare inputs for CTC loss
        input_length = np.ones((len(images), 1)) * (IMAGE_WIDTH // 8)  # Assuming 3 pooling layers of factor 2
        
        # Prepare outputs for CTC loss
        labels = [self._text_to_labels(text) for text in texts]
        label_length = np.array([len(label) for label in labels])
        
        # Pad labels to maximum length
        max_label_length = max(label_length)
        padded_labels = np.zeros((len(labels), max_label_length))
        
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label
        
        # Prepare inputs and outputs
        inputs = {
            'input_image': images,
            'labels': padded_labels,
            'input_length': input_length,
            'label_length': label_length
        }
        
        outputs = {'ctc': np.zeros((len(images), 1))}  # Dummy output for CTC loss
        
        return inputs, outputs
    
    def get_train_generator(self):
        """Get generator for training data.
        
        Returns:
            Generator for training data
        """
        while True:
            # Shuffle training data
            indices = np.arange(len(self.train_images))
            np.random.shuffle(indices)
            
            # Iterate over batches
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                
                # Get batch data
                batch_images = self.train_images[batch_indices]
                batch_texts = [self.train_texts[j] for j in batch_indices]
                
                # Generate CTC inputs
                inputs, outputs = self._generate_ctc_inputs(batch_images, batch_texts)
                
                yield inputs, outputs
    
    def get_val_generator(self):
        """Get generator for validation data.
        
        Returns:
            Generator for validation data
        """
        while True:
            # Iterate over batches
            for i in range(0, len(self.val_images), self.batch_size):
                # Get batch data
                batch_images = self.val_images[i:i+self.batch_size]
                batch_texts = self.val_texts[i:i+self.batch_size]
                
                # Generate CTC inputs
                inputs, outputs = self._generate_ctc_inputs(batch_images, batch_texts)
                
                yield inputs, outputs
    
    def get_prediction_generator(self, images):
        """Get generator for prediction.
        
        Args:
            images: Input images
            
        Returns:
            Generator for prediction
        """
        # Iterate over batches
        for i in range(0, len(images), self.batch_size):
            # Get batch data
            batch_images = images[i:i+self.batch_size]
            
            yield batch_images