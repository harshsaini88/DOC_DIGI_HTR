"""Configuration parameters for the handwritten OCR system."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for dir_path in [DATA_DIR, SAVE_DIR, LOGS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Model parameters
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 800
CHANNELS = 1

# CNN parameters
CNN_FILTERS = [64, 128, 256]
POOL_SIZE = (2, 2)

# RNN parameters
RNN_UNITS = 256
ATTENTION_SIZE = 128

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Character set
CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?()[]{}'\"-+=/\\@#$%^&*<>~` "
NUM_CLASSES = len(CHAR_VECTOR) + 1  # +1 for blank character (CTC requirement)

# Quantization parameters
QUANTIZATION_BITS = 8
PRUNING_SCHEDULE = {
    'initial_sparsity': 0.0,
    'final_sparsity': 0.5,
    'begin_step': 0,
    'end_step': 5000,
    'frequency': 100
}

# Language model parameters
LANGUAGE_MODEL_WEIGHT = 0.3
CONFIDENCE_THRESHOLD = 0.7
MAX_EDIT_DISTANCE = 2
LANGUAGE_MODEL_DICT_PATH = os.path.join(DATA_DIR, "dictionary.txt")
