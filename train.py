"""Model training script for handwritten OCR."""

import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from model.cnn_lstm_model import create_ctc_model
from model.model_quantization import apply_quantization_to_model, save_tflite_model
from utils.data_loader import DataLoader
from model.config import BATCH_SIZE, EPOCHS, SAVE_DIR, LOGS_DIR


def train_model(data_dir, model_name, batch_size=BATCH_SIZE, epochs=EPOCHS, quantize=False):
    """Train handwritten OCR model.
    
    Args:
        data_dir: Directory containing training data
        model_name: Name of model to save
        batch_size: Batch size
        epochs: Number of epochs
        quantize: Whether to quantize model after training
    """
    # Create data loader
    data_loader = DataLoader(data_dir, batch_size=batch_size)
    
    # Create model
    model, _ = create_ctc_model()
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, f"{model_name}_best.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1
        ),
        TensorBoard(
            log_dir=LOGS_DIR,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
    ]
    
    # Train model
    model.fit(
        data_loader.get_train_generator(),
        steps_per_epoch=len(data_loader.train_images) // batch_size,
        epochs=epochs,
        validation_data=data_loader.get_val_generator(),
        validation_steps=len(data_loader.val_images) // batch_size,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(os.path.join(SAVE_DIR, f"{model_name}_final.h5"))
    
    # If quantization is requested, quantize model
    if quantize:
        # Apply quantization
        quantized_model = apply_quantization_to_model(model)
        
        # Save quantized model
        quantized_model.save(os.path.join(SAVE_DIR, f"{model_name}_quantized.h5"))
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TFLite model
        save_tflite_model(tflite_model, os.path.join(SAVE_DIR, f"{model_name}.tflite"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train handwritten OCR model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--model_name', type=str, default='ocr_model', help='Name of model to save')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--quantize', action='store_true', help='Whether to quantize model after training')
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.model_name, args.batch_size, args.epochs, args.quantize)
