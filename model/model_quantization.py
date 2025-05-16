"""Model quantization utilities to reduce model size and improve inference speed."""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from model.config import QUANTIZATION_BITS, PRUNING_SCHEDULE


def apply_quantization_to_model(model):
    """Apply quantization to model to reduce size.
    
    Args:
        model: Keras model to quantize
        
    Returns:
        Quantized Keras model
    """
    # Apply quantization to all applicable layers
    quantize_model = tfmot.quantization.keras.quantize_model
    
    # Create quantization config
    quantize_config = tfmot.quantization.keras.QuantizeConfig(
        {'quantize_weights': True, 'quantize_activations': True},
        {'quantize_weights': True, 'quantize_activations': True},
        {
            'weight_quantizer': tfmot.quantization.keras.quantizers.LastValueQuantizer(
                num_bits=QUANTIZATION_BITS, symmetric=True, narrow_range=False),
            'activation_quantizer': tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
                num_bits=QUANTIZATION_BITS, symmetric=False, narrow_range=False),
        },
        {'quantize_output': True}
    )
    
    # Quantize model
    q_aware_model = quantize_model(model)
    
    # Compile the model
    q_aware_model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=model.metrics
    )
    
    return q_aware_model


def apply_pruning_to_model(model):
    """Apply pruning to model to reduce size.
    
    Args:
        model: Keras model to prune
        
    Returns:
        Pruned Keras model
    """
    # Apply pruning to all applicable layers
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    
    # Create pruning schedule
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=PRUNING_SCHEDULE['initial_sparsity'],
            final_sparsity=PRUNING_SCHEDULE['final_sparsity'],
            begin_step=PRUNING_SCHEDULE['begin_step'],
            end_step=PRUNING_SCHEDULE['end_step'],
            frequency=PRUNING_SCHEDULE['frequency']
        )
    }
    
    # Prune model
    pruned_model = prune_low_magnitude(model, **pruning_params)
    
    # Compile the model
    pruned_model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=model.metrics
    )
    
    return pruned_model


def convert_to_tflite(model, quantize=True):
    """Convert Keras model to TFLite format.
    
    Args:
        model: Keras model to convert
        quantize: Whether to apply quantization during conversion
        
    Returns:
        TFLite model as bytes
    """
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply additional optimizations
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    # Convert model
    tflite_model = converter.convert()
    
    return tflite_model


def save_tflite_model(tflite_model, filepath):
    """Save TFLite model to file.
    
    Args:
        tflite_model: TFLite model as bytes
        filepath: Path to save model to
    """
    with open(filepath, 'wb') as f:
        f.write(tflite_model)


def optimize_model_for_inference(model):
    """Apply all optimizations to model for inference.
    
    Args:
        model: Keras model to optimize
        
    Returns:
        Optimized TFLite model as bytes
    """
    # Apply pruning
    pruned_model = apply_pruning_to_model(model)
    
    # Apply quantization
    quantized_model = apply_quantization_to_model(pruned_model)
    
    # Convert to TFLite
    tflite_model = convert_to_tflite(quantized_model)
    
    return tflite_model