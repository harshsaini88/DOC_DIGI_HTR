"""CNN-LSTM model with attention mechanism for handwritten text recognition."""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from model.config import *


class AttentionLayer(layers.Layer):
    """Attention mechanism layer"""
    
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
        
    def call(self, features, hidden):
        # features shape: (batch_size, max_len, features_dim)
        # hidden shape: (batch_size, hidden_size)
        
        # Expand hidden to match batch size of features
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # hidden_with_time_axis shape: (batch_size, 1, hidden_size)
        
        # Calculate attention scores
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # score shape: (batch_size, max_len, units)
        
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # attention_weights shape: (batch_size, max_len, 1)
        
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # context_vector shape: (batch_size, features_dim)
        
        return context_vector, attention_weights


def create_model(is_training=True):
    """Create CNN-LSTM model with attention mechanism."""
    
    # Input layer
    input_img = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), name='input_image')
    
    # CNN layers
    x = input_img
    for i, filters in enumerate(CNN_FILTERS):
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu', name=f'conv_{i+1}')(x)
        x = layers.BatchNormalization(name=f'batchnorm_{i+1}')(x)
        x = layers.MaxPooling2D(pool_size=POOL_SIZE, name=f'pool_{i+1}')(x)
    
    # Prepare feature map for RNN
    # After 3 pooling layers of factor 2, height is reduced by factor 8
    new_height = IMAGE_HEIGHT // (2**len(CNN_FILTERS))
    new_width = IMAGE_WIDTH // (2**len(CNN_FILTERS))
    
    # Reshape to (batch_size, time_steps, features)
    x = layers.Reshape((new_width, new_height * CNN_FILTERS[-1]))(x)
    
    # Bidirectional LSTM layers
    lstm1 = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True))(x)
    lstm1 = layers.Dropout(0.2)(lstm1)
    lstm2 = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True))(lstm1)
    lstm2 = layers.Dropout(0.2)(lstm2)
    
    # Attention mechanism
    attention = AttentionLayer(ATTENTION_SIZE)
    
    # Get initial hidden state from last LSTM layer
    last_hidden = layers.Lambda(lambda x: x[:, -1, :])(lstm2)
    context_vector, attention_weights = attention(lstm2, last_hidden)
    
    # Concatenate context vector with last hidden state
    decoder_input = layers.Concatenate()([context_vector, last_hidden])
    
    # Dense output layers
    dense1 = layers.Dense(RNN_UNITS, activation='relu')(decoder_input)
    output = layers.Dense(NUM_CLASSES, activation='softmax')(dense1)
    
    # Create model
    model = Model(inputs=input_img, outputs=output)
    
    # Compile model if in training mode
    if is_training:
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model


def create_ctc_model():
    """Create CNN-LSTM model with CTC loss for sequence prediction."""
    
    # Input layer
    input_img = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), name='input_image')
    labels = layers.Input(name='labels', shape=[None], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    
    # CNN layers
    x = input_img
    for i, filters in enumerate(CNN_FILTERS):
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu', name=f'conv_{i+1}')(x)
        x = layers.BatchNormalization(name=f'batchnorm_{i+1}')(x)
        x = layers.MaxPooling2D(pool_size=POOL_SIZE, name=f'pool_{i+1}')(x)
    
    # Prepare feature map for RNN
    new_height = IMAGE_HEIGHT // (2**len(CNN_FILTERS))
    new_width = IMAGE_WIDTH // (2**len(CNN_FILTERS))
    
    # Reshape to (batch_size, time_steps, features)
    x = layers.Reshape((new_width, new_height * CNN_FILTERS[-1]))(x)
    
    # Bidirectional LSTM layers
    lstm1 = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True))(x)
    lstm1 = layers.Dropout(0.2)(lstm1)
    lstm2 = layers.Bidirectional(layers.LSTM(RNN_UNITS, return_sequences=True))(lstm1)
    
    # Output layer (time_distributed)
    y_pred = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(lstm2)
    
    # CTC loss function
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # y_pred = y_pred[:, 2:, :]  # Remove first two outputs of RNN
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length]
    )
    
    # Create model
    model = Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)
    
    # Compile model with dummy loss since real loss is computed in Lambda layer
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss={'ctc': lambda y_true, y_pred: y_pred})
    
    # Create prediction model that outputs character probabilities
    pred_model = Model(inputs=input_img, outputs=y_pred)
    
    return model, pred_model