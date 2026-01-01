# Introduction to Transformers Architecture
# Visualize the architecture of a Transformer model and set up an environment for working with Transformers using PyTorch and/or TensorFlow

# For plot_model to work, we must install pydot



# libraries
from tensorflow import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, LayerNormalization, Add, MultiHeadAttention


# Define a simplied Transformer encoder block
def transformer_encoder(input_dim, num_heads, ff_dim):
  inputs = Input(shape=(None, input_dim))
  # Multi-Head Self Attention
  attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(inputs, inputs)
  attention_output = Add()([inputs, attention_output])   # create residual connection
  attention_output = LayerNormalization()(attention_output)
  
  # Feed-Forward Neural Network
  ff_output = Dense(ff_dim, activation='relu')(attention_output)
  ff_output = Dense(input_dim)(ff_output)
  outputs = Add()([attention_output, ff_output])
  outputs = LayerNormalization()(outputs)
  return Model(inputs, outputs)


# Create and visualize sample Transformer encoder block
encoder_block = transformer_encoder(input_dim=64, num_heads=8, ff_dim=128)
plot_model(encoder_block, show_shapes=True, to_file='transformer_encoder.png')



# Set up environment for Transformer in PyTorch
# Must install torch, torchvision, transformers