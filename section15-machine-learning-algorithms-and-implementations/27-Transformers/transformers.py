# Transformers
# Transformers are deep learning architectures designed for handling sequential data without relying on recurrence, which is commonly used in RNNs. Instead, Tranformers use a mechanism called self-attention to process all tokens in the sequence simultaneously, capturing dependencies between tokens regardless of their distance in the sequence. Transformers have become the foundation of many NLP tasks and models, including BERT and GPT.




# libraries
import tensorflow as tf   # main library for deep learning, providing tools to define and train neural networks
from keras import layers, models   # used to define the layers and structure of the neural network
from keras.datasets import imdb   # a dataset of movie reviews (sentiment analysis), with each review labeled as positive or negative sentiment
from keras.preprocessing import sequence   # A utility for preprocessing sequences specifically to pad or truncate sequences to the same length



# Load and preprocess the IMDB dataset
max_features = 10000   # Vocabulary size, sets the vocabulary size to the top 10,000 most frequent words in the dataset
max_len = 200   # limit reviews to 500 words, reviews longer than 200 words will be truncated, while shorter ones will be padded


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)   # loads the IMDb dataset, limiting it to max features most frequent words. "(X_train, y_train)" are the training data and labels. Each review is represented as a sequence of integers, which are word indices, and "(X_test, y_test)" are test data and labels.
X_train = sequence.pad_sequences(X_train, maxlen=max_len)   # This pads or truncates each review in X_train to ensure all the reviews have exactly max length words
X_test = sequence.pad_sequences(X_test, maxlen=max_len)   # This step ensures that each review is the same length, which is necessary for batch processing in neural networks.



# Define Transformer block
# This defines a custom layer representing a transformer block.
class TransformerBlock(layers.Layer):
  # This is the constructor method where "self", is its own object. "embed_dim", embedding dimension for each word vector; "num_heads", number of attention heads in the multi-head attention layer; "ff_dim", number of units in the feedforward layer; "rate", dropout rate to prevent overfitting.
  def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
    super(TransformerBlock, self).__init__()   # calling super method of initializer
    self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)   # multi-head self-attention mechanism to learn contextual relationship in the input data
    # This defines a feed forward neural network ffn layer
    self.ffn = models.Sequential([
      layers.Dense(ff_dim, activation='relu'),
      layers.Dense(embed_dim),
    ])
    
    # These are layer normalization layers to stabilize and improve model performance.
    self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    
    # Dropout layers to randomly drop units during training, reducing overfitting.
    self.dropout1 = layers.Dropout(rate)
    self.dropout2 = layers.Dropout(rate)
  
  
  # This defines the forward pass for the transformer block.
  def call(self, inputs, training=None):   # ensure "training" is optional
    attn_output = self.att(inputs, inputs)   # applies self-attention, allowing model to consider each word in context of others
    attn_output = self.dropout1(attn_output, training=training)   # applies dropout during training to the attention output
    out1 = self.layernorm1(inputs + attn_output)   # adds the attention output to the original input and normalizes
    ffn_output = self.ffn(out1)   # passes the normalized output through the feedforward network
    ffn_output = self.dropout2(ffn_output, training=training)   # applies dropout to the feedforward network output
    return self.layernorm2(out1 + ffn_output)   # adds feedforward network output to out1 & applies normalization for final output



# Define the model with an embedding layer, transformer block, and output layer
embed_dim = 32   # embedding dimension of each word vector
num_heads = 2   # multi-head attention layer
ff_dim = 32   # units in the feedforward network

inputs = layers.Input(shape=(max_len,))   # input layer accepting sequence of length "max_len"
embedding_layer = layers.Embedding(input_dim=max_features, output_dim=embed_dim, input_length=max_len)   # This is embedding layer that maps word indices to dense vector of size "embed_dim"
x = embedding_layer(inputs)   # passes input sequences through the embedding layer
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)   # initializes the transformer block with specified dimensions
x = transformer_block(x, training=True)   # applies the transformer block to x, explicitly setting training=True
x = layers.GlobalAveragePooling1D()(x)   # reduces each sequence's dimension by averaging across the time axis
x = layers.Dropout(0.1)(x)   # applies dropout with 10% rate
x = layers.Dense(20, activation='relu')(x)   # fully connected layer with 20 units and ReLU activation
x = layers.Dropout(0.1)(x)   # applies dropout with 10% rate
outputs = layers.Dense(1, activation='sigmoid')(x)   # output layer with a single unit and sigmoid activation for binary classification.


model = models.Model(inputs=inputs, outputs=outputs)   # creates a Keras model object from the input and output



# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])   # This configures the model for training; Optimizer Adam uses Adam optimizer, which dynamically adjusts learning rates; Binary cross entropy loss for binary classification is used here and metrics measures Accuracy during training and evaluation.

model.fit(X_train, y_train, batch_size=64, epochs=3, validation_split=0.2)   # This will train the model on the training data; "(X_train, y_train)", the training data and labels; "batch_size=64" process 64 samples per training batch; "epochs=3", the model goes through the entire dataset three times; "validation_split=0.2", reserves 20% of the training data for validation.



# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)   # This evaluates the model on the test set. The test loss is the computed loss of the test data, and the test accuracy is the accuracy of the model on the test data.

print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')







# Implemented Transformers using Python