# Autoencoders
# Autoencoders are neural networks used for unsupervised learning, especially for dimensionality reduction and feature extraction. They work by encoding input data into a compressed (latent) representation and then reconstructing the original input from this representation. Autoencoders are useful for tasks like denoising, anomaly detection, and pretraining for other neural networks.




# libraries
from keras.models import Model   # to create and combine the input and output layers to define the entire autoencoder model
from keras.layers import Input, Dense   # Input is used to specify the input layers of the model, where you define the shape of the input data, and dense is a fully connected neural network layer commonly used in feed-forward networks. Here it is used in both the encoder and decoder parts of the autoencoder.
import numpy as np



# Sample data (eg. points in 5-dimensional array)
X = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [5, 6, 7, 8, 9], [5, 7, 8, 9, 10], [8, 9, 10, 11, 12]])   # defines the feature data X as a numpy array where each sublist represents a data point in a 5D space with five values. Shape of X is (6, 5), where there are 6 data points with 5 features each.



# Define the autoencoder model
input_dim = X.shape[1]   # sets the variable input_dim (input dimensions which is 5 in this case) to the number of features in the dataset X. It returns the number of columns which are features in X, which will be input dimension for autoencoders.

encoding_dim = 2   # compressing to 2 dimension. This specifies the number of dimensions to which we want to compress the input data. Here, setting 2 means compressing the 5D input data into 2D representation.



# Encoder
input_layer = Input(shape=(input_dim,))   # This defines the input layer of the autoencoder with the shape of input dimension where input_dim=5. This specifies that each input data points has five features.

encoded = Dense(encoding_dim, activation='relu')(input_layer)   # This defines the encoder layer, which is Dense fully connected layer with "encoding_dim=2" neurons. "activation='relu'", which introduces non-linearity to the model. The encoder reduces the input dimension from 5 to 2 by learning a compressed representation of the input data in this case.



# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)   # This defines the decoder layer, which is also dense layer with input_dim=5 neurons, to reconstruct the original input. The sigmoid activation function outputs values between 0 and 1, commonly used for reconstruction tasks. The decoder expands the compressed 2D encoding back to the original 5D space.



# Autoencoder Model
autoencoder = Model(input_layer, decoded)   # This combines the input and output layers to create a full autoencoder model. This model takes input_layer as the input and decoded as the output, meaning it will take a 5D input, compress it to 2D in the encoder, and then reconstruct it back to a 5D in the decoder.



# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')   # This compiles the autoencoder model specifying "optimizer='adam'", which uses Adam optimizer and adaptive learning rate optimization algorithm, commonly used for training deep learning models and "loss='mse'", specifies MSE (Mean Square Error) as the loss function, which calculates the average of squared differences between reconstructed and original input values. Lower loss indicates better reconstruction accuracy.



# Train the model
autoencoder.fit(X, X, epochs=100, batch_size=2, verbose=0)   # This trains te autoencoder model on the dataset X, where X is used as both input and target. The model tries to reconstruct its own input. "epochs=100", which specifies the model will iterate over the entire dataset 100 times, "batch_size=2", specifies the model processes two samples at a time before updating its weights, which helps stabilize training, "verbose=0", suppresses the output of the training progress for cleaner output



# Get the encoded (compressed) representation
encoder = Model(input_layer, encoded)   # This defines an encoder model using only the encoding part of the original autoencoder. This model takes input layer as input and outputs encoded, allowing it to generate compressed representations.

X_compressed = encoder.predict(X)   # It passes input data X through the encoder model to obtain the compressed 2D representation. The X_compressed now contains the 2D representation of the original 5D input data.


print(f'Compressed Representation: \n {X_compressed}')   # prints the compressed 2D representation of the input data, allowing us to view the lower dimensional encoding learned by the autoencoder






# This code trains an autoencoder to reduce 5D data to 2D, which can then be used for tasks like visualization or feature extraction. The autoencoder learns to reconstruct the original data while compressing it into more compact, 2D form in the middle layer. The encoder part of the model can be used separately to generate a compressed representation of the new data.