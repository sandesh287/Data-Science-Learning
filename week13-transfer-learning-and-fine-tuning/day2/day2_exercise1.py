# Transfer Learning in Computer Vision
# 1. Load a pre-trained ResNet or VGG model and fine-tue it for a new image classification task (eg. classifying animals or plants).
# 2. Experiment with freezing and unfreezing layers and observe the impact on performance.

# Using Tensorflow



# libraries
import tensorflow as tf
from keras.applications import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam


# Load pre-trained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Freeze all layers in base model
for layer in base_model.layers:
  layer.trainable = False


# Add custom classification head
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(5, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=output)


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# Data preparation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
  'PATH_TO_TRAIN_DATASET',    # must have datasets inside this folder to train the datsets
  target_size=(224, 224),
  batch_size=32,
  class_mode='categorical',
  subset='training'
)

val_data = datagen.flow_from_directory(
  'PATH_TO_TEST_DATASET',    # must have datasets inside this folder to test the datsets
  target_size=(224, 224),
  batch_size=32,
  class_mode='categorical',
  subset='validation'
)


# Train the model
history = model.fit(
  train_data,
  validation_data=val_data,
  epochs=10,
  steps_per_epoch=len(train_data),
  validation_steps=len(val_data)
)


# Unfreeze the last 5 layers
for layer in base_model.layers[-5:]:
  layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_data)