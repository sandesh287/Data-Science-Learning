# Fine-Tuning Techniques in Computer Vision
# Apply data augmentation to a dataset and train a fine-tuned model. Experiment with hyperparameters to observe their impact on performance.

# Using TensorFlow



# libraries
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam


# Load pre-tained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Freeze all layers
for layer in base_model.layers:
  layer.trainable=False


# Add classification head
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)


# Define Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
  rescale=1./255,
  rotation_range=20,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  validation_split=0.2
)


# Create train and validation data
train_data = datagen.flow_from_directory(
  'PATH_TO_DATASET',    # replace with your train dataset path
  target_size=(224,224),
  batch_size=32,
  class_mode='categorical',
  subset='training'
)

val_data = datagen.flow_from_directory(
  'PATH_TO_DATASET',   # replace with your test dataset path
  target_size=(224,224),
  batch_size=32,
  class_mode='categorical',
  subset='validation'
)


# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(
  train_data, validation_data=val_data, epochs=10, steps_per_epoch=len(train_data), validation_steps=len(val_data)
)


# Note: So, once you have trained the model, next step you can tune the hyperparameters. You can adjust learning rate, batch sizes, test different optimizers