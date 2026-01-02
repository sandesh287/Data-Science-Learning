# Transfer Learning Project: Fine-Tuneing for a Custom Task
# Fine-tune a pre-trained model (ResNet for computer vision) to solve a custom task
# Evaluate and document results against a baseline

# Computer Vision: (Dataset: Kaggle cats vs dogs)
# NLP: (Dataset: Amazon review sentiment analysis)

# Cmputer Vision Project



# libraries
import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D


# Data augmentation and preprocessing
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


train_data = datagen.flow_from_directory(
  'PATH_TO_DATASET',
  target_size=(224, 224),
  batch_size=32,
  class_mode='binary',
  subset='training'
)

val_data = datagen.flow_from_directory(
  'PATH_TO_DATASET',
  target_size=(224, 224),
  batch_size=32,
  class_mode='binary',
  subset='validation'
)


# Load pre-trained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Freeze model
for layer in base_model.layers:
  layer.trainable=False


# Add custom head
x = GlobalAveragePooling2D()(base_model.output)

output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)


# Compile and Train model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
  train_data,
  validation_data=val_data,
  epochs=10
)


# Evaluate and document result
val_loss, val_accuracy = model.evaluate(val_data)

print(f'Validation Accuracy: {val_accuracy}')