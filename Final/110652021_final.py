import os
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

# Check if CUDA is available
if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    # Use GPU
    print("CUDA is available. Using GPU.")
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    K.set_session(session)
else:
    # Use CPU
    print("CUDA is not available. Using CPU.")

img_width, img_height = 224, 224

train_data_dir = 'data/train'
nb_train_samples = 9788
batch_size = 8
nb_epochs = 30
validation_split = 0.9  # 90% of the data will be used for validation

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split)  # Set validation split

# Flow from directory with validation split
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')  # Specify subset as 'training'

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')  # Specify subset as 'validation'

# Import ResNet50 with pre-trained weights
if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    inception_base = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
else:
    inception_base = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3), backend=K.backend(), layers=K.layers, models=K.models, utils=K.utils)

# Add layers and create the full network
x = inception_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(200, activation='softmax')(x)
inception_transfer = Model(inputs=inception_base.input, outputs=predictions)

# Compile the model
inception_transfer.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                          metrics=['accuracy'])

# Display available devices
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
checkpoint_filepath='/tmp/checkpoint'
# Set up callbacks
callbacks = [ModelCheckpoint('imagenet', monitor='val_accuracy', save_best_only=True)]

# Train the model
history = inception_transfer.fit_generator(
    train_generator,
    callbacks=callbacks,
    steps_per_epoch=nb_train_samples * (1 - validation_split) // batch_size,  # Adjust as needed
    epochs=nb_epochs,
    validation_data=validation_generator,
    validation_steps=nb_train_samples * validation_split // batch_size)  # Adjust as needed

print('Training Completed!')

# Plot accuracy history
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
