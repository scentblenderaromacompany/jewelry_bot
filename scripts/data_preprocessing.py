import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_dir = os.path.join(base_dir, 'data', 'train')
val_dir = os.path.join(base_dir, 'data', 'val')
test_dir = os.path.join(base_dir, 'data', 'test')

# Image data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=16,  # Adjust batch size
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(32, 32),
    batch_size=16,  # Adjust batch size
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),
    batch_size=16,  # Adjust batch size
    class_mode='categorical'
)