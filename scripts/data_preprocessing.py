import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_data_dir = os.path.join(base_dir, 'data', 'train')
validation_data_dir = os.path.join(base_dir, 'data', 'val')

# Image data generators with more augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Data generators with a smaller batch size
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(32, 32),
    batch_size=16,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(32, 32),
    batch_size=16,
    class_mode='categorical'
)
