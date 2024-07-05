import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from scripts.data_preprocessing import train_generator, validation_generator

# Define the model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=15,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=15
)

# Ensure the models directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)

# Save the model
model.save(os.path.join(os.path.dirname(__file__), '..', 'models', 'jewelry_classifier.h5'))

# Evaluate the model
score = model.evaluate(validation_generator, steps=32)
print("Evaluation: ===============================================================")
print("Loss:", score[0])
print("Accuracy:", score[1])
