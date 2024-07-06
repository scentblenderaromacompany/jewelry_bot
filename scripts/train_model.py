import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from sklearn.metrics import precision_score, recall_score, f1_score
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

# Callbacks for logging and monitoring
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_callback = ModelCheckpoint(filepath=os.path.join(os.path.dirname(__file__), '..', 'models', 'model-{epoch:02d}.h5'), save_freq='epoch')

def custom_logging(epoch, logs):
    print(f"Epoch {epoch}: {logs}")

logging_callback = LambdaCallback(on_epoch_end=custom_logging)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=5,  # Reduced steps per epoch for slower training
    epochs=25,
    validation_data=validation_generator,
    validation_steps=5,  # Reduced validation steps
    callbacks=[tensorboard_callback, checkpoint_callback, logging_callback],
    verbose=2  # Detailed output
)

# Save the final model
model.save(os.path.join(os.path.dirname(__file__), '..', 'models', 'jewelry_classifier.h5'))

# Detailed evaluation
y_true = np.concatenate([validation_generator.next()[1] for _ in range(validation_generator.__len__())])
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print("Evaluation: ===============================================================")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)