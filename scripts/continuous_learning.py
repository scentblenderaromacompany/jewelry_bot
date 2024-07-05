import tensorflow as tf
from data_preprocessing import train_generator, validation_generator

# Load the saved model
model = tf.keras.models.load_model('models/jewelry_classifier.h5')

# Continue training with new data
model.fit(
    train_generator,
    steps_per_epoch=15,
    epochs=10,  # Use fewer epochs for incremental learning
    validation_data=validation_generator,
    validation_steps=10
)

# Save the updated model
model.save('models/jewelry_classifier_updated.h5')
