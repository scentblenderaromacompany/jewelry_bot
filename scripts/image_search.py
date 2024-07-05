import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'jewelry_classifier.h5')
model = tf.keras.models.load_model(model_path)

# Function to preprocess and predict the class of a single image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_indices = {'BRACELET': 0, 'EARRINGS': 1, 'NECKLACE': 2, 'RINGS': 3, 'WRISTWATCH': 4}
    class_names = list(class_indices.keys())
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# Example usage
if __name__ == '__main__':
    img_path = 'path/to/your/image.jpg'
    print(f'The jewelry in the image is predicted to be: {predict_image(img_path)}')
