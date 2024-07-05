import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from scripts.image_search import predict_image

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'jewelry_classifier.h5')
model = tf.keras.models.load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    # This endpoint will not retrain the model, but it's kept for potential future use
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        prediction = predict_image(file_path)
        return render_template('index.html', prediction=prediction)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
