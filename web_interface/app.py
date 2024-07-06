import sys
import os
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import LambdaCallback
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.data_preprocessing import train_generator, validation_generator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

model = None
training_thread = None
training_log = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def train_model():
    global model, training_log

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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def custom_logging(epoch, logs):
        training_log.append(f"Epoch {epoch}: {logs}")

    logging_callback = LambdaCallback(on_epoch_end=custom_logging)

    model.fit(
        train_generator,
        steps_per_epoch=5,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=5,
        callbacks=[logging_callback],
        verbose=2
    )

    model.save(os.path.join(os.path.dirname(__file__), 'models', 'jewelry_classifier.h5'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global training_thread
    if training_thread is None or not training_thread.is_alive():
        training_log.clear()
        training_thread = threading.Thread(target=train_model)
        training_thread.start()
    return redirect(url_for('index'))

@app.route('/train_status')
def train_status():
    global training_log
    return jsonify(training_log)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = tf.keras.preprocessing.image.load_img(filepath, target_size=(32, 32))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        class_index = np.argmax(prediction, axis=1)[0]
        class_labels = ['BRACELET', 'EARRINGS', 'NECKLACE', 'RINGS', 'WRISTWATCH']
        result = class_labels[class_index]
        return jsonify({'prediction': result})
    return redirect(url_for('index'))

if __name__ == '__main__':
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'jewelry_classifier.h5')
    if os.path.exists(model_path):
        model = load_model(model_path)
    app.run(debug=True)