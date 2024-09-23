import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('my_model.h5')

# Define class labels
class_labels = ['Apple', 'Bell pepper', 'Cherry', 'Citrus', 'Corn', 'Grape', 'Peach', 'Potato', 'Strawberry', 'Tomato']

# Set a confidence threshold
confidence_threshold = 0.6

# Path for saving uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1] range
    return img_array

def predict_image_class(model, image_path, class_labels, confidence_threshold):
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)
    confidence = predictions[0][predicted_class_index[0]]

    if confidence < confidence_threshold:
        return "Unknown"
    else:
        return class_labels[predicted_class_index[0]]

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_label = predict_image_class(model, filepath, class_labels, confidence_threshold)

        return render_template('result.html', predicted_label=predicted_label)

if __name__ == "__main__":
    app.run(debug=True)
