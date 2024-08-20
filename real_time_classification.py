from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import json
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('staff_mobilenet_v2_model.h5')

# Load class names
with open('class_names.json') as f:
    class_names = json.load(f)

def preprocess_image(image):
    # Convert image to numpy array
    img = Image.open(io.BytesIO(image)).resize((224, 224))  # Adjust size if needed
    img_array = np.array(img) / 255.0  # Normalize if required
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        image = file.read()
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        class_name = list(class_names.keys())[predicted_class]
        details = class_names[class_name]

        return jsonify({
            'class_name': class_name,
            'drink_preference': details['drink_preference'],
            'dietary_restrictions': details['dietary_restrictions']
        })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)