from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import base64
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = tf.keras.models.load_model('staff_mobilenet_v2_model.h5')

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Convert base64 image data to PIL image
        image_data = image_data.split(',')[1]  # Remove the data URL scheme
        image_bytes = io.BytesIO(base64.b64decode(image_data))
        image = Image.open(image_bytes)
        
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        
        predicted_class = list(class_names.keys())[predicted_class_index]
        probability = np.max(predictions)
        
        result = f"{predicted_class} with probability {probability:.2f}"
        
        return jsonify({'result': result, 'probability': probability})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5001)
