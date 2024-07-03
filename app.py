import os
from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Define the absolute path to your model file
model_path = os.path.join(os.path.dirname(__file__), 'models', 'skin_cancer_model.h5')

# Load the model
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading the model:", e)

# Function to preprocess uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Ensure the size matches your model's expected input
    img = np.asarray(img) / 255.0  # Normalize the image
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file part")
        
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty part without filename
        if file.filename == '':
            return render_template('index.html', prediction="No selected file")

        # Process the file and make predictions
        try:
            img = preprocess_image(file)
            img = np.expand_dims(img, axis=0)  # Expand dimensions for prediction
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)

            # Map predicted class index to class name
            class_names = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions',
                           'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma']
            class_name = class_names[predicted_class]

            return render_template('index.html', prediction=class_name)

        except Exception as e:
            print("Error processing image:", e)
            return render_template('index.html', prediction="Error processing image")

if __name__ == '__main__':
    app.run(debug=True)
