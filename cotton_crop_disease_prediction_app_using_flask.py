from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model once when the app starts
model = load_model(r'cotton_disease_detection.h5')

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Adjust target size according to your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale if your model was trained on normalized images
    return img_array

@app.route('/')
def signup():
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('main'))
    return render_template('login.html')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # Handle image upload and prediction logic
        uploaded_file = request.files['image']
        if uploaded_file.filename != '':
            img_path = 'static/uploads/' + uploaded_file.filename
            uploaded_file.save(img_path)

            img = prepare_image(img_path)
            prediction = model.predict(img)

            # You might need to decode the prediction based on your model's output format
            prediction_result = decode_prediction(prediction)

            return render_template('main.html', prediction=prediction_result)
    return render_template('main.html')

def decode_prediction(prediction):
    # Example decoding, replace with actual labels
    labels = ['Aphids','Army_worm', 'Bacterial_Blight','Healthy','Powdery_Mildew','Target_spot']
    result = labels[np.argmax(prediction)]
    return result

if __name__ == '__main__':
    app.run(debug=True)
