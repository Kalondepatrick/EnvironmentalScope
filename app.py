import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

predicted_class = ''
class_probability = 0

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Load the trained model
model = load_model('newmodel.h5')

# Define class names
class_names = ["apple rot", "healthy"]

# Function to preprocess and make predictions
def predict_image(image_path):

    # load the image using pillow
   
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image = np.array(image)
    image = image / 255.0

    # Make a prediction
    prediction = model.predict(np.expand_dims(image, axis=0))

    # Get the class with the highest probability
    class_index = np.argmax(prediction)
    class_probability = prediction[0][class_index]
    predicted_class = class_names[class_index]

    return predicted_class, class_probability


@app.route('/home')
def home():
    return render_template('index.html')


# Route for image upload
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded image
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Make predictions
            predicted_class, class_probability = predict_image(filename)

            class_probability = class_probability*100

            class_probability = int(class_probability)

            # Render the result template with predictions
            return render_template('upload.html', image_path=filename, predicted_class=predicted_class, class_probability=class_probability)

    return render_template('upload.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)