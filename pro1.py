import os
import numpy as np
import cv2
from flask import Flask, request, render_template_string
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from scipy.spatial.distance import cosine
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load model once at startup for efficiency
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_image_similarity(feature_vector1, feature_vector2):
    return 1 - cosine(feature_vector1, feature_vector2)

def extract_image_features(img):
    img_resized = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img_resized)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array_expanded_dims)
    features = model.predict(preprocessed_img)
    return features.flatten()

def find_similar_images(target_image, image_files, threshold):
    target_features = extract_image_features(target_image)
    similar_images = []

    for image_file in image_files:
        if not allowed_file(image_file.filename):
            continue  # Skip non-image files

        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        image_features = extract_image_features(img)
        similarity = calculate_image_similarity(target_features, image_features)
        if similarity >= threshold:
            buffered = BytesIO()
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_pil.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            similar_images.append(img_str)

    return similar_images

@app.route('/')
def index():
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Similar Image Finder</title>
        <style>
            body, html {
                margin: 0;
                padding: 0;
                height: 100%;
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #A6C4B5;
                flex-direction: column;
            }
            .upload label {
                font-size: 18px;
                color: white;
                display: block;
                margin-bottom: 10px;
            }
            .upload button {
                background-color: #ffffff8f;
                padding: 10px;
                border: none;
                cursor: pointer;
            }
            #result {
                margin-top: 20px;
                text-align: center;
            }
            .image-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 10px;
                margin-top: 10px;
            }
            .image-container img {
                max-width: 200px;
                max-height: 200px;
                border: 2px solid #ccc;
                border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <h2>Upload Images to Find Similarity</h2>
        <form id="upload-form" action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="target_image" accept="image/*" required><br><br>
            <input type="file" name="image_files" accept="image/*" multiple required><br><br>
            <input type="number" name="threshold" step="0.01" min="0" max="1" placeholder="Enter threshold (0-1)" required><br><br>
            <button type="submit">Search</button>
        </form>

        <div id="loading-spinner" style="display: none;">
            <p>Processing images...</p>
            <img src="https://i.gifer.com/4V0b.gif" alt="Loading..." width="100">
        </div>

        <div id="result"></div>

        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
            $(document).ready(function () {
                $('#upload-form').on('submit', function (event) {
                    event.preventDefault();

                    $('#loading-spinner').show();
                    $('#result').html('');

                    var formData = new FormData(this);
                    $.ajax({
                        url: '/upload',
                        type: 'POST',
                        data: formData,
                        processData: false,
                        contentType: false,
                        success: function (response) {
                            $('#loading-spinner').hide();
                            $('#result').html(response);
                        },
                        error: function (error) {
                            $('#loading-spinner').hide();
                            $('#result').html('<p style="color:red;">Error: ' + error.responseText + '</p>');
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_content)

@app.route('/upload', methods=['POST'])
def upload():
    if 'target_image' not in request.files or 'image_files' not in request.files:
        return "Error: No file uploaded.", 400

    target_image_file = request.files['target_image']
    image_files = request.files.getlist('image_files')
    threshold = float(request.form['threshold'])

    if not allowed_file(target_image_file.filename):
        return "Error: Only image files are allowed.", 400

    target_image = cv2.imdecode(np.frombuffer(target_image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if target_image is None:
        return "Error: Target image could not be loaded.", 400

    valid_images = [f for f in image_files if allowed_file(f.filename)]
    
    similar_images = find_similar_images(target_image, valid_images, threshold)

    if similar_images:
        result = "Similar images found:<br><div class='image-container'>" + "".join(
            [f"<img src='data:image/jpeg;base64,{img_str}' alt='Similar Image'>" for img_str in similar_images]
        ) + "</div>"
    else:
        result = "No similar images found."

    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Render default port is 10000
