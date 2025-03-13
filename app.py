from flask import Flask
import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret"

# Load model
model = load_model("model.h5")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            img = load_img(filepath, target_size=(224, 224))  # Sesuaikan ukuran input model
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            print(prediction)
            result = np.argmax(prediction)  # Sesuaikan output model

            return f"Prediksi: {result}"

    return render_template("index.html")
if __name__ == '__main__':
    app.run()
