import os

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.secret_key = "secret"
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = load_model("model.h5")

CLASS_NAMES = ["Normal", "Glaucoma"]
EXPLANATIONS = {
    "Normal": "Hasil analisis menunjukkan bahwa mata Anda berada dalam kondisi normal. Tidak ditemukan indikasi glaukoma dalam gambar yang diberikan.",
    "Glaucoma": "Hasil analisis menunjukkan kemungkinan adanya glaukoma. Glaukoma adalah penyakit mata yang dapat menyebabkan kehilangan penglihatan jika tidak ditangani. Silakan konsultasikan dengan dokter spesialis mata untuk pemeriksaan lebih lanjut."
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img


def generate_gradcam(image_path, model, layer_name="conv1_conv", alpha=0.6):
    img = preprocess_image(image_path)
    img_tensor = tf.convert_to_tensor(img)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0]).numpy()
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (224, 224))

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Convert to 8-bit format
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Add color

    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)

    gradcam_filename = f"gradcam_{os.path.basename(image_path)}"
    gradcam_path = os.path.join(RESULT_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_path, overlay)

    return gradcam_filename, predictions.numpy(), class_idx


@app.route("/", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        gradcam_filename, predictions, class_idx = generate_gradcam(filepath, model)
        predicted_class = CLASS_NAMES[class_idx]
        confidence = float(round(100 * np.max(predictions), 2))
        explanation = EXPLANATIONS[predicted_class]
        return jsonify({
            "uploaded_image": filename,
            "gradcam_image": gradcam_filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "explanation": explanation,
        })

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/gradcam/<filename>', methods=['GET'])
def get_gradcam_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)