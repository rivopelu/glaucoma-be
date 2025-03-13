from flask import Flask, request, render_template, redirect
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.secret_key = "secret"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = load_model("model.h5")

CLASS_NAMES = ["Normal", "Glaucoma"]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img


def generate_gradcam(image_path, model, layer_name="conv1_conv"):
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
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    gradcam_path = os.path.join(RESULT_FOLDER, "gradcam.jpg")
    cv2.imwrite(gradcam_path, superimposed_img)

    return gradcam_path, predictions.numpy(), class_idx


def explain_prediction(class_idx):
    explanations = {
        0: "AI memprediksi ini sebagai **Mata Normal**, karena tidak terlihat gejala khas seperti peningkatan tekanan intraokular atau perubahan pada saraf optik.",
        1: "AI mendeteksi **Glaukoma**, yang ditandai dengan perubahan di saraf optik dan tekanan mata yang tinggi.",
    }
    return explanations.get(class_idx, "AI tidak bisa memberikan penjelasan untuk prediksi ini.")


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            gradcam_path, predictions, class_idx = generate_gradcam(filepath, model)
            predicted_class = CLASS_NAMES[class_idx]
            confidence = round(100 * np.max(predictions), 2)
            explanation = explain_prediction(class_idx)

            return render_template(
                "index.html",
                uploaded_image=filepath,
                gradcam_image=gradcam_path,
                predicted_class=predicted_class,
                confidence=confidence,
                explanation=explanation,
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)