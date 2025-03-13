from flask import Flask
import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2
import os
app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret"

# Load model
model = load_model("model.h5")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adjust size if needed
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize
    return img

def generate_gradcam(image_path, model, layer_name="conv2d"):
    img = preprocess_image(image_path)

    # Convert image to tensor
    img_tensor = tf.convert_to_tensor(img)

    # Get the model's last convolutional layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Compute gradients using GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply conv layer activations with pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Load original image
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (224, 224))

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap.numpy(), (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # Save the Grad-CAM output
    gradcam_path = os.path.join(RESULT_FOLDER, "gradcam.jpg")
    cv2.imwrite(gradcam_path, superimposed_img)

    return gradcam_path
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            gradcam_path = generate_gradcam(filepath, model)

            return render_template("index.html", uploaded_image=filepath, gradcam_image=gradcam_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run()
