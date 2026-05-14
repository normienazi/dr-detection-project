import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


MODEL_PATH = "models/best_binary_dr_cnn.keras"
IMG_SIZE = (224, 224)


def build_functional_model():
    inputs = layers.Input(shape=(224, 224, 3), name="input_image")

    x = layers.Conv2D(32, (3, 3), activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(2, 2, name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(2, 2, name="pool2")(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(2, 2, name="pool3")(x)

    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def load_trained_weights():
    saved_model = tf.keras.models.load_model(MODEL_PATH)
    functional_model = build_functional_model()

    functional_model.set_weights(saved_model.get_weights())

    return functional_model


def load_and_preprocess_image(image_path):
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.resize(image_bgr, IMG_SIZE)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_array = image_rgb / 255.0
    image_batch = np.expand_dims(image_array, axis=0).astype(np.float32)

    return image_rgb, image_batch


def make_gradcam_heatmap(image_batch, model, last_conv_layer_name="conv3"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    image_tensor = tf.convert_to_tensor(image_batch)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        class_score = predictions[:, 0]

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None. Grad-CAM could not be computed.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if max_value != 0:
        heatmap = heatmap / max_value

    return heatmap.numpy()


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)

    return overlay


def predict_label(prediction_score):
    if prediction_score >= 0.5:
        label = "No Diabetic Retinopathy Detected"
        confidence = prediction_score
    else:
        label = "Diabetic Retinopathy Detected"
        confidence = 1 - prediction_score

    return label, confidence


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('python src/gradcam.py "path_to_image"')
        return

    image_path = sys.argv[1]

    model = load_trained_weights()

    original_image, image_batch = load_and_preprocess_image(image_path)

    prediction_score = model.predict(image_batch)[0][0]
    label, confidence = predict_label(prediction_score)

    print("Using last convolutional layer: conv3")

    heatmap = make_gradcam_heatmap(
        image_batch,
        model,
        last_conv_layer_name="conv3"
    )

    overlay = overlay_heatmap(original_image, heatmap)

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"{label}\nConfidence: {confidence * 100:.2f}%")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/gradcam_output.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Raw prediction score: {prediction_score:.4f}")
    print(f"Result: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("Grad-CAM output saved at results/gradcam_output.png")


if __name__ == "__main__":
    main()