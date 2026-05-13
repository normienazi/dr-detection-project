import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


MODEL_PATH = "models/binary_dr_cnn.keras"
IMG_SIZE = (224, 224)


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.resize(image, IMG_SIZE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    return image_rgb, image_batch


def predict(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    original_image, processed_image = preprocess_image(image_path)

    prediction = model.predict(processed_image)[0][0]

    if prediction >= 0.5:
        label = "No Diabetic Retinopathy Detected"
        confidence = prediction
    else:
        label = "Diabetic Retinopathy Detected"
        confidence = 1 - prediction

    print(f"Image path: {image_path}")
    print(f"Raw prediction score: {prediction:.4f}")
    print(f"Result: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

    plt.imshow(original_image)
    plt.title(f"{label}\nConfidence: {confidence * 100:.2f}%")
    plt.axis("off")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('python src/predict.py "path_to_image"')
        return

    image_path = sys.argv[1]
    predict(image_path)


if __name__ == "__main__":
    main()