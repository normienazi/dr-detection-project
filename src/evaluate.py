import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator


MODEL_PATH = "models/best_binary_dr_cnn.keras"
BINARY_DATA_DIR = "dataset/binary_data"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    val_generator = datagen.flow_from_directory(
        BINARY_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    print("Class indices:", val_generator.class_indices)

    predictions = model.predict(val_generator)
    y_pred = (predictions >= 0.5).astype(int).reshape(-1)
    y_true = val_generator.classes

    class_names = list(val_generator.class_indices.keys())

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    print("\nClassification Report:")
    print(report)

    os.makedirs("results", exist_ok=True)

    with open("results/evaluation_report.txt", "w") as file:
        file.write("Classification Report - Binary DR Detection\n")
        file.write("=" * 45)
        file.write("\n\n")
        file.write(report)

    print("Evaluation report saved at results/evaluation_report.txt")

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot()
    plt.title("Confusion Matrix - Binary DR Detection")

    plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Confusion matrix saved at results/confusion_matrix.png")


if __name__ == "__main__":
    main()