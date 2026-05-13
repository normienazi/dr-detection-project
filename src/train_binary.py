import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


IMAGE_DIR = "dataset/gaussian_filtered_images/gaussian_filtered_images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5


def create_binary_dataset_structure():
    """
    Creates a binary dataset folder structure:
    dataset/binary_data/
        No_DR/
        DR/
    Images are not duplicated if already copied.
    """

    import shutil

    binary_dir = "dataset/binary_data"
    no_dr_dir = os.path.join(binary_dir, "No_DR")
    dr_dir = os.path.join(binary_dir, "DR")

    os.makedirs(no_dr_dir, exist_ok=True)
    os.makedirs(dr_dir, exist_ok=True)

    class_mapping = {
        "No_DR": "No_DR",
        "Mild": "DR",
        "Moderate": "DR",
        "Severe": "DR",
        "Proliferate_DR": "DR"
    }

    for class_name, target_class in class_mapping.items():
        source_folder = os.path.join(IMAGE_DIR, class_name)
        target_folder = os.path.join(binary_dir, target_class)

        for filename in os.listdir(source_folder):
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            if not os.path.exists(target_path):
                shutil.copy(source_path, target_path)

    print("Binary dataset structure created successfully.")


def build_model():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model():
    binary_dir = "dataset/binary_data"

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        binary_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        binary_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

    print("Class indices:", train_generator.class_indices)

    model = build_model()

    model.summary()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/binary_dr_cnn.keras")

    print("Model saved at models/binary_dr_cnn.keras")


def main():
    create_binary_dataset_structure()
    train_model()


if __name__ == "__main__":
    main()