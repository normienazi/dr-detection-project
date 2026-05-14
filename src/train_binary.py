import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


IMAGE_DIR = "dataset/gaussian_filtered_images/gaussian_filtered_images"
BINARY_DATA_DIR = "dataset/binary_data"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20


def create_binary_dataset_structure():
    """
    Creates a binary dataset folder structure:
    dataset/binary_data/
        No_DR/
        DR/

    Mapping:
        No_DR -> No_DR
        Mild, Moderate, Severe, Proliferate_DR -> DR
    """

    no_dr_dir = os.path.join(BINARY_DATA_DIR, "No_DR")
    dr_dir = os.path.join(BINARY_DATA_DIR, "DR")

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
        target_folder = os.path.join(BINARY_DATA_DIR, target_class)

        if not os.path.exists(source_folder):
            print(f"Warning: source folder not found: {source_folder}")
            continue

        for filename in os.listdir(source_folder):
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            if os.path.isfile(source_path) and not os.path.exists(target_path):
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
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        BINARY_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        BINARY_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

    print("Class indices:", train_generator.class_indices)

    model = build_model()
    model.summary()

    os.makedirs("models", exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "models/best_binary_dr_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )

    model.save("models/binary_dr_cnn.keras")

    print("Final model saved at models/binary_dr_cnn.keras")
    print("Best model saved at models/best_binary_dr_cnn.keras")


def main():
    create_binary_dataset_structure()
    train_model()


if __name__ == "__main__":
    main()