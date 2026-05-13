import cv2
import matplotlib.pyplot as plt


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    return image


def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)


def apply_clahe(image):
    # Convert BGR image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced_l = clahe.apply(l_channel)

    # Merge enhanced L channel with original A and B channels
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    # Convert LAB back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr


def show_images(original, processed):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed_rgb)
    plt.title("CLAHE Enhanced Image")
    plt.axis("off")

    plt.show()


def main():
    image_path = "dataset/sample.png"

    original = load_image(image_path)
    resized = resize_image(original)
    enhanced = apply_clahe(resized)

    cv2.imwrite("results/enhanced_sample.png", enhanced)

    show_images(resized, enhanced)

    print("Preprocessed image saved at: results/enhanced_sample.png")

if __name__ == "__main__":
    main()