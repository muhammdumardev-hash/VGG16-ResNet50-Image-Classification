"""
Predict the class of a new image using the saved VGG16 and ResNet50 models.

Usage from the project folder:
    python predict.py test_images/test_cat.jpg

The saved models include their own model-specific preprocessing layers,
so this script passes resized RGB pixel values directly to each model.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "results")
CLASS_NAMES = ["cats", "dogs", "horses"]
IMAGE_SIZE = (224, 224)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path-to-new-image>")
        print("Example: python predict.py test_images/test_cat.jpg")
        raise SystemExit(1)

    image_path = sys.argv[1]
    if not os.path.isabs(image_path):
        image_path = os.path.join(PROJECT_DIR, image_path)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resize to 224x224 RGB. Model-specific preprocessing is already
    # included inside each saved model, so do not preprocess a second time.
    pil_image = load_img(image_path, target_size=IMAGE_SIZE, color_mode="rgb")
    image_array = img_to_array(pil_image)
    batch = np.expand_dims(image_array, axis=0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    for model_name in ("vgg16", "resnet50"):
        model_path = os.path.join(PROJECT_DIR, "outputs", f"{model_name}_model.keras")
        if not os.path.isfile(model_path):
            print(f"Skipping {model_name.upper()}: saved model not found at {model_path}")
            continue

        model = tf.keras.models.load_model(model_path)
        probabilities = model.predict(batch, verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[class_index]
        confidence = float(probabilities[class_index]) * 100
        results.append((model_name.upper(), predicted_class, confidence))
        print(f"{model_name.upper()} prediction: {predicted_class} ({confidence:.2f}%)")
        del model
        tf.keras.backend.clear_session()

    if not results:
        raise FileNotFoundError(
            "No saved models found. Run main.py first to create the model files."
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(pil_image)
    ax.axis("off")
    lines = [f"{name}: {label} ({confidence:.2f}%)"
             for name, label, confidence in results]
    ax.set_title("New Image Prediction\n" + "\n".join(lines), fontsize=12, pad=12)
    fig.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "new_image_prediction.png")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    print(f"Prediction image saved to: {output_path}")


if __name__ == "__main__":
    main()
