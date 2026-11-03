"""
Predict image class using saved VGG16 and ResNet50 models.
Classes: cats, dogs, horses
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "results")
CLASS_NAMES = ["cats", "dogs", "horses"]
IMAGE_SIZE = (224, 224)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py test_images/images.jpg")
        raise SystemExit(1)

    image_path = sys.argv[1]

    if not os.path.isabs(image_path):
        image_path = os.path.join(PROJECT_DIR, image_path)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_image = load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image_array = img_to_array(pil_image)
    batch = np.expand_dims(image_array, axis=0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    model_configs = [
        ("vgg16", vgg16_preprocess),
        ("resnet50", resnet50_preprocess),
    ]

    for model_name, preprocess_function in model_configs:
        model_path = os.path.join(
            PROJECT_DIR, "outputs", f"{model_name}_model.keras"
        )

        if not os.path.isfile(model_path):
            print(f"Model not found: {model_path}")
            continue

        print(f"\nLoading {model_name.upper()} model...")

        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    "preprocess_input": preprocess_function
                },
                compile=False
            )

            # Preprocessing is already included inside the saved model.
            probabilities = model.predict(batch, verbose=0)[0]

            class_index = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[class_index]
            confidence = float(probabilities[class_index]) * 100

            results.append(
                (model_name.upper(), predicted_class, confidence)
            )

            print(
                f"{model_name.upper()} prediction: "
                f"{predicted_class} ({confidence:.2f}%)"
            )

            del model
            tf.keras.backend.clear_session()

        except Exception as error:
            print(f"Error loading {model_name.upper()}: {error}")

    if not results:
        raise RuntimeError(
            "Neither model could make a prediction. "
            "Check the saved models and the error messages above."
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(pil_image)
    ax.axis("off")

    lines = [
        f"{name}: {label} ({confidence:.2f}%)"
        for name, label, confidence in results
    ]

    ax.set_title(
        "New Image Prediction\n" + "\n".join(lines),
        fontsize=12,
        pad=12
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR, "new_image_prediction.png"
    )

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"\nPrediction image saved to:\n{output_path}")


if __name__ == "__main__":
    main()