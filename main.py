"""
VGG16 vs ResNet50 transfer-learning image classification.
Expected dataset structure:
dataset/
  train/{cats,dogs,horses}/
  validation/{cats,dogs,horses}/

Run:
  python main.py
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
CLASS_NAMES = ["cats", "dogs", "horses"]

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
CM_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")


def make_datasets():
    for split_dir in (TRAIN_DIR, VAL_DIR):
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Dataset folder not found: {split_dir}")
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir) or not os.listdir(class_dir):
                raise FileNotFoundError(
                    f"Missing images for '{class_name}' in {class_dir}. "
                    "Run prepare_dataset.py first."
                )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    return train_ds, val_ds


def build_model(model_name, preprocess_function):
    if model_name == "VGG16":
        base_model = VGG16(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
    else:
        base_model = ResNet50(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )

    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Lambda(preprocess_function, name=f"{model_name.lower()}_preprocess")(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"{model_name}_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plot(history, model_name):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name.lower()}_training.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(CM_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("TensorFlow version:", tf.__version__)
    print("Loading images...")
    train_ds, val_ds = make_datasets()

    results = []
    all_reports = {}

    model_configs = [
        ("VGG16", vgg_preprocess),
        ("ResNet50", resnet_preprocess),
    ]

    for model_name, preprocess_function in model_configs:
        print(f"\n{'=' * 60}\nTraining {model_name}\n{'=' * 60}")
        model = build_model(model_name, preprocess_function)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            verbose=1,
        )
        save_training_plot(history, model_name)

        y_true = np.concatenate([labels.numpy() for _, labels in val_ds], axis=0)
        probabilities = model.predict(val_ds, verbose=1)
        y_pred = np.argmax(probabilities, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        report_text = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
        all_reports[model_name] = report_text

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"{model_name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(CM_DIR, f"{model_name.lower()}_confusion_matrix.png"), dpi=150)
        plt.close()

        model_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_model.keras")
        model.save(model_path)

        results.append({
            "model": model_name,
            "validation_accuracy": float(accuracy),
            "precision_macro": float(report_dict["macro avg"]["precision"]),
            "recall_macro": float(report_dict["macro avg"]["recall"]),
            "f1_macro": float(report_dict["macro avg"]["f1-score"]),
            "epochs": EPOCHS,
        })

        print(f"\n{model_name} validation accuracy: {accuracy:.4f}")
        print(report_text)
        del model
        tf.keras.backend.clear_session()

    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(os.path.join(RESULTS_DIR, "classification_reports.txt"), "w", encoding="utf-8") as file:
        for model_name, report_text in all_reports.items():
            file.write(f"===== {model_name} =====\n{report_text}\n\n")

    with open(os.path.join(RESULTS_DIR, "model_comparison.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nTraining and evaluation complete.")
    print("Comparison CSV:", csv_path)
    print("Plots:", PLOT_DIR)
    print("Confusion matrices:", CM_DIR)
    print("Saved models:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
