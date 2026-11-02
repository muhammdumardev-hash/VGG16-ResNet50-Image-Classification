import os
import random
import shutil

# Source dataset downloaded by KaggleHub
SOURCE_DIR = r"C:\Users\baba\.cache\kagglehub\datasets\alessiocorrado99\animals10\versions\2\raw-img"

# Project dataset folder
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

# Italian folder names in Animals-10 dataset -> project class names
CLASSES = {
    "gatto": "cats",
    "cane": "dogs",
    "cavallo": "horses",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TRAIN_COUNT = 50
VALIDATION_COUNT = 10

random.seed(42)


def main():
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(
            "Animals-10 dataset folder was not found. Download it with KaggleHub first:\n"
            "kagglehub.dataset_download('alessiocorrado99/animals10')"
        )

    for source_name, class_name in CLASSES.items():
        source_folder = os.path.join(SOURCE_DIR, source_name)

        if not os.path.isdir(source_folder):
            print(f"Skipping {class_name}: source folder not found: {source_folder}")
            continue

        images = [
            name for name in os.listdir(source_folder)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        ]
        random.shuffle(images)

        required = TRAIN_COUNT + VALIDATION_COUNT
        if len(images) < required:
            print(f"Skipping {class_name}: found {len(images)} images; need {required}.")
            continue

        selections = {
            "train": images[:TRAIN_COUNT],
            "validation": images[TRAIN_COUNT:required],
        }

        for split, filenames in selections.items():
            target_folder = os.path.join(DATASET_DIR, split, class_name)
            os.makedirs(target_folder, exist_ok=True)

            for filename in filenames:
                source_path = os.path.join(source_folder, filename)
                target_path = os.path.join(target_folder, filename)
                shutil.copy2(source_path, target_path)

        print(
            f"{class_name}: {len(selections['train'])} training + "
            f"{len(selections['validation'])} validation images"
        )

    print("\nDataset preparation finished.")
    print(f"Dataset location: {DATASET_DIR}")


if __name__ == "__main__":
    main()
