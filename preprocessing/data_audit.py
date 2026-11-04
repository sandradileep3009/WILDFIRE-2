from PIL import Image
from collections import Counter
import json

import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
def inspect_images(files):

    resolutions = Counter()
    formats = Counter()
    modes = Counter()
    corrupted = []

    for path in files:

        try:
            with Image.open(path) as img:
                resolutions[f"{img.width}x{img.height}"] += 1
                formats[img.format] += 1
                modes[img.mode] += 1

        except Exception as e:
            corrupted.append({"file": str(path), "error": str(e)})

    return {
        "count": len(files),
        "resolutions": dict(resolutions),
        "formats": dict(formats),
        "modes": dict(modes),
        "corrupted": corrupted,
    }


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("WILDFIRE DATASET AUDIT")
    print("=" * 70)

    folders = {
    "train_fire": config.TRAIN_FIRE,
    "train_no_fire": config.TRAIN_NO_FIRE,
    "test_fire": config.TEST_FIRE,
    "test_no_fire": config.TEST_NO_FIRE,
    "segmentation_images": config.SEG_IMAGES,
    "masks": config.MASKS
    }

    report = {}

    for name, folder in folders.items():

        files = config.get_image_files(folder)

        print()
        print(f"{name}: {len(files)} files")

        info = inspect_images(files)

        report[name] = {
            "folder": str(folder),
            "count": info["count"],
            "resolutions": info["resolutions"],
            "formats": info["formats"],
            "modes": info["modes"],
            "corrupted": info["corrupted"],
        }

        if info["resolutions"]:
            print("Resolutions:")
            for resolution, count in info["resolutions"].items():
                print(f"  {resolution}: {count}")

    output = config.REPORT_DIR / "dataset_audit.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print()
    print("Report saved to:")
    print(output)


if __name__ == "__main__":
    main()
