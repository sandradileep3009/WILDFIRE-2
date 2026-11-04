from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "Segmentation"
    / "Images"
)

MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "Masks"
)

OUTPUT = (
    PROJECT_ROOT
    / "data_processed"
    / "segmentation_pairs.csv"
)


def get_number(path):

    match = re.search(
        r"image_(\d+)",
        path.stem.lower()
    )

    if match:
        return int(match.group(1))

    return None


def main():

    images = {}

    for path in IMAGE_DIR.glob("*.jpg"):

        number = get_number(path)

        if number is not None:
            images[number] = path


    masks = {}

    for path in MASK_DIR.glob("*.png"):

        number = get_number(path)

        if number is not None:
            masks[number] = path


    rows = []

    all_numbers = sorted(
        set(images.keys())
        | set(masks.keys())
    )


    for number in all_numbers:

        image = images.get(number)
        mask = masks.get(number)

        if image is not None and mask is not None:

            status = "matched"

        elif image is not None:

            status = "missing_mask"

        else:

            status = "missing_image"


        rows.append(
            {
                "frame_number": number,

                "image":
                    str(image)
                    if image
                    else "",

                "mask":
                    str(mask)
                    if mask
                    else "",

                "status":
                    status,
            }
        )


    df = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT,
        index=False
    )


    print("=" * 70)
    print("FLAME SEGMENTATION MATCHING")
    print("=" * 70)

    print(
        f"Images found:        {len(images)}"
    )

    print(
        f"Masks found:         {len(masks)}"
    )

    print(
        f"Matched pairs:       "
        f"{(df['status'] == 'matched').sum()}"
    )

    print(
        f"Missing masks:       "
        f"{(df['status'] == 'missing_mask').sum()}"
    )

    print(
        f"Missing images:      "
        f"{(df['status'] == 'missing_image').sum()}"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()