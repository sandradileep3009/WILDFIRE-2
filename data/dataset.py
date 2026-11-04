from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

def extract_frame_number(path):
    match = re.search(r"image_(\d+)", str(path))
    if match:
        return int(match.group(1))
    return None

def build_runs(frame_dir):
    frame_dir = Path(frame_dir)
    frames = []
    for path in frame_dir.glob("*"):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        frame_number = extract_frame_number(path)
        if frame_number is not None:
            frames.append((frame_number, path))

    frames.sort(key=lambda x: x[0])
    runs = []
    current_run = []
    previous_number = None
    for frame_number, path in frames:
        if (
            previous_number is not None
            and frame_number != previous_number + 1
        ):
            if current_run:
                runs.append(current_run)

            current_run = []

        current_run.append((frame_number, path))
        previous_number = frame_number

    if current_run:
        runs.append(current_run)

    return runs


def load_mask_lookup(csv_path):
    df = pd.read_csv(csv_path)

    lookup = {}

    for _, row in df.iterrows():

        status = str(
            row.get("status", "")
        ).lower()

        if "match" not in status:
            continue
        frame_number = row.get("frame_number")
        if pd.isna(frame_number):
            continue
        frame_number = int(frame_number)
        mask_path = row.get("mask")
        if pd.isna(mask_path):
            continue
        lookup[frame_number] = Path(mask_path)
    return lookup


def build_sequence_index(
    frame_dir,
    pairs_csv,
    sequence_length=3,
):
    runs = build_runs(frame_dir)

    mask_lookup = load_mask_lookup(pairs_csv)

    records = []

    run_id = 0

    for run in runs:

        frame_numbers = [
            x[0] for x in run
        ]

        frame_paths = {
            number: path
            for number, path in run
        }

        for i in range(
            sequence_length,
            len(frame_numbers)
        ):

            input_numbers = frame_numbers[
                i - sequence_length:i
            ]

            target_number = frame_numbers[i]

            expected = list(
                range(
                    input_numbers[0],
                    target_number + 1
                )
            )

            if (
                input_numbers + [target_number]
                != expected
            ):
                continue

            if not all(
                n in mask_lookup
                for n in input_numbers
            ):
                continue

            if target_number not in mask_lookup:
                continue

            records.append(
                {
                    "run_id": run_id,

                    "input_frames": "|".join(
                        str(frame_paths[n])
                        for n in input_numbers
                    ),

                    "current_mask": str(
                        mask_lookup[
                            input_numbers[-1]
                        ]
                    ),

                    "target_mask": str(
                        mask_lookup[
                            target_number
                        ]
                    ),

                    "current_frame_number":
                        input_numbers[-1],

                    "target_frame_number":
                        target_number,
                }
            )

        run_id += 1

    return pd.DataFrame(records)


class SpreadSequenceDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_size=(128, 128),
    ):
        self.df = dataframe.reset_index(
            drop=True
        )

        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def load_image(self, path):

        image = Image.open(path).convert("RGB")

        image = image.resize(
            self.image_size
        )

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        return torch.tensor(
            image,
            dtype=torch.float32
        )

    def load_mask(self, path):

        mask = Image.open(path).convert("L")

        mask = mask.resize(
            self.image_size,
            Image.Resampling.NEAREST
        )

        mask = np.asarray(
            mask,
            dtype=np.float32
        )

        mask = (
            mask > 0
        ).astype(np.float32)

        return torch.tensor(
            mask,
            dtype=torch.float32
        )

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        frame_paths = (
            str(row["input_frames"])
            .split("|")
        )

        frames = [
            self.load_image(path)
            for path in frame_paths
        ]

        sequence = torch.stack(
            frames,
            dim=0
        )

        current_mask = self.load_mask(
            row["current_mask"]
        )

        target_mask = self.load_mask(
            row["target_mask"]
        )

        growth_mask = (
            target_mask
            * (1.0 - current_mask)
        )

        return (
            sequence,
            target_mask.unsqueeze(0),
            current_mask.unsqueeze(0),
            growth_mask.unsqueeze(0),
        )
