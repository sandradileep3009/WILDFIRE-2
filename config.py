from pathlib import Path
import re
import random
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data"
TRAIN_FIRE = DATASET_ROOT / "Training" / "Fire"
TRAIN_NO_FIRE = DATASET_ROOT / "Training" / "No_Fire"
TEST_FIRE = DATASET_ROOT / "Test" / "Fire"
TEST_NO_FIRE = DATASET_ROOT / "Test" / "No_Fire"

SEGMENTATION_DIR = DATASET_ROOT / "Segmentation"
SEG_IMAGES = SEGMENTATION_DIR / "Images"

SEG_MASKS = DATASET_ROOT / "Masks"
MASKS = SEG_MASKS

REPORT_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data_processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
EXAMPLES_DIR = RESULTS_DIR / "examples"
PREDS_DIR = RESULTS_DIR / "preds"

for directory in (REPORT_DIR, PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR, EXAMPLES_DIR, PREDS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

IMAGE_MASK_PAIRS_CSV = PROCESSED_DIR / "image_mask_pairs.csv"
SEGMENTATION_PAIRS_CSV = PROCESSED_DIR / "segmentation_pairs.csv"
SEQUENCES_CSV = PROCESSED_DIR / "sequences.csv"

SEGMENTATION_SIZE = (256, 256)
TEMPORAL_SIZE = (128, 128) 

SEQUENCE_LENGTH = 3
PREDICTION_HORIZON = 1
CONVLSTM_HIDDEN_DIMS = (32, 32)

BATCH_SIZE_UNET = 4
BATCH_SIZE_CONVLSTM = 8

UNET_EPOCHS = 30
CONVLSTM_EPOCHS = 30

LEARNING_RATE_UNET = 1e-3
LEARNING_RATE_CONVLSTM = 1e-3
UNET_POS_WEIGHT = 3.0

GROWTH_POS_WEIGHT = 10.0

FULLMASK_POS_WEIGHT = 3.0

DEFAULT_THRESHOLD = 0.50

THRESHOLD_VALUES = (0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337

UNET_CHECKPOINT = CHECKPOINT_DIR / "unet_best.pt"
GROWTH_CHECKPOINT = CHECKPOINT_DIR / "growth_convlstm.pt"
FULLMASK_CHECKPOINT = CHECKPOINT_DIR / "fullmask_convlstm.pt"

UNET_HISTORY_CSV = RESULTS_DIR / "unet_training_history.csv"
GROWTH_HISTORY_CSV = RESULTS_DIR / "growth_training_history.csv"
FULLMASK_HISTORY_CSV = RESULTS_DIR / "fullmask_training_history.csv"

EVALUATION_CSV = RESULTS_DIR / "evaluation_results.csv"
THRESHOLD_RESULTS_CSV = RESULTS_DIR / "threshold_results.csv"
LATENCY_JSON = RESULTS_DIR / "latency.json"

IMAGE_EXTENSIONS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",}
FIRE_FRAME_PATTERN = re.compile(r"frame[_]?(\d+)",re.IGNORECASE,)
MASK_FRAME_PATTERN = re.compile( r"image_(\d+)",re.IGNORECASE,)

def fire_frame_number(path):
    """Extract frame number from classification filename."""
    match = FIRE_FRAME_PATTERN.search(Path(path).stem)
    if match:
        return int(match.group(1))
    return None


def mask_frame_number(path):
    """Extract frame number from segmentation filename."""
    match = MASK_FRAME_PATTERN.search(Path(path).stem)
    if match:
        return int(match.group(1))
    return None

def get_image_files(folder):
    """Return all supported image files recursively."""
    folder = Path(folder)
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_DRONE_FRAMES = "drone-frames"
TOPIC_PREDICTIONS = "fire-predictions"

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

HORIZONS = (1, 5, 10, 20, 30)

INPUT_LEN = 3
INPUT_STRIDE = 1

SEEDS = (1337, 2023, 7, 42, 99)

SPLIT_FRACTIONS = (0.70, 0.72, 0.85, 0.87)

GROWTH_LOSS_POS_WEIGHT = 5.0

GROWTH_BAND_PX = None

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 10
NUM_WORKERS = 0

BOUNDARY_TOLERANCE_PX = 3          
BOOTSTRAP_BLOCK = 20              
BOOTSTRAP_SAMPLES = 1000
PREDS_DIR = RESULTS_DIR / "preds"
PREDS_DIR.mkdir(parents=True, exist_ok=True)
