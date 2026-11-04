"""
Reproducibility utilities.

Call set_seed() at the very start of any training or evaluation
script's main(), BEFORE building datasets, models, or DataLoaders.

Previously, config.py defined SEED = 1337 but nothing ever called
torch.manual_seed() or seeded the DataLoader shuffling, so every
reported metric (growth IoU, etc.) came from one uncontrolled run.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seeded_generator(seed):
    """
    Pass as generator= to DataLoader(shuffle=True, ...) so the
    shuffle order is reproducible across runs with the same seed.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id):
    """
    Pass as worker_init_fn= to DataLoader whenever num_workers > 0,
    so any numpy/random calls inside worker processes (e.g. future
    augmentation) are also reproducible.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)