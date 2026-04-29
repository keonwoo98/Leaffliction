"""Centralized random seeding for reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed every RNG that may affect training/eval results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
