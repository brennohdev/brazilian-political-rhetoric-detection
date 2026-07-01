import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for all random number generators.

    Covers: Python random, NumPy, and PyTorch (if available).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
