"""Experimental drift budget; not a theorem or semantic consistency metric."""
import math
from dataclasses import dataclass
import numpy as np

@dataclass
class DriftBudget:
    budget: float = 1.0
    total: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.budget) or self.budget <= 0:
            raise ValueError('Budget must be finite and positive')

    def update(self, previous, current):
        # Channel means detect coarse appearance changes, not identity/geometry.
        a, b = np.asarray(previous, dtype=float), np.asarray(current, dtype=float)
        if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError('Expected matching RGB frames')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Non-finite frame')
        drift = float(np.abs(a.mean((0, 1)) - b.mean((0, 1))).mean() / 255)
        self.total += drift
        return {'appearance_drift': drift, 'cumulative_drift': self.total,
                'budget': self.budget, 'stop': self.total >= self.budget}
