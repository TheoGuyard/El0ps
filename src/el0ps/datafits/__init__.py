"""Data-fidelity functions."""

from .base import BaseDatafit, MipDatafit, SmoothDatafit, StronglyConvexDatafit
from .kullbackleibler import Kullbackleibler
from .leastsquares import Leastsquares
from .logcosh import Logcosh
from .logistic import Logistic
from .squaredhinge import Squaredhinge

__all__ = [
    "BaseDatafit",
    "MipDatafit",
    "SmoothDatafit",
    "StronglyConvexDatafit",
    "Kullbackleibler",
    "Leastsquares",
    "Logcosh",
    "Logistic",
    "Squaredhinge",
]
