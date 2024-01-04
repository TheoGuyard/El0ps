"""Data-fidelity functions."""

from .base import BaseDatafit, SmoothDatafit, ProximableDatafit
from .kullbackleibler import Kullbackleibler
from .leastsquares import Leastsquares
from .logcosh import Logcosh
from .logistic import Logistic
from .squaredhinge import Squaredhinge

__all__ = [
    "BaseDatafit",
    "ProximableDatafit",
    "SmoothDatafit",
    "Kullbackleibler",
    "Leastsquares",
    "Logcosh",
    "Logistic",
    "Squaredhinge",
]
