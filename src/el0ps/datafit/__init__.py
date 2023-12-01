"""Data-fidelity functions."""

from .base import BaseDatafit, SmoothDatafit, ProximableDatafit
from .kullbackleibler import KullbackLeibler
from .leastsquares import Leastsquares
from .logistic import Logistic
from .squaredhinge import Squaredhinge

__all__ = [
    "BaseDatafit",
    "ProximableDatafit",
    "SmoothDatafit",
    "KullbackLeibler",
    "Leastsquares",
    "Logistic",
    "Squaredhinge",
]
