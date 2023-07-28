"""Data-fidelity functions."""

from .base import BaseDatafit, SmoothDatafit, ProximableDatafit
from .leastsquares import Leastsquares
from .logistic import Logistic

__all__ = [
    "BaseDatafit",
    "ProximableDatafit",
    "SmoothDatafit",
    "Leastsquares",
    "Logistic",
]
