"""Datafit functions module."""

from .base import BaseDatafit, MipDatafit
from .kullbackleibler import KullbackLeibler
from .leastsquares import Leastsquares
from .logcosh import Logcosh
from .logistic import Logistic
from .squaredhinge import Squaredhinge

__all__ = [
    "BaseDatafit",
    "MipDatafit",
    "KullbackLeibler",
    "Leastsquares",
    "Logcosh",
    "Logistic",
    "Squaredhinge",
]
