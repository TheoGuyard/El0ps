"""Test utilities."""

import numpy as np


def make_classification(k, m, n, snr=10.0):
    x = np.zeros(n)
    s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
    x[s] = np.sign(np.random.randn(k))
    A = np.random.randn(m, n)
    p = 1.0 / (1.0 + np.exp(-snr * (A @ x)))
    y = 2.0 * np.random.binomial(1, p, size=m) - 1.0
    return A, y, x


def make_regression(k, m, n, snr=10.0):
    x = np.zeros(n)
    s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
    x[s] = np.sign(np.random.randn(k))
    A = np.random.randn(m, n)
    y = A @ x
    e = np.random.randn(m)
    e *= np.sqrt((y @ y) / (snr * (e @ e)))
    y += e
    return A, y, x


def make_svc(k, m, n, snr=10.0):
    x = np.zeros(n)
    s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
    x[s] = np.sign(np.random.randn(k))
    A = np.random.randn(m, n)
    p = 1.0 / (1.0 + np.exp(-snr * (A @ x)))
    y = 2.0 * (p > 0.5) - 1.0
    return A, y, x
