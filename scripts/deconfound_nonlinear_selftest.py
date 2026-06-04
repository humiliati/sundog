#!/usr/bin/env python
"""Attack-B nonlinear de-confound self-test.

Demonstrates (rather than asserts) the R1.5 boundary: u = XOR(b_0,b_1,b_2) is LINEARLY
undecodable from the digit features b (the de-confound HOLDS, det ~ 0.077) but NONLINEARLY
trivial (an MLP reads it) -> u is an *explicit function of visible inputs*, linearly-hidden
only. So the linear de-confound is a LINEAR-readout guard matched to the linear
determining-shadow read, NOT a claim that u is hidden. A random-label control confirms the
nonlinear decoder is reading real structure, not overfitting (Hewitt-Liang style).
"""
import numpy as np
from numpy.random import default_rng
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


def digit_bits():
    X = load_digits().images.astype(float)
    n = X.shape[0]
    feats = np.zeros((n, 8)); f = 0
    for br in range(4):
        for bc in range(2):
            feats[:, f] = X[:, br * 2:(br + 1) * 2, bc * 4:(bc + 1) * 4].mean((1, 2)); f += 1
    return (feats > np.median(feats, 0)).astype(int)


def det(acc, base):
    return (acc - base) / max(1 - base, 1e-9)


if __name__ == "__main__":
    b = digit_bits()
    u = (b[:, 0] ^ b[:, 1] ^ b[:, 2])
    base_u = max(u.mean(), 1 - u.mean())
    rng = default_rng(0)
    rand = (rng.random(len(u)) < u.mean()).astype(int)            # matched-base random label
    base_r = max(rand.mean(), 1 - rand.mean())

    def cv(model, y):
        return float(cross_val_score(model, b, y, cv=5).mean())

    lin = cv(LogisticRegression(max_iter=300), u)
    mlp = cv(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=0), u)
    ctrl = cv(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=0), rand)

    print(f"u base rate = {base_u:.3f}   (n={len(u)})\n")
    print(f"  LINEAR   probe  b -> u    : acc {lin:.3f}   det {det(lin, base_u):+.3f}"
          f"   <- de-confound bar (want ~0; matches 0-pre 0.077)")
    print(f"  NONLINEAR probe b -> u    : acc {mlp:.3f}   det {det(mlp, base_u):+.3f}"
          f"   <- u is an explicit input-function (nonlinearly trivial)")
    print(f"  NONLINEAR control b->rand : acc {ctrl:.3f}   det {det(ctrl, base_r):+.3f}"
          f"   <- want ~0: decoder reads real structure, not overfit")
    print("\nReading: u is LINEARLY-hidden (linear det ~ 0 = de-confound holds) yet")
    print("NONLINEARLY-trivial (the MLP reads it). The linear de-confound is the matched")
    print("guard for a LINEAR determining-shadow read; it is inadequate for any stronger")
    print("'u is hidden / more than we know' claim BY CONSTRUCTION. R1.5 boundary demonstrated.")
