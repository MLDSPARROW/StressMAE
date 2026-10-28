#!/usr/bin/env python3
"""Compute the 19 handcrafted ECG features (HRV, band powers, signal statistics) of every non-overlapping 5 s window
of every recording and store them in cache/hrv_U.npz. Used by the handcrafted-feature baselines. No labels are used."""
import os
import numpy as np
import run_experiments as rx
from run_experiments import SUBJECTS, hrv_features

if __name__ == "__main__":
    path = os.path.join(rx.CACHE_DIR, "hrv_U.npz")
    np.savez(path, **{s: hrv_features(rx.load_subject(s)["U"]) for s in SUBJECTS})
    print("written", path)
