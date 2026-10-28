#!/usr/bin/env python3
"""Cache all RespiBAN chest channels at 100 Hz for the multi-channel StressMAE.
Channels: 0 ECG (0.5 Hz high-pass), 1 EDA (1 Hz low-pass), 2 EMG envelope (20 Hz high-pass, rectified, 2 Hz low-pass),
3 RESP (0.05-1 Hz band-pass), 4 TEMP, 5 ACC magnitude. Channels 1-5 are z-scored with the subject's own whole
recording (no labels). ECG keeps the per-window robust normalisation of the original StressMAE."""
import os, pickle
import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt
import run_experiments as rx
from run_experiments import SUBJECTS, WESAD_DIR, FS0, FS

OUT = os.path.join(rx.CACHE_DIR, "multi"); os.makedirs(OUT, exist_ok=True)
NAMES = ["ECG", "EDA", "EMG", "RESP", "TEMP", "ACC"]


def down(x, cutoff):
    """low-pass below cutoff, then keep every 7th sample (700 -> 100 Hz)"""
    sos = butter(4, cutoff / (FS0 / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x)[:: FS0 // FS]


def main():
    for sid in SUBJECTS:
        out = os.path.join(OUT, f"{sid}.npz")
        if os.path.exists(out):
            continue
        with open(os.path.join(WESAD_DIR, sid, f"{sid}.pkl"), "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        c = raw["signal"]["chest"]
        g = lambda k: np.asarray(c[k], dtype=np.float64)
        ecg = g("ECG").ravel(); b, a = butter(3, 0.5 / (FS0 / 2), btype="high"); ecg = down(filtfilt(b, a, ecg), 40)
        eda = down(g("EDA").ravel(), 1.0)
        emg = g("EMG").ravel(); sos = butter(4, 20 / (FS0 / 2), btype="high", output="sos")
        emg = down(np.abs(sosfiltfilt(sos, emg)), 2.0)
        sos = butter(2, [0.05 / (FS0 / 2), 1.0 / (FS0 / 2)], btype="band", output="sos")
        resp = down(sosfiltfilt(sos, g("Resp").ravel()), 2.0)
        temp = down(g("Temp").ravel(), 1.0)
        acc = down(np.linalg.norm(g("ACC"), axis=1), 5.0)
        n = min(map(len, [ecg, eda, emg, resp, temp, acc]))
        S = np.stack([x[:n] for x in [ecg, eda, emg, resp, temp, acc]]).astype(np.float64)
        S[2] = np.log1p(np.maximum(S[2], 0) * 1e3)                         # EMG envelope is heavy-tailed
        for k in range(1, 6):                                               # per-subject z-score, label-free
            S[k] = np.clip((S[k] - S[k].mean()) / (S[k].std() + 1e-8), -8, 8)
        lab = np.asarray(raw["label"]).ravel()[:: FS0 // FS][:n]
        np.savez_compressed(out, sig=S.astype(np.float32), lab=lab.astype(np.int8))
        print(f"cached {sid}: {n / FS / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
