#!/usr/bin/env python3
"""
REFERENCE implementation — TUSZ EDF -> pyspi instance generator.

This is a drop-in reference for the sibling compute repo
(../mts-spi-study-cluster), a sibling of `var_chat_a`. It is NOT wired into
this (consuming) repo; it documents the exact preprocessing the SPI pipeline
assumes so the per-instance dirs match the VAR format:

    <out>/<class>/<patient>__<session>__w<k>/
        timeseries.npy   (T, M) float64, per-channel z-scored
        spi_mpis.npz     pyspi MPI matrices, keys = SPI names
        meta.json        {"pyspi": {"spis": [{"name": ...}, ...]}, ...}

Dependencies (install in the cluster env, NOT here): mne, pyspi, numpy, scipy.

------------------------------------------------------------------------------
CORRECTNESS FLAGS — verify before a bulk run (do not trust blind):

1. MONTAGE (TCP_AR_PAIRS below) is reconstructed from the standard NEDC
   01_tcp_ar montage and is UNVERIFIED. Confirm against the montage file in
   the sample's DOCS/ (e.g. 01_tcp_ar_montage.txt). TUH uses OLD 10-20 names
   (T3/T4/T5/T6, not T7/T8/P7/P8). Annotations are on THIS bipolar montage,
   so the SPI channels must be these derivations, not raw *-REF channels.

2. SAMPLING RATE varies per file (250 typical, but 256/400/512 occur). Always
   read fs from the EDF and resample to TARGET_FS; never assume 250.

3. WINDOWING: instances are drawn from annotated seizure intervals only. Keep
   every window from one PATIENT in one SPLIT (use the official train/dev/eval
   dirs). Do not pool patients across splits.

4. LEAKAGE: z-score each window per channel (done here). Any cross-instance
   scaling (e.g. the SPIScaler in the consuming repo) is fit on TRAIN only —
   that already holds in run_pipeline.py; nothing to do here.
------------------------------------------------------------------------------
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# --- config ------------------------------------------------------------------
TARGET_FS = 128          # resample target (Hz); keeps T in pyspi's tested regime
WINDOW_S = 8.0           # window length (s) -> T = TARGET_FS * WINDOW_S = 1024
WINDOWS_PER_SEIZURE = 3  # non-overlapping windows sampled per seizure interval
BANDPASS = (0.5, 45.0)   # Hz
NOTCH = 60.0             # Hz (US mains)
FOCUS_TYPES = ("fnsz", "gnsz")

# UNVERIFIED — confirm against DOCS/ montage file (see flag 1).
TCP_AR_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"),
    ("C4", "T4"), ("T4", "A2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]  # M = 22 bipolar derivations


def _clean(ch: str) -> str:
    """Normalise an EDF channel label e.g. 'EEG FP1-REF' -> 'FP1'."""
    return ch.upper().replace("EEG", "").replace("-REF", "").replace("-LE", "").strip()


def load_bipolar(edf_path: Path):
    """Return (montage_signal (M, n_samp), fs) on the TCP_AR bipolar montage.

    Requires mne. Raises if a montage electrode is missing (do not silently
    drop channels — a missing electrode changes M and breaks pair alignment).
    """
    import mne  # cluster dependency

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    by_name = {_clean(ch): i for i, ch in enumerate(raw.ch_names)}
    data = raw.get_data()  # (n_ch, n_samp)

    rows = []
    for a, b in TCP_AR_PAIRS:
        if a not in by_name or b not in by_name:
            raise KeyError(f"{edf_path.name}: missing electrode {a} or {b}")
        rows.append(data[by_name[a]] - data[by_name[b]])
    return np.asarray(rows), fs


def preprocess(sig: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """Bandpass + notch + resample to TARGET_FS. Returns (M, n_samp_new)."""
    from scipy.signal import butter, filtfilt, iirnotch, resample_poly

    ny = fs / 2.0
    b, a = butter(4, [BANDPASS[0] / ny, BANDPASS[1] / ny], btype="band")
    sig = filtfilt(b, a, sig, axis=1)
    bn, an = iirnotch(NOTCH / ny, Q=30)
    sig = filtfilt(bn, an, sig, axis=1)
    # rational resample fs -> TARGET_FS
    from math import gcd
    g = gcd(int(round(fs)), TARGET_FS)
    up, down = TARGET_FS // g, int(round(fs)) // g
    sig = resample_poly(sig, up, down, axis=1)
    return sig, float(TARGET_FS)


def read_seizure_intervals(csv_path: Path, wanted_type: str) -> list[tuple[float, float]]:
    """Merged (start, stop) seconds for events whose label == wanted_type."""
    spans: list[tuple[float, float]] = []
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            cells = [c.strip().lower() for c in row]
            if wanted_type in cells:
                # term .csv: channel,start,stop,label,confidence
                try:
                    start = float(row[1]); stop = float(row[2])
                    spans.append((start, stop))
                except (ValueError, IndexError):
                    continue
    # merge overlapping spans (per-channel rows repeat the same interval)
    spans.sort()
    merged: list[tuple[float, float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def windows_from_span(sig: np.ndarray, fs: float, span: tuple[float, float]):
    """Yield non-overlapping (M, T) windows fully inside a seizure span."""
    T = int(round(WINDOW_S * fs))
    s0 = int(round(span[0] * fs))
    s1 = int(round(span[1] * fs))
    n = 0
    pos = s0
    while pos + T <= s1 and n < WINDOWS_PER_SEIZURE:
        yield sig[:, pos:pos + T]
        pos += T
        n += 1


def zscore(win: np.ndarray) -> np.ndarray:
    """Per-channel z-score; returns (T, M) to match VAR timeseries.npy layout."""
    mu = win.mean(axis=1, keepdims=True)
    sd = win.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((win - mu) / sd).T.astype(np.float64)  # (T, M)


def run_pyspi(ts_TM: np.ndarray, pyspi_config: str) -> dict[str, np.ndarray]:
    """Compute SPIs on (T, M). Returns {spi_name: (M, M) matrix}.

    pyspi's Calculator expects (n_processes, n_observations) = (M, T).
    """
    from pyspi.calculator import Calculator  # cluster dependency

    calc = Calculator(dataset=ts_TM.T, configfile=pyspi_config)
    calc.compute()
    out: dict[str, np.ndarray] = {}
    # calc.table is a MultiIndex DataFrame keyed by SPI; adapt if the pyspi
    # version differs (verify the accessor once in the pilot).
    for spi in calc.spis:
        out[spi] = np.asarray(calc.table[spi].values, dtype=np.float64)
    return out


def generate_session(edf: Path, csv_lab: Path, split: str, out_root: Path,
                     pyspi_config: str) -> int:
    """Emit per-instance dirs for one session. Returns #instances written."""
    seiz_type = next((t for t in FOCUS_TYPES
                      if read_seizure_intervals(csv_lab, t)), None)
    if seiz_type is None:
        return 0
    sig, fs = load_bipolar(edf)
    sig, fs = preprocess(sig, fs)

    patient = edf.parts[-4]   # <split>/<patient>/<session>/<montage>/<file>
    session = edf.parts[-3]
    written = 0
    for span in read_seizure_intervals(csv_lab, seiz_type):
        for k, win in enumerate(windows_from_span(sig, fs, span)):
            ts = zscore(win)                        # (T, M)
            spis = run_pyspi(ts, pyspi_config)
            name = f"{patient}__{session}__s{int(span[0])}_w{k}"
            d = out_root / split / seiz_type / name
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "timeseries.npy", ts)
            np.savez(d / "spi_mpis.npz", **spis)
            meta = {"pyspi": {"spis": [{"name": n} for n in spis]},
                    "source": {"patient": patient, "session": session,
                               "split": split, "seizure_type": seiz_type,
                               "fs": fs, "M": ts.shape[1], "T": ts.shape[0]}}
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
            written += 1
    return written


# Pilot driver: iterate a small list of (edf, csv, split) tuples produced by
# docs/tuh_discovery.py, cap total instances, measure wall-clock per instance.
# Full run = the same, fanned out as Gadi array jobs over sessions.
