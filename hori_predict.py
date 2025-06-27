#!/usr/bin/env python
# hori_detect.py -----------------------------------------------------------
"""
Rule-based detector for Hori stage-4 (flattening) and stage-5 (theta ripples)
using Muse Mind Monitor CSV files.

Now accepts an *optional* “minutes” argument to limit how much data is analysed.

Usage
-----
    python hori_detect.py <folder_with_csvs> [minutes]

    minutes : positive number → analyse only the first N minutes of each file.
              omit to process the entire recording.

Outputs
-------
* <file>_pred.csv  – per-epoch predictions (0 = None, 4 = Hori-4, 5 = Hori-5)
* <file>.png       – diagnostic plot with six panels
"""
import sys, os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, detrend
from scipy.stats  import gmean
import antropy as ant

# ─── tweakables ───────────────────────────────────────────────────────────
WIN_LEN, STEP      = 5.0, 1              # epoch length & hop (s)
ACC_RMS_THRESH_G   = 0.01                # movement cut-off (g)
RMS_FLAT_THRESH    = 40.0                # µV  → gate for stage-4
FLATNESS_THRESH    = 0.55                # dimensionless
THETA_ALPHA_RATIO4 = 1.1                 # θ/α upper bound for stage-4
THETA_ALPHA_RATIO5 = 1.5                 # θ/α lower bound for stage-5
WIN_HYSTERESIS     = 10                  # consecutive hits to enter a stage
SMOOTH_WIN         = 5                   # centred MA smoothing (epochs)
EPSILON            = 1e-12               # avoid ÷0
REQ_CHANNELS       = (
    "RAW_AF7", "RAW_AF8",
    "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"
)
# ──────────────────────────────────────────────────────────────────────────


# ─── helpers ──────────────────────────────────────────────────────────────
def bandpass(sig, lo, hi, fs, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def load_mindmonitor_csv(path):
    df = pd.read_csv(path)
    try:
        df = df[df["Elements"].isna()]                 # drop event rows
    except:
        pass
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    df = df.set_index("TimeStamp").sort_index()
    return df[~df.index.duplicated()]


def smooth_metrics(met, win=SMOOTH_WIN):
    cols = ["rms_uv", "flatness",
            "p_theta", "p_alpha", "p_total",
            "theta_alpha_ratio", "move_rms_g", "lz_complexity"]  # ADDED
    met[cols] = (met[cols]
                 .rolling(window=win, center=True, min_periods=1)
                 .mean())
    return met


# ─── per-epoch features ───────────────────────────────────────────────────
def epoch_features(df, fs):
    nper  = int(WIN_LEN * fs)
    nstep = int(STEP    * fs)
    starts = np.arange(0, len(df) - nper, nstep, dtype=int)

    front_avg = df[["RAW_AF7", "RAW_AF8"]].mean(axis=1).to_numpy()
    ax, ay, az = [df[c].to_numpy()
                  for c in ("Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z")]
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

    rows = []
    for i0 in starts:
        i1 = i0 + nper
        t_mid = df.index[i0:i1].mean()

        am_seg = detrend(accel_mag[i0:i1])
        move_rms = np.sqrt(np.mean(am_seg**2))
        is_moving = move_rms > ACC_RMS_THRESH_G

        seg_filt = bandpass(front_avg[i0:i1], 2, 30, fs)
        rms_uv   = np.sqrt(np.mean(seg_filt**2))

        f, Pxx = welch(seg_filt, fs=fs, nperseg=nper // 2)
        band = (f >= 2) & (f <= 30)
        f_b, P_b = f[band], Pxx[band]
        flatness = gmean(P_b) / P_b.mean()

        def bpow(lo, hi):
            m = (f_b >= lo) & (f_b <= hi)
            return np.trapz(P_b[m], f_b[m])

        p_theta = bpow(4, 7)
        p_alpha = bpow(8, 12)
        p_total = np.trapz(P_b, f_b)
        theta_alpha = p_theta / (p_alpha + EPSILON)

        # --- Lempel-Ziv complexity using antropy
        seg_norm = (seg_filt - np.mean(seg_filt)) / (np.std(seg_filt) + 1e-12)
        seg_bin = (seg_norm > 0).astype(int)  # Binarize

        lzc = ant.lziv_complexity(seg_bin, normalize=True)


        rows.append(dict(time=t_mid, rms_uv=rms_uv, flatness=flatness,
                         p_theta=p_theta, p_alpha=p_alpha, p_total=p_total,
                         theta_alpha_ratio=theta_alpha,
                         move_rms_g=move_rms, is_moving=is_moving,
                         lz_complexity=lzc))  # <- ADDED

    return pd.DataFrame(rows).set_index("time")


# ─── staging ──────────────────────────────────────────────────────────────
def stage_prediction(met):
    clean = met[~met.is_moving].copy()
    hit4 = clean.flatness > FLATNESS_THRESH
    hit5 = (clean.theta_alpha_ratio > THETA_ALPHA_RATIO5) & hit4

    clean["flag4"] = hit4.astype(int)
    clean["flag5"] = hit5.astype(int)

    h = WIN_HYSTERESIS
    stable4 = np.zeros(len(clean), int)
    run = 0
    for i, f in enumerate(clean.flag4):
        run = run + 1 if f else 0
        if run >= h: stable4[i] = 1
        if not f:    run = 0
    clean["stable4"] = stable4

    pred = np.zeros(len(clean), int)
    pred[stable4 == 1]        = 4
    pred[clean.flag5 == 1]    = 5
    for i in range(len(pred)):
        if pred[i] == 5 and 4 not in pred[max(0, i-h):i]:
            pred[i] = 0

    met = met.join(pd.Series(pred, index=clean.index, name="stage_pred"))
    met["stage_pred"].fillna(0, inplace=True)
    return met.astype({"stage_pred": int})


# ─── plotting ─────────────────────────────────────────────────────────────
def quick_plot(met, title):
    t = (met.index - met.index[0]).total_seconds().to_numpy()
    rel_theta = met.p_theta / (met.p_total + EPSILON)
    abs_theta = met.p_theta

    plot_cols = ["rms_uv", "flatness", "p_theta", "p_alpha", "p_total", "theta_alpha_ratio", "lz_complexity"]
    for col in plot_cols:
        met.loc[met.is_moving, col] = np.nan
    rel_theta[met.is_moving.values] = np.nan
    abs_theta[met.is_moving.values] = np.nan

    fig, ax = plt.subplots(8, 1, figsize=(14, 16), sharex=True, gridspec_kw=dict(hspace=0.35))
    fig.suptitle(title, fontsize=14)

    ax[0].plot(t, met.move_rms_g, lw=0.7)
    ax[0].axhline(ACC_RMS_THRESH_G, color="r", ls="--", lw=0.8)
    ax[0].set_ylabel("Move\nRMS (g)")

    ax[1].plot(t, met.rms_uv, lw=0.7)
    ax[1].axhline(RMS_FLAT_THRESH, color="r", ls="--", lw=0.8)
    ax[1].set_ylabel("RMS µV")

    ax[2].plot(t, met.flatness, lw=0.7)
    ax[2].axhline(FLATNESS_THRESH, color="r", ls="--", lw=0.8)
    ax[2].set_ylabel("Flatness")

    ax[3].plot(t, rel_theta, lw=0.7)
    ax[3].set_ylabel("θ / Σ₂₋₃₀ Hz")

    ax[4].plot(t, abs_theta, lw=0.7)
    ax[4].set_ylabel("Abs θ")

    ax[5].plot(t, met.theta_alpha_ratio, lw=0.7)
    ax[5].axhline(THETA_ALPHA_RATIO4, color="g", ls="--", lw=0.8, label="stage-4 upper")
    ax[5].axhline(THETA_ALPHA_RATIO5, color="r", ls="--", lw=0.8, label="stage-5 lower")
    ax[5].set_ylabel("θ / α")
    ax[5].legend()

    ax[6].plot(t, met.lz_complexity, lw=0.7)
    ax[6].set_ylabel("LZc (front)")

    ax[7].step(t, met.stage_pred, where="mid")
    ax[7].set_ylabel("Stage")
    ax[7].set_xlabel("Time (s)")

    fig.savefig(f"{title.replace(' ', '_')}.png", bbox_inches="tight")
    plt.show()


# ─── per-file driver ──────────────────────────────────────────────────────
def process_file(path, minutes=None):
    name = os.path.basename(path)
    print("Processing", name)
    df = load_mindmonitor_csv(path)

    missing = [ch for ch in REQ_CHANNELS if ch not in df.columns]
    if missing:
        warnings.warn(f"{name}: missing {missing}; skipping file"); return

    # estimate fs from median Δt
    diffs = np.diff(df.index.view(np.int64) / 1e9)
    diffs = diffs[diffs > 0]
    # fs = 1.0 / np.median(diffs) if len(diffs) else 256.0
    fs = 256
    # optional truncation and removal of first 30s -------------------------
    if minutes is not None and minutes > 0:
        n_samples = int(minutes * 60 * fs)
        df = df.iloc[:n_samples]
    # remove first 30 seconds of data
    n_remove = int(30 * fs)
    if len(df) > n_remove:
        df = df.iloc[n_remove:]

    met = stage_prediction(smooth_metrics(epoch_features(df, fs)))

    out_csv = path.replace(".csv", "_pred.csv", "_zscored.csv", )
    met.to_csv(out_csv)
    print("  → saved predictions to", os.path.basename(out_csv))

    quick_plot(met, f"Front-avg metrics + Hori stages – {name}")


# ─── CLI entry point ──────────────────────────────────────────────────────
def main(folder, minutes=None):
    csv_files = sorted(f for f in glob.glob(os.path.join(folder, "*.csv"))
                       if "_pred" not in os.path.basename(f))
    if not csv_files:
        print("No CSV files found in", folder); return
    for path in csv_files:
        process_file(path, minutes)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__); sys.exit(1)

    minutes_arg = float(sys.argv[2]) if len(sys.argv) == 3 else None
    if minutes_arg is not None and minutes_arg <= 0:
        print("Minutes must be > 0"); sys.exit(1)

    main(sys.argv[1], minutes_arg)
