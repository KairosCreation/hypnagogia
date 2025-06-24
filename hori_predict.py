#!/usr/bin/env python
# hori_detect.py -----------------------------------------------------------
"""
Rule-based detector for Hori stage-4 (flattening) and stage-5 (theta ripples)
using Muse Mind Monitor CSV files.

Usage
-----
    python hori_detect.py <folder_with_csvs>

Outputs
-------
    For each <file>.csv ➜ <file>_pred.csv with one row per 2-second epoch
    and a column `stage_pred` : 0=None, 4=Hori-4, 5=Hori-5
"""
import sys, os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, detrend
from scipy.stats  import gmean

# ───── parameters you may tweak ───────────────────────────────────────────
WIN_LEN            = 5.0     # seconds
STEP               = 1     # seconds (50 % overlap)
ACC_RMS_THRESH_G   = 0.05    # movement cut-off
RMS_FLAT_THRESH    = 40.0    # µV  → first gate for stage-4
FLATNESS_THRESH    = 0.55    # dimensionless
THETA_ALPHA_RATIO4 = 1.1     # keep θ/α below this for stage-4
THETA_ALPHA_RATIO5 = 1.5     # θ/α above this for stage-5
WIN_HYSTERESIS     = 10       # consecutive hits to enter a stage
SMOOTH_WIN = 5          # epochs; 5×1-s-hop = 5 s centred moving-average

# ──────────────────────────────────────────────────────────────────────────


def bandpass(sig, lo, hi, fs, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def load_mindmonitor_csv(path):
    df = pd.read_csv(path)
    df = df[df["Elements"].isna()]                   # drop event rows
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    df = df.set_index("TimeStamp").sort_index()
    df = df[~df.index.duplicated()]
    return df

def smooth_metrics(met, win=SMOOTH_WIN):
    """Centered moving-average smoothing for the numeric feature columns."""
    num_cols = ["rms_uv", "flatness", "p_theta", "p_alpha",
                "theta_alpha_ratio", "move_rms_g"]
    met[num_cols] = (
        met[num_cols]
        .rolling(window=win, center=True, min_periods=1)
        .mean()
    )
    # keep is_moving unchanged: movement flag stays per-epoch
    return met



def epoch_features(df, fs):
    """Return metrics DataFrame indexed by epoch centre-time."""
    nper  = int(WIN_LEN * fs)
    nstep = int(STEP    * fs)
    starts = np.arange(0, len(df) - nper, nstep, dtype=int)

    # signals --------------------------------------------------------------
    front_avg = df[["RAW_AF7", "RAW_AF8"]].mean(axis=1).to_numpy()

    ax, ay, az = [df[c].to_numpy()
                  for c in ("Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z")]
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

    rows = []
    for i0 in starts:
        i1 = i0 + nper
        t_mid = df.index[i0:i1].mean()

        # movement --------------------------------------------------------
        am_seg = detrend(accel_mag[i0:i1])
        move_rms = np.sqrt(np.mean(am_seg**2))
        is_moving = move_rms > ACC_RMS_THRESH_G

        # EEG metrics -----------------------------------------------------
        seg = front_avg[i0:i1]
        seg_filt = bandpass(seg, 2, 30, fs)

        rms_uv = np.sqrt(np.mean(seg_filt**2))

        f, Pxx = welch(seg_filt, fs=fs, nperseg=nper//2)
        band = (f >= 2) & (f <= 30)
        f_b, P_b = f[band], Pxx[band]

        flatness = gmean(P_b) / P_b.mean()

        def bpow(lo, hi):
            m = (f_b >= lo) & (f_b <= hi)
            return np.trapz(P_b[m], f_b[m])

        p_theta = bpow(4, 7)
        p_alpha = bpow(8, 12)
        ratio   = p_theta / (p_alpha + 1e-12)

        rows.append(dict(time=t_mid,
                         rms_uv=rms_uv,
                         flatness=flatness,
                         p_theta=p_theta,
                         p_alpha=p_alpha,
                         theta_alpha_ratio=ratio,
                         move_rms_g=move_rms,
                         is_moving=is_moving))
    return pd.DataFrame(rows).set_index("time")


def stage_prediction(met):
    """
    Add stage_pred column (0/4/5) with hysteresis, enforcing:
    ─ stage-5 can only occur if at least WIN_HYSTERESIS consecutive
      stage-4 epochs have appeared in the immediately preceding window.
    """
    clean = met[~met.is_moving].copy()

    # -- dynamic theta threshold -----------------------------------------
    theta_thresh = clean["p_theta"].quantile(0.99)

    # -- per-epoch rule flags --------------------------------------------
    hit4 = (
        # (clean.rms_uv < RMS_FLAT_THRESH) &
        (clean.flatness > FLATNESS_THRESH)
    )
    hit5 = (
        (clean.theta_alpha_ratio > THETA_ALPHA_RATIO5) &
        (hit4)                       # never mark theta when already flat
    )

    clean["flag4"] = hit4.astype(int)
    clean["flag5"] = hit5.astype(int)

    # -- hysteresis to get stable stage-4 -------------------------------
    h = WIN_HYSTERESIS
    flag4_arr = clean.flag4.values
    stable4   = np.zeros_like(flag4_arr)

    # rolling count of consecutive 1s
    run = 0
    for i, f in enumerate(flag4_arr):
        run = run + 1 if f else 0
        if run >= h:
            stable4[i] = 1
        if not f:
            run = 0
    clean["stable4"] = stable4

    # -- initial prediction array (flat + theta) -------------------------
    pred = np.zeros_like(flag4_arr)
    pred[stable4 == 1] = 4
    pred[clean.flag5.values == 1] = 5  # provisional theta

    # -- enforce “theta only after flat” rule ----------------------------
    for i in range(len(pred)):
        if pred[i] == 5:
            # look back WIN_HYSTERESIS epochs
            window_start = max(0, i - h)
            if 4 not in pred[window_start:i]:
                pred[i] = 0               # discard this theta epoch

    clean["stage_pred"] = pred

    # merge back into the full DataFrame
    met = met.join(clean["stage_pred"], how="left").fillna({"stage_pred": 0})
    met["stage_pred"] = met["stage_pred"].astype(int)
    return met


def quick_plot(met, title):
    """Diagnostic plot with x-axis in seconds-since-start and a θ/α panel."""
    # relative time in seconds
    t_rel = (met.index - met.index[0]).total_seconds().to_numpy()
    mask_good = ~met.is_moving.values        # movement-free epochs

    fig, ax = plt.subplots(5, 1, figsize=(14, 10), sharex=True,
                           gridspec_kw=dict(hspace=0.35))
    fig.suptitle(title, fontsize=14)

    # 0) accelerometer RMS -------------------------------------------------
    ax[0].plot(t_rel, met.move_rms_g, lw=0.7)
    ax[0].axhline(ACC_RMS_THRESH_G, color="r", ls="--", lw=0.8)
    ax[0].set_ylabel("Move\nRMS (g)")

    # 1) EEG RMS (clean only) ---------------------------------------------
    ax[1].plot(t_rel[mask_good], met.rms_uv.values[mask_good], lw=0.7)
    ax[1].axhline(RMS_FLAT_THRESH, color="r", ls="--", lw=0.8)
    ax[1].set_ylabel("RMS µV")

    # 2) spectral flatness -------------------------------------------------
    ax[2].plot(t_rel[mask_good], met.flatness.values[mask_good], lw=0.7)
    ax[2].axhline(FLATNESS_THRESH, color="r", ls="--", lw=0.8)
    ax[2].set_ylabel("Flatness")

    # 3) θ/α ratio ---------------------------------------------------------
    ax[3].plot(t_rel[mask_good],
               met.theta_alpha_ratio.values[mask_good], lw=0.7)
    ax[3].axhline(THETA_ALPHA_RATIO4, color="g", ls="--", lw=0.8,
                  label="stage-4 upper")
    ax[3].axhline(THETA_ALPHA_RATIO5, color="r", ls="--", lw=0.8,
                  label="stage-5 lower")
    ax[3].set_ylabel("θ / α")
    ax[3].legend()

    # 4) stage predictions -------------------------------------------------
    ax[4].step(t_rel, met.stage_pred, where="mid")
    ax[4].set_ylabel("Stage")
    ax[4].set_xlabel("Time (s)")

    fig.savefig(f"{title.replace(' ', '_')}.png", bbox_inches="tight")
    plt.show()


def main(folder):
    csv_files = sorted(
        f for f in glob.glob(os.path.join(folder, "*.csv"))
        if "_pred" not in os.path.basename(f)
    )
    if not csv_files:
        print("No CSV files found in", folder)
        return

    for path in csv_files:
        name = os.path.basename(path)
        print("Processing", name)
        df = load_mindmonitor_csv(path)

        missing = []
        for ch in ("RAW_AF7", "RAW_AF8",
                   "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"):
            if ch not in df.columns:
                missing.append(ch)
        if missing:
            warnings.warn(f"{name}: missing {missing}; skipping file")
            continue

        # sampling rate
        diffs = np.diff(df.index.view(np.int64) / 1e9)
        diffs = diffs[diffs > 0]
        fs = 1.0 / np.median(diffs) if len(diffs) else 256.0

        met = epoch_features(df, fs)
        met = smooth_metrics(met)       # ← NEW
        met = stage_prediction(met)


        out_csv = path.replace(".csv", "_pred.csv")
        met.to_csv(out_csv)
        print("  → saved predictions to", os.path.basename(out_csv))

        # comment out next line to disable plots
        quick_plot(met, f"Front-avg metrics + Hori stages – {name}")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
