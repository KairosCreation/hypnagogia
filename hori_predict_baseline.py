#!/usr/bin/env python
# hori_detect.py -----------------------------------------------------------
import sys, os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, detrend
from scipy.stats import gmean
import antropy as ant
import yasa

WIN_LEN, STEP = 10.0, 1
ACC_RMS_THRESH_G = 0.01
RMS_FLAT_THRESH = 40.0
FLATNESS_THRESH = 0.55
THETA_ALPHA_RATIO = 3
WIN_HYSTERESIS = 10
SMOOTH_WIN = 1
EPSILON = 1e-12
REQ_CHANNELS = ("RAW_AF7", "RAW_AF8", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z")
#REQ_CHANNELS = ("RAW_TP9", "RAW_TP10", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z")
def detect_spindles(signal, sf, ch_name="Front-Avg"):
    """Run YASA spindle detection on a 1D numpy EEG array."""
    sp = yasa.spindles_detect(signal, sf)
    if sp is None:
        return pd.DataFrame(), np.zeros_like(signal, dtype=bool)
    summary = sp.summary()
    mask = sp.get_mask()
    return summary, mask

def compute_spindles_per_minute(spindle_df, total_duration_sec, window_sec=60, step_sec=10):
    """Compute spindle density curve (spindles/min) using sliding windows."""
    if spindle_df.empty:
        return np.array([]), np.array([])
    spindle_peaks = spindle_df["Peak"].values
    windows = np.arange(0, total_duration_sec - window_sec + step_sec, step_sec)
    spm_values, win_mids = [], []
    for w_start in windows:
        w_end = w_start + window_sec
        count = np.sum((spindle_peaks >= w_start) & (spindle_peaks < w_end))
        spm = count / (window_sec / 60)
        spm_values.append(spm)
        win_mids.append(w_start + window_sec / 2)
    return np.array(win_mids), np.array(spm_values)

def bandpass(sig, lo, hi, fs, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)

def load_mindmonitor_csv(path):
    df = pd.read_csv(path)
    df = df[df["Elements"].isna()]
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    df = df.set_index("TimeStamp").sort_index()
    return df[~df.index.duplicated()]

def smooth_metrics(met, win=SMOOTH_WIN):
    cols = ["rms_uv", "flatness", "p_theta", "p_alpha", 'p_delta', "p_total",
            "theta_alpha_ratio", "move_rms_g", "lz_complexity", 'sample_entropy']
    met[cols] = met[cols].rolling(window=win, center=True, min_periods=1).mean()
    return met

def epoch_features(df, fs):
    nper = int(WIN_LEN * fs)
    nstep = int(STEP * fs)
    starts = np.arange(0, len(df) - nper, nstep, dtype=int)
    front_avg = df[["RAW_AF7", "RAW_AF8"]].mean(axis=1).to_numpy()
    ax, ay, az = [df[c].to_numpy() for c in ("Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z")]
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

    rows = []
    for i0 in starts:
        i1 = i0 + nper
        t_mid = df.index[i0:i1].mean()
        am_seg = detrend(accel_mag[i0:i1])
        move_rms = np.sqrt(np.mean(am_seg**2))
        is_moving = move_rms > ACC_RMS_THRESH_G

        seg_filt = bandpass(front_avg[i0:i1], 2, 30, fs)
        rms_uv = np.sqrt(np.mean(seg_filt**2))
        f, Pxx = welch(seg_filt, fs=fs, nperseg=nper // 2)
        band = (f >= 2) & (f <= 30)
        f_b, P_b = f[band], Pxx[band]
        flatness = gmean(P_b) / P_b.mean()

        def bpow(lo, hi):
            m = (f_b >= lo) & (f_b <= hi)
            return np.trapz(P_b[m], f_b[m])
        p_theta = bpow(4, 7)
        p_alpha = bpow(8, 12)
        p_delta = bpow(2, 4)
        p_total = np.trapz(P_b, f_b)
        theta_alpha = p_theta / (p_alpha + EPSILON)

        seg_norm = (seg_filt - seg_filt.mean()) / (seg_filt.std() + EPSILON)
        seg_bin = (seg_norm > np.median(seg_norm)).astype(int)
        lzc = ant.lziv_complexity(seg_bin, normalize=True)
        seg_filt = np.asarray(seg_filt, dtype=np.float32)
        sampent = ant.sample_entropy(seg_filt)

        rows.append(dict(time=t_mid, rms_uv=rms_uv, flatness=flatness, p_delta=p_delta,
                         p_theta=p_theta, p_alpha=p_alpha, p_total=p_total,
                         theta_alpha_ratio=theta_alpha,
                         move_rms_g=move_rms, is_moving=is_moving,
                         lz_complexity=lzc, sample_entropy=sampent))
    return pd.DataFrame(rows).set_index("time")

def compute_zscores(met, baseline):
    means = baseline.mean()
    stds = baseline.std(ddof=0) + EPSILON
    return (met - means) / stds

def stage_prediction(met):
    clean = met[~met.is_moving].copy()
    hit5 = (clean.theta_alpha_ratio > THETA_ALPHA_RATIO)
    pred = np.zeros(len(clean), dtype=int)
    pred[hit5] = 5
    met.loc[clean.index, "stage_pred"] = pred
    met["stage_pred"] = met["stage_pred"].fillna(0)
    return met

def process_file(path, minutes=None):
    name = os.path.basename(path)
    print("Processing", name)
    df = load_mindmonitor_csv(path)
    missing = [ch for ch in REQ_CHANNELS if ch not in df.columns]
    if missing:
        warnings.warn(f"{name}: missing {missing}; skipping file"); return
    fs = 256.0
    if minutes is not None and minutes > 0:
        n_samples = int(minutes * 60 * fs)
        df = df.iloc[:n_samples]
    met = smooth_metrics(epoch_features(df, fs))

    # 1. Get only epochs without movement for baseline
    no_movement = met[~met.is_moving]
    if no_movement.empty:
        raise RuntimeError("No stationary (non-movement) epochs found for baseline.")
    nonmove_idx = no_movement.index
    window_size = 30  # seconds
    positions = met.index.get_indexer(nonmove_idx)
    start_idx = None
    for i in range(len(positions) - window_size + 1):
        if np.all(np.diff(positions[i:i+window_size]) == 1):
            start_idx = i
            break
    if start_idx is not None:
        baseline_idx = nonmove_idx[start_idx : start_idx + window_size]
        baseline = met.loc[baseline_idx]
        print(f"Baseline: {len(baseline)} consecutive non-movement epochs from {baseline.index[0]} to {baseline.index[-1]}")
    else:
        raise RuntimeError("Could not find 30 consecutive seconds of non-movement for baseline.")

    met_z = compute_zscores(met, baseline)
    met_z["is_moving"] = met["is_moving"]
    met_z = stage_prediction(met_z)
    maskcols = ["rms_uv", "flatness", 'p_delta', "p_theta", "p_alpha", "p_total",
                "theta_alpha_ratio", "lz_complexity", 'sample_entropy']
    for col in maskcols:
        met_z.loc[met_z["is_moving"], col] = np.nan
    out_csv = path.replace(".csv", "_zscored.csv")
    met_z.to_csv(out_csv)
    print(" → saved z-scored metrics to", os.path.basename(out_csv))

    # ------------ SPINDLE DETECTION SECTION -----------------------
    front_avg = df[["RAW_AF7", "RAW_AF8"]].mean(axis=1).to_numpy()
    front_avg = bandpass(front_avg, 2, 30, fs)
    times = np.arange(front_avg.size) / fs
    spindle_summary, spindle_mask = detect_spindles(front_avg, fs)

    spindle_csv = path.replace('.csv', '_spindles.csv')
    spindle_summary.to_csv(spindle_csv, index=False)
    print(f" → saved spindle summary to {os.path.basename(spindle_csv)}")
    print("Spindle summary (first 5):\n", spindle_summary.head())
    total_duration_sec = len(front_avg) / fs
    spm_x, spm_y = compute_spindles_per_minute(
        spindle_summary, total_duration_sec, window_sec=60, step_sec=10
    )

    # ------------ Main Metrics & Hori Stage Plot (+ SPM) --------------------
    colors = {
        "move_rms_g": "#cccccc",            # gray
        "rms_uv": "#92bce6",                # pastel blue
        "flatness": "#c8c8c8",              # pastel grey
        'p_delta': "#92b4f2",    
        "p_theta": "#f29292",               # pastel red
        "p_alpha": "#b6e3a8",               # pastel green
        "p_total": "#d4b0e5",               # pastel purple
        "theta_alpha_ratio": "#f2e6a7",     # pastel yellow
        "lz_complexity": "#f2b5d4",         # pastel pink
        'sample_entropy':"#a54776", 
        "stage_pred": "black"
    }
    t0 = met_z.index[0]
    elapsed_min = (met_z.index - t0).total_seconds() / 60

    # Interpolate SPM curve onto the metrics timebase
    if len(spm_x) > 1:
        from scipy.interpolate import interp1d
        spm_interp = interp1d(spm_x/60, spm_y, bounds_error=False, fill_value=0)
        spm_on_elapsed = spm_interp(elapsed_min)
    else:
        spm_on_elapsed = np.zeros_like(elapsed_min)

    # Cap the plots at the requested number of minutes
    if minutes is not None:
        mask_in_range = elapsed_min <= minutes
    else:
        mask_in_range = np.full(elapsed_min.shape, True)

    fig, axs = plt.subplots(12, 1, figsize=(14, 20), sharex=True)
    axs[0].plot(elapsed_min[mask_in_range], met_z["move_rms_g"][mask_in_range], color=colors["move_rms_g"], label="Acc RMS")
    axs[0].axhline(ACC_RMS_THRESH_G, color="r", ls="--", lw=1, label="ACC thresh")
    axs[0].set_ylabel("Move RMS (g)")
    axs[1].plot(elapsed_min[mask_in_range], met_z["rms_uv"][mask_in_range], color=colors["rms_uv"], label="RMS µV")
    axs[1].set_ylabel("RMS µV (z)")
    axs[2].plot(elapsed_min[mask_in_range], met_z["flatness"][mask_in_range], color=colors["flatness"], label="Flatness")
    axs[2].set_ylabel("Flatness (z)")
    axs[3].plot(elapsed_min[mask_in_range], met_z["p_delta"][mask_in_range], color=colors["p_delta"], label="Delta")
    axs[3].set_ylabel("δ (z)")
    axs[4].plot(elapsed_min[mask_in_range], met_z["p_theta"][mask_in_range], color=colors["p_theta"], label="Theta")
    axs[4].set_ylabel("θ (z)")
    axs[5].plot(elapsed_min[mask_in_range], met_z["p_alpha"][mask_in_range], color=colors["p_alpha"], label="Alpha")
    axs[5].set_ylabel("α (z)")
    axs[6].plot(elapsed_min[mask_in_range], met_z["p_total"][mask_in_range], color=colors["p_total"], label="Total Pow")
    axs[6].set_ylabel("ΣP (z)")
    axs[7].plot(elapsed_min[mask_in_range], met_z["theta_alpha_ratio"][mask_in_range], color=colors["theta_alpha_ratio"], label="θ/α")
    axs[7].set_ylabel("θ/α (z)")
    axs[8].plot(elapsed_min[mask_in_range], met_z["lz_complexity"][mask_in_range], color=colors["lz_complexity"], label="LZc")
    axs[8].set_ylabel("LZc (z)")
    axs[9].step(elapsed_min[mask_in_range], met_z["sample_entropy"][mask_in_range], where="mid", color=colors["sample_entropy"], label="SampEnt")
    axs[9].set_ylabel("SampEnt")
    axs[10].step(elapsed_min[mask_in_range], met_z["stage_pred"][mask_in_range], where="mid", color=colors["stage_pred"], label="Stage")
    axs[10].set_ylabel("Stage")

    # SPM plottage
    axs[11].plot(elapsed_min[mask_in_range], spm_on_elapsed[mask_in_range], marker='o', color='navy', lw=2, label="Spindles/min (window=60s)")
    axs[11].set_ylabel("Spindles/min")
    axs[11].set_xlabel("Elapsed Time (min)")
    axs[11].legend(loc="upper right")

    # Hide empty legend if no spindles
    for i in range(11):
        axs[i].legend(loc="upper right")
    if minutes is not None:
        for ax in axs:
            ax.set_xlim(0, minutes)
    plt.suptitle(f"Front-avg metrics (z, baseline stationary) + Hori stages + Spindle Density\n{name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def main(folder, minutes=None):
    csv_files = sorted(
        f for f in glob.glob(os.path.join(folder, "*.csv"))
        if "_pred" not in os.path.basename(f)
        and "_zscored" not in os.path.basename(f)
        and "_spindles" not in os.path.basename(f)
    )
    if not csv_files:
        print("No CSV files found in", folder); return
    for path in csv_files:
        process_file(path, minutes)

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python hori_detect.py <folder_with_csvs> [minutes]"); sys.exit(1)
    minutes_arg = float(sys.argv[2]) if len(sys.argv) == 3 else None
    if minutes_arg is not None and minutes_arg <= 0:
        print("Minutes must be > 0"); sys.exit(1)
    main(sys.argv[1], minutes_arg)
