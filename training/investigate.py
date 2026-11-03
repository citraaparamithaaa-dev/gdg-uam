"""Training/demo distribution summaries and reproducible signal plots."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from competition_api import load_dataset

OUT = Path("analysis")
OUT.mkdir(exist_ok=True)
summary = {}
for name in ("train_100k", "test_demo"):
    path = Path("data") / f"{name}.npz"
    try:
        ds = load_dataset(path)
    except Exception:
        continue
    summary[name] = {"n": len(ds), "class_counts": np.bincount(ds.labels).tolist(), "classes": {}}
    fig, axs = plt.subplots(8, 2, figsize=(16, 19), sharex=True)
    for k in range(8):
        ids = np.flatnonzero(ds.labels == k)
        take = ids[:min(len(ids), 1000)]
        signals = np.asarray(ds.series[take])
        by_class = {
            "nan_fraction": float(np.isnan(signals).mean()),
            "channel_mean": np.nanmean(signals, axis=(0, 2)).tolist(),
            "channel_std": np.nanstd(signals, axis=(0, 2)).tolist(),
            "metadata": {},
        }
        for key, values in ds.meta.items():
            values = values[ids]
            if values.dtype.kind in "fiu":
                by_class["metadata"][key] = {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}
            else:
                v, n = np.unique(values, return_counts=True)
                by_class["metadata"][key] = dict(zip(v.tolist(), n.tolist()))
        summary[name]["classes"][str(k)] = by_class
        for j in range(2):
            row = ids[j * min(7, len(ids) - 1)]
            x = np.array(ds.series[row], dtype=float)
            for c in range(4):
                z = (x[c] - np.nanmean(x[c])) / (np.nanstd(x[c]) + 1e-8)
                axs[k, j].plot(z + 5*c, lw=.8, label=f"Channel {c}")
            axs[k, j].set_title(f"Class {k} | row {row}", loc="left", fontsize=10)
            axs[k, j].set_yticks([0,5,10,15], ["Ch 0", "Ch 1", "Ch 2", "Ch 3"])
            axs[k, j].grid(alpha=.15)
    axs[-1, 0].set_xlabel("Time step")
    axs[-1, 1].set_xlabel("Time step")
    fig.suptitle(f"{name}: signal shapes (each channel centered and scaled for display)", fontsize=15)
    fig.tight_layout(rect=[0,0,1,.98])
    fig.savefig(OUT / f"{name}_signals.png", dpi=120)
    plt.close(fig)
(OUT / "distribution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
