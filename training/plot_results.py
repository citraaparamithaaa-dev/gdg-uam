"""Plot measured validation and demo results; preserve the small demo sample size."""
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
records = [json.loads(p.read_text()) for p in sorted((ROOT / "experiments").glob("*/metrics.json"))]
labels = [r["name"].replace("_", " ") for r in records]
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(records))
ax.bar(x-.18, [r["validation"]["accuracy"]*100 for r in records], .36, label="Development check (15,000-20,000)", color="#2563eb")
ax.bar(x+.18, [r["demo"]["accuracy"]*100 for r in records], .36, label="Demo distribution (400)", color="#f59e0b")
ax.set_xticks(x, labels, rotation=12)
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Feature experiments on separate validation and demo data", loc="left")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=.15)
fig.tight_layout()
fig.savefig(ROOT / "analysis/model_comparison.png", dpi=160)
plt.close(fig)
with (ROOT/"analysis/experiments.csv").open("w",newline="") as stream:
    writer=csv.writer(stream)
    writer.writerow(["name","check_rows","check_accuracy","demo_accuracy","check_score","demo_score","demo_min_recall","training_seconds"])
    for r in records:
        writer.writerow([r["name"],r["validation_rows"],r["validation"]["accuracy"],r["demo"]["accuracy"],
                         r["validation"]["finite_file_score"],r["demo"]["finite_file_score"],
                         min(r["demo"]["recall"]),r["training_seconds"]])
