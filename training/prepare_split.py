"""Create fixed 80/20 stratified train/validation datasets without modifying sources."""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from competition_api import load_dataset

ROOT = Path(__file__).resolve().parent


def main():
    ds = load_dataset(ROOT / "data/train_100k.npz")
    train, val = train_test_split(np.arange(len(ds)), test_size=.2, random_state=42,
                                  stratify=ds.labels)
    for name, ids in (("team_train", train), ("team_val", val)):
        target = ROOT / f"data/{name}.npz"
        np.savez(target, series=ds.series[ids], labels=ds.labels[ids], sample_index=ids,
                 **{f"meta__{k}": v[ids] for k, v in ds.meta.items()})
        print(name, len(ids), np.bincount(ds.labels[ids]).tolist(), flush=True)
    np.savez(ROOT / "data/team_split_ids.npz", train=train, val=val)
    (ROOT / "analysis").mkdir(exist_ok=True)
    (ROOT / "analysis/split.json").write_text(json.dumps({
        "seed": 42, "stratified": True, "training_rows": len(train),
        "validation_rows": len(val), "demo_used_for_training": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
