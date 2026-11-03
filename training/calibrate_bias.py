"""Prioritize weak classes using a training-file calibration subset and official score.

No demo labels are used to learn class offsets. Fifteen thousand development rows
remain outside both classifier fitting and offset calibration.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from competition_api import load_dataset
from train_candidates import metrics

ROOT = Path(__file__).resolve().parent


def score(y, prediction):
    n=np.bincount(y,minlength=8)
    c=np.bincount(y[prediction==y],minlength=8)
    recall=c/np.maximum(n,1)
    terms=np.maximum(.01,np.sqrt(n)*(1.33*recall-.33)*recall**2)
    return float(np.mean(np.log(terms)))


def fit_bias(raw, labels):
    bias=np.zeros(8)
    best=score(labels,np.argmax(raw+bias,axis=1))
    for step in (1.,.5,.25,.125):
        for repeat in range(3):
            changed=False
            for k in range(8):
                origin=bias[k]
                chosen=origin
                for shift in (-step,step):
                    trial=bias.copy()
                    trial[k]=origin+shift
                    value=score(labels,np.argmax(raw+trial,axis=1))
                    if value>best+1e-10:
                        best=value
                        chosen=trial[k]
                if chosen!=origin:
                    bias[k]=chosen
                    changed=True
            bias-=bias.mean()
            if not changed:
                break
    return bias


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--out",default="balanced_core_200")
    args=parser.parse_args()
    source=ROOT/"experiments"/args.candidate
    target=ROOT/"experiments"/args.out
    target.mkdir(exist_ok=True)
    record=json.loads((source/"metrics.json").read_text())
    with (source/"development.pkl").open("rb") as f:
        saved=pickle.load(f)
    classifier,indices=saved["classifier"],saved["indices"]
    ds=load_dataset(ROOT/"data/train_100k.npz")
    split=np.load(ROOT/"data/team_split_ids.npz")
    cal,test=train_test_split(split["val"],train_size=5000,random_state=314,
                              stratify=ds.labels[split["val"]])
    X=np.load(ROOT/f"data/features_{record['feature_kind']}_train.npy")
    raw=classifier._raw_predict(X[cal][:,indices])
    bias=fit_bias(raw,ds.labels[cal])
    test_raw=classifier._raw_predict(X[test][:,indices])
    demo=load_dataset(ROOT/"data/test_demo.npz")
    D=np.load(ROOT/f"data/features_{record['feature_kind']}_demo.npy")[:,indices]
    result=dict(record)
    result.update(name=args.out,source_candidate=args.candidate,class_bias=bias.tolist(),
                  calibration_rows=len(cal),calibration_seed=314,validation_rows=len(test),
                  validation=metrics(ds.labels[test],np.argmax(test_raw+bias,axis=1)),
                  uncalibrated_matching_validation=metrics(ds.labels[test],np.argmax(test_raw,axis=1)),
                  demo=metrics(demo.labels,np.argmax(classifier._raw_predict(D)+bias,axis=1)))
    np.savez(target/"calibration_split_ids.npz",calibration=cal,validation=test)
    np.savez(ROOT/"data/team_calibrated_val.npz",series=ds.series[test],labels=ds.labels[test],
             sample_index=test,**{f"meta__{k}":v[test] for k,v in ds.meta.items()})
    saved["bias"]=bias
    with (target/"development.pkl").open("wb") as f:
        pickle.dump(saved,f)
    (target/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("name","class_bias","validation","uncalibrated_matching_validation","demo")},indent=2),flush=True)


if __name__=="__main__":
    main()
