"""Verify prediction contract and time the whole demo using only NumPy/stdlib."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path
import numpy as np
from competition_api import Sample,load_dataset

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--report',type=Path,required=True)
args=parser.parse_args()
folder=args.model.resolve().parent
sys.path.insert(0,str(folder))
def deny_network(*args,**kwargs):
    raise AssertionError('Model attempted network access')
class OfflineSocket(socket.socket):
    connect=deny_network
    connect_ex=deny_network
socket.socket=OfflineSocket
socket.create_connection=deny_network
start=time.perf_counter()
spec=importlib.util.spec_from_file_location('checked_submission',folder/'model.py')
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
model=module.Model()
setup=time.perf_counter()-start
ds=load_dataset(ROOT/'data/test_demo.npz')
samples=list(ds)
durations=[]
predictions=None
for repeat in range(3):
    start=time.perf_counter()
    answers=[model.predict(sample) for sample in samples]
    durations.append(time.perf_counter()-start)
    assert all(type(p) is int and 0<=p<=7 for p in answers)
    if predictions is not None:
        assert predictions==answers,'Predictions changed between repetitions'
    predictions=answers
edge_arrays=[np.zeros((4,256),np.float32),np.ones((4,256),np.float32),
             np.full((4,256),np.nan,np.float32),samples[0].series.copy(),samples[0].series.copy(),samples[0].series.copy()]
edge_arrays[3][:,20:62]=np.nan
edge_arrays[4][0]=np.nan
edge_arrays[5][:,0]=np.nan
edge_arrays[5][:,-1]=np.nan
for array in edge_arrays:
    before=array.copy()
    sample=Sample(array,{})
    result=model.predict(sample)
    assert type(result) is int and 0<=result<=7
    assert result==model.predict(sample)
    assert np.array_equal(before,array,equal_nan=True),'Input modified'
for i in range(100):
    original=samples[i]
    changed=Sample(original.series,{'sample_rate':999.,'group':'unseen','level':-999.})
    assert model.predict(original)==model.predict(changed),'Unexpected metadata dependency'
labels=ds.labels
predictions=np.asarray(predictions)
confusion=np.zeros((8,8),np.int64)
np.add.at(confusion,(labels,predictions),1)
n=confusion.sum(axis=1)
c=np.diag(confusion)
recall=c/n
terms=np.maximum(.01,np.sqrt(n)*(1.33*recall-.33)*recall**2)
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
report=dict(model=str(folder/'model.py'),python=sys.version.split()[0],numpy=np.__version__,
    scored_samples=len(samples),accuracy=float(np.mean(labels==predictions)),correct=int(np.sum(labels==predictions)),
    class_recall=recall.tolist(),confusion_matrix=confusion.tolist(),finite_demo_score=float(np.exp(np.log(terms).mean())),
    setup_seconds=setup,full_demo_prediction_seconds=durations,median_full_demo_seconds=float(np.median(durations)),
    all_repetitions_under_10_seconds=all(t<10 for t in durations),setup_under_120_seconds=setup<120,
    valid_python_integer_outputs=True,immutable_inputs=True,deterministic=True,edge_cases_passed=len(edge_arrays),
    metadata_independence_checks=100,network_disabled_during_import_setup_and_prediction=True,
    file_sha256=files,largest_file_bytes=max(p.stat().st_size for p in folder.iterdir() if p.is_file()))
args.report.parent.mkdir(parents=True,exist_ok=True)
args.report.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('file_sha256','confusion_matrix')},indent=2),flush=True)
assert report['all_repetitions_under_10_seconds'],report
assert report['setup_under_120_seconds'],report
assert report['largest_file_bytes']<100_000_000
