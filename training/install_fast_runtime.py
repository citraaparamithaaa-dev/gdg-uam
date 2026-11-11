"""Install the tested fast runtime around an exported 231-feature model."""
import argparse
import shutil
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--weights',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1]
expected=np.array(list(range(72))+list(range(144,186))+list(range(222,300))+list(range(378,417)))
with np.load(args.weights,allow_pickle=False) as packed:
    arrays={name:packed[name] for name in packed.files}
if not np.array_equal(arrays['indices'],expected):
    raise ValueError('This runtime requires the declared 231 core features in their original order')
arrays['kind']=np.asarray('core')
args.out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(args.out/'weights.npz',**arrays)
for name in ('model.py','core_features.py','fused_features.py','requirements.txt'):
    if (repo/name).resolve()!=(args.out/name).resolve():
        shutil.copyfile(repo/name,args.out/name)
print('Fast runtime installed:',args.out)
