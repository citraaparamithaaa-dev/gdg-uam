"""Make a self-contained, versioned notebook from the selected verified entry."""
import argparse
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MATERIALS=None
parser=argparse.ArgumentParser()
parser.add_argument('--candidate',type=Path,default=ROOT)
parser.add_argument('--evidence',type=Path,default=ROOT/'evidence/latest/github_checkout_contract.json')
parser.add_argument('--output',type=Path,default=ROOT/'verify_colab.ipynb')
parser.add_argument('--materials',type=Path,required=True)
args=parser.parse_args()
MATERIALS=args.materials.resolve()
candidate=args.candidate.resolve()
evidence=json.loads(args.evidence.read_text())
assert evidence['all_repetitions_under_10_seconds']
assert evidence['setup_under_120_seconds']
requirements=[line.strip() for line in (candidate/'requirements.txt').read_text().splitlines() if line.strip()]
assert requirements and all(re.fullmatch(r'(numpy|numba)==[0-9.]+',r) for r in requirements)
files={f'candidate/{p.name}':p.read_bytes() for p in candidate.iterdir() if p.is_file() and p.suffix in ('.py','.npz')}
files['candidate/requirements.txt']=(candidate/'requirements.txt').read_bytes()
for name in ('harness.py','competition_api.py','data/test_demo.npz'):
    files[name]=(MATERIALS/name).read_bytes()
files['verify_contract.py']=(ROOT/'training/check_submission_contract.py').read_bytes()
files['local_reference.json']=args.evidence.read_bytes()
manifest={name:hashlib.sha256(content).hexdigest() for name,content in files.items()}
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for name,content in files.items():
        z.writestr(name,content)
payload=base64.b64encode(buffer.getvalue()).decode()
literal='(\n'+''.join(repr(payload[i:i+4096])+'\n' for i in range(0,len(payload),4096))+')'
identity=hashlib.sha256(json.dumps({n:d for n,d in manifest.items() if n.startswith('candidate/')},sort_keys=True).encode()).hexdigest()
def md(text):
    return dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True))
def code(text,form=False):
    return dict(cell_type='code',metadata={'cellView':'form'} if form else {},execution_count=None,outputs=[],source=text.splitlines(keepends=True))
cells=[md(f'''# Verified competition model — CPU check

**Local reference: {evidence['correct']}/400 correct ({evidence['accuracy']:.2%}).** This is the selected verified candidate from the completed experiments. Model identity: `{identity[:16]}`.

Choose **Runtime → Change runtime type → CPU**, connect, then **Runtime → Run all**. Setup may take several minutes; the 10-second target applies to all 400 predictions after setup. This notebook loads trained weights and does not train a model.

The notebook contains the model, the public demo, and unmodified official evaluator. It installs model packages in a separate folder, checks all embedded file hashes, tests prediction safety and speed, and runs three official evaluation processes. It needs no GitHub login, Google Drive mounting, or paid runtime.

The demo has been reused during development. Its accuracy does not establish hidden-test accuracy or a winning leaderboard position. Colab results will be measured when you run this notebook; the numbers above were measured locally.

Official rules and materials: https://github.com/GDG-UAM/ml-contest-materials
'''),code(f'''#@title 1. Prepare isolated model packages
import os,sys,subprocess,json
from pathlib import Path
packages=Path('/content/gdg_packages_{identity[:10]}')
packages.mkdir(exist_ok=True)
requirements={requirements!r}
result=subprocess.run([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check','--no-input',
                       '--only-binary=:all:','--target',str(packages),'--upgrade','--ignore-installed',*requirements],
                      text=True,capture_output=True)
if result.returncode:
    print(result.stdout)
    print(result.stderr)
    result.check_returncode()
print('Model packages ready in their separate folder',flush=True)
''',True),code(f'''#@title 2. Restore the verified model and demo
import base64,hashlib,io,zipfile
workspace=Path('/content/gdg_check_{identity[:10]}')
workspace.mkdir(exist_ok=True)
payload={literal}
manifest={manifest!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
    assert set(archive.namelist())==set(manifest)
    for name,digest in manifest.items():
        content=archive.read(name)
        assert hashlib.sha256(content).hexdigest()==digest,name
        destination=workspace/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(content)
print('Verified and restored',len(manifest),'files',flush=True)
''',True),code('''#@title 3. Test all 400 predictions, missing data, and output safety
environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONPATH=str(packages))
subprocess.run([sys.executable,'verify_contract.py','--model','candidate/model.py','--report','colab_contract.json'],
               cwd=workspace,env=environment,check=True,timeout=240)
''',True),code('''#@title 4. Run the unmodified official evaluator three times
subprocess.run([sys.executable,'harness.py','evaluate','--model','candidate/model.py',
                '--source','file:data/test_demo.npz','--runs','3','--out','official_results'],
               cwd=workspace,env=environment,check=True,timeout=600)
''',True),code('''#@title 5. Show results and download the model plus evidence
contract=json.loads((workspace/'colab_contract.json').read_text())
runs=[json.loads(p.read_text()) for p in (workspace/'official_results').rglob('run_*.json')]
assert len(runs)==3
assert all(r['status']=='ok' and r['n_scored']==400 and
           sum(r[k] for k in ('n_exception','n_invalid','n_soft_timeout','n_hard_timeout'))==0 for r in runs)
summary=dict(correct=contract['correct'],samples=400,accuracy=contract['accuracy'],
             all_400_prediction_seconds=contract['full_demo_prediction_seconds'],
             median_prediction_seconds=contract['median_full_demo_seconds'],
             under_10_seconds=contract['all_repetitions_under_10_seconds'],
             model_setup_seconds=contract['setup_seconds'],
             official_scores=[r['score'] for r in runs],
             official_setup_seconds=[r['setup_time_s'] for r in runs],official_runs_passed=3,
             zero_exceptions_and_timeouts=True)
(workspace/'COLAB_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
import shutil
from google.colab import files
shutil.make_archive(str(workspace/'submission'),'zip',workspace/'candidate')
result_zip=shutil.make_archive('/content/BEST_MODEL_COLAB_RESULTS','zip',workspace)
files.download(result_zip)
''',True)]
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python'),colab=dict(name=args.output.name)),cells=cells)
args.output.write_bytes(json.dumps(notebook,indent=1).encode('utf-8'))
print(json.dumps(dict(notebook=str(args.output.resolve()),bytes=args.output.stat().st_size,model=candidate.name,
    local_demo_accuracy=evidence['accuracy'],identity=identity,embedded_files=len(files)),indent=2))
