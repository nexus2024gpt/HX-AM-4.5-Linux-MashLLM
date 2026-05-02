# run_stress_batch.ps1
$script = @'
from math_core import MathCore
import json
from pathlib import Path

core = MathCore()
for f in sorted(Path('artifacts').glob('*.json')):
    if 'graph' in f.stem or 'hyx-portal' in f.name:
        continue
    sim_path = Path('sim_results') / f'{f.stem}_stress.json'
    if sim_path.exists():
        continue
    art = json.loads(f.read_text())
    fd = art.get('data', {}).get('gen', {}).get('four_d_matrix')
    if fd:
        print(f'Testing {f.stem}...', end=' ', flush=True)
        r = core.stress_test(f.stem, fd)
        core.index_artifact(f.stem, fd, art.get('data', {}).get('domain', '?'), r['stability_score'])
        print(f'score={r["stability_score"]} model={r["model_used"]}')
'@

python -c "$script"