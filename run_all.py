"""Reproduce v16 results and publication figures using the two archived source files."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
(root / 'results').mkdir(exist_ok=True)
for step in ['analyze_paired', 'analyze_support', 'analyze_nomogram', 'analyze_imputation',
             'simulate', 'verify_identity', 'make_figures']:
    print(f'Running {step}', flush=True)
    subprocess.run([sys.executable, str(root / 'scripts' / f'{step}.py')], cwd=root, check=True)
