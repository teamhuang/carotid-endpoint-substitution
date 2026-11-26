"""Reproduce the parameters in sim_beta_calibration.json on a large reference draw."""
import json
from pathlib import Path
from math import tan, pi
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

BASE_PREVALENCE = .646
ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(12345)
score = rng.standard_normal(400_000)
uniform = rng.random(len(score))

def reference_auc(beta):
    lower, upper = -12., 4.
    for _ in range(60):
        mid = (lower + upper) / 2
        if expit(mid + beta * score).mean() < BASE_PREVALENCE:
            lower = mid
        else:
            upper = mid
    intercept = (lower + upper) / 2
    label = uniform < expit(intercept + beta * score)
    return roc_auc_score(label, score), intercept

def main():
    frozen = json.loads((ROOT / 'sim_beta_calibration.json').read_text(encoding='utf-8'))
    estimates = {}
    for target in (.63, .75):
        start = abs(tan(pi / 2 * (2 * target - 1)))
        lower, upper = .5 * start, 2. * start
        for _ in range(40):
            mid = (lower + upper) / 2
            auc, _ = reference_auc(mid)
            if auc < target:
                lower = mid
            else:
                upper = mid
        beta = (lower + upper) / 2
        auc, intercept = reference_auc(beta)
        estimates[str(target)] = {'beta': round(beta, 4), 'realised_auc': round(auc, 4),
                                  'intercept_for_prev_0.646': round(intercept, 4)}
    print(json.dumps(estimates, indent=2))
    assert estimates == frozen, (estimates, frozen)

if __name__ == '__main__':
    main()
