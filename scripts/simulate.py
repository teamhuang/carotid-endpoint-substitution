# -*- coding: utf-8 -*-
"""How does the mechanism of endpoint-related reclassification
affect prediction-model performance metrics?  Vectorised implementation.

Motivation. The empirical analysis establishes an exact algebraic result for
calibration-in-the-large (a function of the positive count for a fixed prediction set) but
cannot answer the question the identity leaves open: under what conditions do AUC, the
recalibration slope and the Brier score stay stable, and when do they move?

Design. The outcome Y is drawn from P(Y=1) = expit(b0 + b1*lp) so it genuinely depends on the
score. b1 (the score effect) is calibrated by bisection (calibrate_sim_beta.py) so the REALISED
AUC equals the labelled base AUC exactly at prevalence 0.646 (the analytic binormal shortcut
under-realises by ~0.01 in this generator); b0 is calibrated for the base prevalence. The second
endpoint Y* is generated FIVE mechanisms; M4 lowers prevalence at the chosen base rate:

  M1  risk-ordered expansion  -- negatives flipped in DESCENDING predicted risk (the
                                 reclassified are the highest-risk negatives; resembles the
                                 empirical segment-only group, mean predicted risk 0.611)
  M2  random expansion        -- negatives flipped at random (pure prevalence)
  M3  low-risk expansion      -- negatives flipped in ASCENDING predicted risk
  M4  non-differential misclassification -- a (sensitivity, specificity) pair applied to Y
  M5  latent-threshold shift  -- intercept shifted on the latent axis, outcome re-drawn

Factors. prevalence increase dpi in {0.10, 0.223, 0.30}; base AUC in {0.63, 0.75}; n in {400,
2000}; 2,000 repetitions per cell with a fixed seed.

Fast solvers (verified against statsmodels in a self-test):
  cal_large  : bisection on  sum_i expit(lp_i + a) = k      (monotone in a)
  cal_slope  : logistic regression of y on [1, logit(p)]

Writes results/simulation.json.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
import statsmodels.api as sm
from simulation_core import cal_large_batch, cal_slope_one, brier_batch, scaled_brier_batch, make_lp, draw_outcome, expand, misclass
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

SEED = 20261116
REPS = 2000
GRID = [(0.10, 0.63, 400), (0.10, 0.63, 2000), (0.10, 0.75, 400), (0.10, 0.75, 2000),
        (0.223, 0.63, 400), (0.223, 0.63, 2000), (0.223, 0.75, 400), (0.223, 0.75, 2000),
        (0.30, 0.63, 400), (0.30, 0.63, 2000), (0.30, 0.75, 400), (0.30, 0.75, 2000)]
BASE_PI = 0.646

# ---------------------------------------------------------------- fast solvers




# ---------------------------------------------------------------- self-test
rng = np.random.default_rng(7)
for n in (400, 2000):
    lp = rng.standard_normal(n).reshape(1,n)
    lp = np.repeat(lp, 10, axis=0)
    y = (rng.random(lp.shape) < 1 / (1 + np.exp(-(lp * 0.9)))).astype(int)
    a_fast = cal_large_batch(lp, y)[0]
    a_ref = float(sm.Logit(y[0].astype(float), np.ones((lp.shape[1], 1)), offset=lp[0]).fit(disp=0).params[0])
    assert abs(a_fast - a_ref) < 1e-8, f'FAST SOLVER MISMATCH (cal_large) at n={n}'

# ---------------------------------------------------------------- helpers












# ---------------------------------------------------------------- run

results = []
for dpi, auc0, n in GRID:
    print('Simulation cell',dpi,auc0,n,flush=True)
    lp, b1 = make_lp(n, auc0)
    lpB = np.repeat(lp[None, :], REPS, axis=0)
    r = np.random.default_rng(SEED + int(round(dpi * 1000)) + int(round(auc0 * 100)) + n)
    # calibrate the base prevalence
    lo, hi = -12.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if float((1 / (1 + np.exp(-(mid + b1 * lp)))).mean()) < BASE_PI: lo = mid
        else: hi = mid
    b0 = (lo + hi) / 2

    y1 = draw_outcome(lpB, b0, b1, r)
    target_k = np.full(REPS, int(round(n * (BASE_PI + dpi))))

    count_cache = {}
    def intercepts_for_counts(labels):
        counts = labels.sum(axis=1)
        for count in np.unique(counts):
            if count not in count_cache:
                count_cache[count] = brentq(lambda a: expit(lp+a).sum()-count, -40, 40)
        return np.array([count_cache[count] for count in counts])

    a_k1 = intercepts_for_counts(y1)

    m1 = {'auc': np.array([roc_auc_score(y1[i], lp) for i in range(REPS)]),
          'cal_large': a_k1,
          'slope': np.array([cal_slope_one(lp, y1[i]) for i in range(REPS)]),
          'brier': brier_batch(lpB, y1),
          'scaled_brier': scaled_brier_batch(lpB, y1)}

    def compare(y2, label, extra_dpi=None):
        a_k2 = intercepts_for_counts(y2)
        m2 = {'auc': np.array([roc_auc_score(y2[i], lp) for i in range(REPS)]),
              'cal_large': a_k2,
              'slope': np.array([cal_slope_one(lp, y2[i]) for i in range(REPS)]),
              'brier': brier_batch(lpB, y2),
              'scaled_brier': scaled_brier_batch(lpB, y2)}
        row = {'dpi': round(float(extra_dpi if extra_dpi is not None else dpi), 3),
               'target_dpi': dpi,
               'base_prevalence_mean': float(y1.mean()),
               'alternative_prevalence_mean': float(y2.mean()),
               'prevalence_change_mean': float(y2.mean()-y1.mean()),
               'base_auc': auc0, 'n': n, 'mechanism': label}
        row['usable_repetitions'] = {}
        for key, name in [('auc', 'dAUC'), ('slope', 'dSlope'), ('brier', 'dBrier'),
                          ('scaled_brier', 'dScaledBrier'), ('cal_large', 'dCITL')]:
            dv = m1[key] - m2[key]
            dv = dv[np.isfinite(dv)]
            row['usable_repetitions'][name] = len(dv)
            row[name] = (float(np.median(dv)), float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5)))
        row['dCITL_predicted_by_a_of_k'] = float(np.median(a_k1-a_k2))
        assert np.isfinite(row['prevalence_change_mean'])
        results.append(row)

    compare(expand(y1, lpB, target_k, 'desc', r), 'M1 risk-ordered expansion')
    compare(expand(y1, lpB, target_k, 'rand', r), 'M2 random expansion')
    compare(expand(y1, lpB, target_k, 'asc', r), 'M3 low-risk expansion')
    y4 = misclass(y1, 0.90, 0.90, r)
    compare(y4, 'M4 non-differential misclassification', extra_dpi=float((y4.mean(1) - y1.mean(1)).mean()))
    # M5: latent-threshold shift, re-drawn from a shifted intercept
    shift = np.log((BASE_PI + dpi) / (1 - BASE_PI - dpi)) - np.log(BASE_PI / (1 - BASE_PI))
    y5 = draw_outcome(lpB, b0 + shift, b1, r)
    compare(y5, 'M5 latent-threshold shift')

# ---------------------------------------------------------------- report
with (ROOT/'results/simulation.json').open('w', encoding='utf-8') as output:
    json.dump({'seed':SEED,'reps':REPS,'base_prevalence':BASE_PI,'grid':GRID,'results':results},
              output,indent=2,default=float)
print('Saved results/simulation.json',flush=True)
