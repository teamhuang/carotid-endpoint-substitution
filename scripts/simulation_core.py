"""Shared simulation functions; importing this module does not run simulations."""
from pathlib import Path
import json
import warnings
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
ROOT=Path(__file__).resolve().parents[1]
SEED=20261116
def cal_large_batch(lp, y):
    """calibration-in-the-large for many rows/reps at once.

    lp : (reps, n) linear predictors (fixed within a cell across the two endpoints)
    y  : (reps, n) 0/1
    returns (reps,) intercept a with sum_i expit(lp+a) = sum_i y_i
    Bisection: expit(lp+a) is strictly increasing in a, so the root is unique.
    """
    k = y.sum(1, keepdims=True)
    lo = np.full((lp.shape[0], 1), -30.0)
    hi = np.full((lp.shape[0], 1), 30.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        s = (1 / (1 + np.exp(-(lp + mid)))).sum(1, keepdims=True)
        too_small = s < k
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    return ((lo + hi) / 2).ravel()

def cal_slope_one(lp_row, y_row):
    """Free recalibration slope for ONE rep: coefficient of lp in the logistic model
    y ~ [1, lp].  Uses statsmodels (battle-tested); note that logit(expit(lp)) == lp, so this
    is exactly the regression of y on logit(p) used by the manuscript pipeline.
    ~1 ms per call, so the grid is trimmed to keep total runtime at a few minutes."""
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            fit = sm.Logit(y_row.astype(float), sm.add_constant(lp_row)).fit(disp=0)
        except (np.linalg.LinAlgError, ValueError):
            return float('nan')
    if not fit.mle_retvals['converged'] or any(
        issubclass(w.category, (ConvergenceWarning, PerfectSeparationWarning)) for w in notices
    ):
        return float('nan')
    return float(fit.params[1])

def brier_batch(lp, y):
    """Brier score for many reps at once: mean (p - y)^2."""
    pr = 1 / (1 + np.exp(-lp))
    return ((pr - y) ** 2).mean(1)

def scaled_brier_batch(lp, y):
    """Scaled Brier (Brier skill score) vs the intercept-only model, per rep."""
    bs = brier_batch(lp, y)
    bs_null = ((y.mean(1)[:, None] - y) ** 2).mean(1)
    return np.divide(bs_null - bs, bs_null, out=np.full_like(bs, np.nan), where=bs_null > 0)

def make_lp(n, target_auc):
    """Standard-normal score and the outcome-linked effect.

    beta is READ FROM sim_beta_calibration.json (produced by calibrate_sim_beta.py by
    bisection so that the REALISED AUC of expit(b0 + beta*Z) at prevalence 0.646 equals the
    labelled target exactly). The earlier analytic shortcut a = |tan(pi/2*(2*AUC-1))|
    under-realised the AUC by ~0.01 (0.617/0.741) because the outcome is logistic in a normal
    score, not binormal-equal-variance; the calibrated values (0.4785 for 0.63, 1.0521 for
    0.75) realise 0.6300/0.7500 to four decimals on 400k draws.
    """
    z = np.random.default_rng(SEED + n).standard_normal(n)
    cal = json.load(open(ROOT / 'sim_beta_calibration.json', encoding='utf-8'))
    a = float(cal[str(target_auc)]['beta'])
    return z, a

def draw_outcome(lp, b0, b1, r):
    pr = 1 / (1 + np.exp(-(b0 + b1 * lp)))
    return (r.random(lp.shape) < pr).astype(int)

def expand(y, lp, k, how, r):
    """Flip negatives to positive in descending/ascending/random predicted-risk order."""
    yy = y.copy()
    out = []
    for i in range(yy.shape[0]):
        row = yy[i].copy()
        neg = np.where(row == 0)[0]
        need_i = k[i] - row.sum()
        if need_i <= 0:
            # prevalence decrease: sample k positives at random
            pos = np.where(row == 1)[0]
            drop = r.choice(pos, size=min(int(row.sum() - k[i]), len(pos)), replace=False)
            row[drop] = 0
        else:
            if how == 'rand':
                pick = r.choice(neg, size=min(need_i, len(neg)), replace=False)
            elif how == 'desc':
                pick = neg[np.argsort(-lp[i][neg])][:min(need_i, len(neg))]
            else:
                pick = neg[np.argsort(lp[i][neg])][:min(need_i, len(neg))]
            row[pick] = 1
            short = k[i] - row.sum()
            if short > 0:
                left = neg[~np.isin(neg, pick)]
                if len(left):
                    row[r.choice(left, size=min(short, len(left)), replace=False)] = 1
        out.append(row)
    return np.array(out)

def misclass(y, sens, spec, r):
    keep = r.random(y.shape) < np.where(y == 1, sens, spec)
    return np.where(keep, y, 1 - y).astype(int)
