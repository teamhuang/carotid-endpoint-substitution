# -*- coding: utf-8 -*-
"""JCE v15 figure suite.

Nature-style output: Arial 8 pt, panel letters 10 pt bold (combined figures only),
600 dpi PNG + editable vector PDF (Type 42 fonts), no SVG.

Independent panels -> figures/panels/  (no panel letters)
Combined figures   -> figures/combined/ (A/B/C letters)
Plotting source    -> figures/source/  (this script) + figures/README.md

Data sources (frozen): analysis_results_v14_corrected.json (paired endpoint analysis),
analysis_results_v11_sim.json (simulation grid, seed 20261116), harmonized_A/B.csv
(descriptives, spline ORs, distributions). No smoothing or imputation is applied here.
"""
import json
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
from patsy import dmatrix
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT
# frozen result JSONs live in results/ inside this package
RES = ROOT / 'results'
PANELS = ROOT / 'figures' / 'panels'
COMBINED = ROOT / 'figures' / 'combined'
SOURCE = ROOT / 'figures' / 'source'
for d in (PANELS, COMBINED, SOURCE):
    d.mkdir(parents=True, exist_ok=True)

rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.dpi': 600,
})

BLUE, VERM, ORANGE, GREY = '#0072B2', '#D55E00', '#E69F00', '#666666'


def save(fig, name):
    fig.savefig(PANELS / f'{name}.png', bbox_inches='tight', facecolor='white')
    fig.savefig(PANELS / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'panel  {name}.png/.pdf')


def save_combined(fig, name):
    fig.savefig(COMBINED / f'{name}.png', bbox_inches='tight', facecolor='white', dpi=600)
    fig.savefig(COMBINED / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'combined {name}.png/.pdf')


def panel_letter(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top', ha='left')


# ----------------------------------------------------------------- frozen inputs
res = json.loads((RES / 'analysis_results_v14_corrected.json').read_text(encoding='utf-8'))
sim = json.loads((RES / 'analysis_results_v11_sim.json').read_text(encoding='utf-8'))
A = pd.read_csv(PKG / 'harmonized_A.csv')
B = pd.read_csv(PKG / 'harmonized_B.csv')
assert len(A) == 682 and len(B) == 463, (len(A), len(B))

PRED = ['age', 'male', 'BMI', 'SBP', 'smoke_ever', 'TC', 'HDL', 'TG', 'FPG', 'HbA1c']

# ================================================================ Figure 1: flow
def draw_flow(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    box = {'boxstyle': 'round,pad=0.55', 'facecolor': '#e8f2f8',
           'edgecolor': '#39789c', 'linewidth': 1.1}
    rows = [(0.84, 'PROLOGUE baseline cohort\nn = 463'),
            (0.52, 'Ten predictors observed\nn = 404'),
            (0.20, 'Both IMT endpoint fields observed\nn = 400')]
    for y, label in rows:
        ax.text(0.42, y, label, ha='center', va='center', fontsize=8, bbox=box)
    for y0, y1 in [(0.78, 0.60), (0.46, 0.28)]:
        ax.annotate('', xy=(0.42, y1), xytext=(0.42, y0),
                    arrowprops={'arrowstyle': '-|>', 'lw': 1.2, 'color': '#39789c'})
    ax.text(0.80, 0.69, '59 excluded:\nmissing predictor(s)', ha='center',
            va='center', fontsize=7.5)
    ax.text(0.80, 0.37, '4 excluded:\nno segment IMT value', ha='center',
            va='center', fontsize=7.5)
    ax.text(0.42, 0.045, 'Endpoint agreement set (both fields, agreement analysis only):\n453 participants',
            ha='center', va='center', fontsize=7, color='#333333')

fig, ax = plt.subplots(figsize=(3.4, 2.9), facecolor='white')
draw_flow(ax)
save(fig, 'fig1_flow')

fig, ax = plt.subplots(figsize=(4.6, 3.4), facecolor='white')
draw_flow(ax)
save_combined(fig, 'Figure1_flow_v15')

# ============================================== Figure 2: paired endpoint results
p = res['primary']

def draw_agreement(ax, title='Endpoint agreement (n = 453)'):
    matrix = np.array([[269, 17], [127, 40]])
    ax.imshow(matrix, cmap='Blues', vmin=0, vmax=269)
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            ax.text(j, i, f'{val}', ha='center', va='center', fontsize=10,
                    color='white' if val > 150 else '#17202a')
    ax.set_xticks([0, 1], ['Segment +', 'Segment −'])
    ax.set_yticks([0, 1], ['Designated +', 'Designated −'])
    ax.set_title(title, loc='left')

def draw_deltas(ax):
    names = ['ΔAUC', 'ΔBrier score', 'Δoffset\nintercept', 'Δrecalibration\nslope']
    subgrid = ax.get_subgridspec() if hasattr(ax, 'get_subgridspec') else None
    for i, (name, point) in enumerate(zip(names, p['delta'])):
        pass  # replaced by draw_deltas_panel below

def deltas_panel(fig, spec):
    labels = ['AUC', 'Brier score', 'Offset intercept', 'Recalibration slope']
    for i, label in enumerate(labels):
        sub = fig.add_subplot(spec[i, 0])
        point = p['delta'][i]
        lo, hi = p['bootstrap']['ci'][0][i], p['bootstrap']['ci'][1][i]
        sub.axvline(0, color='#777777', lw=0.7)
        sub.plot([lo, hi], [0, 0], color='#246b9a', lw=1.6)
        sub.plot([lo, lo], [-0.12, 0.12], color='#246b9a', lw=0.9)
        sub.plot([hi, hi], [-0.12, 0.12], color='#246b9a', lw=0.9)
        sub.scatter([point], [0], color='#246b9a', s=22, zorder=3)
        pad = max((hi - lo) * 0.18, 0.02)
        sub.set_xlim(min(0, lo) - pad, max(0, hi) + pad)
        sub.set_ylim(-0.3, 0.3)
        sub.set_yticks([])
        sub.set_title(f'{label}   {point:+.3f} ({lo:+.3f} to {hi:+.3f})',
                      fontsize=7, loc='left', pad=14)
        sub.spines[['top', 'right', 'left']].set_visible(False)
        sub.tick_params(axis='x', labelsize=6.5, pad=1)
        if i == 3:
            sub.set_xlabel('Designated − segment (metric-specific scale)', fontsize=7)

def draw_calibration(ax):
    ax.plot(p['counts'], [p['designated'][2], p['segment'][2]], '-o',
            color=VERM, lw=1.3, ms=5)
    ax.annotate(f'a({p["counts"][0]}) = {p["designated"][2]:+.3f}',
                (p['counts'][0], p['designated'][2]), textcoords='offset points',
                xytext=(8, 8), fontsize=7)
    ax.annotate(f'a({p["counts"][1]}) = {p["segment"][2]:+.3f}',
                (p['counts'][1], p['segment'][2]), textcoords='offset points',
                xytext=(-8, -16), fontsize=7, ha='right')
    ax.set_xlabel('Observed positive label count k')
    ax.set_ylabel('Offset calibration intercept')
    ax.set_title('Fixed score (n = 400): intercept determined by k', loc='left')
    ax.set_xticks(p['counts'])
    ax.set_xlim(p['counts'][0] - 45, p['counts'][1] + 45)
    ax.set_ylim(-0.25, 1.85)
    ax.grid(axis='y', alpha=0.2)

# independent panels
fig, ax = plt.subplots(figsize=(3.0, 2.7), facecolor='white')
draw_agreement(ax)
save(fig, 'fig2a_agreement')

fig = plt.figure(figsize=(3.6, 3.2), facecolor='white')
gs = fig.add_gridspec(4, 1, hspace=0.85)
deltas_panel(fig, gs)
save(fig, 'fig2b_deltas')

fig, ax = plt.subplots(figsize=(3.0, 2.7), facecolor='white')
draw_calibration(ax)
save(fig, 'fig2c_calibration')

# combined Figure 2 (A/B/C)
fig = plt.figure(figsize=(7.2, 3.4), facecolor='white')
grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.0], wspace=0.46)
axA = fig.add_subplot(grid[0, 0])
draw_agreement(axA, 'Endpoint agreement (n = 453)')
axA.text(-0.16, 1.14, 'A', transform=axA.transAxes, fontsize=10,
         fontweight='bold', va='top', ha='left')
sub = grid[0, 1].subgridspec(4, 1, hspace=1.05)
deltas_panel(fig, sub)
axB = fig.add_subplot(grid[0, 1])
axB.axis('off')
axB.text(-0.10, 1.42, 'B', transform=axB.transAxes, fontsize=10, fontweight='bold')
axC = fig.add_subplot(grid[0, 2])
draw_calibration(axC)
axC.text(-0.22, 1.10, 'C', transform=axC.transAxes, fontsize=10,
         fontweight='bold', va='top', ha='left')
fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.10)
save_combined(fig, 'Figure2_endpoint_substitution_v15')

# ===================================================== Figure 3: simulation grid
MECH_ORDER = ['M1 risk-ordered expansion', 'M2 random expansion',
              'M3 low-risk expansion', 'M4 non-differential misclassification',
              'M5 latent-threshold shift']
TICKS = ['M1\nhigh risk', 'M2\nrandom', 'M3\nlow risk', 'M4\nmisclass.*', 'M5\nlatent†']

rows = {}
for r in sim['results']:
    if r['n'] != 2000 or r['base_auc'] != 0.63:
        continue
    if r['mechanism'].startswith('M4'):
        if abs(r['dpi'] - 0.029) < 1e-3:     # realised +2.9-point scenario
            rows['M4 non-differential misclassification'] = r
    elif abs(r['dpi'] - 0.223) < 1e-8:
        rows[r['mechanism']] = r
assert len(rows) == 5, sorted(rows)
mechs = [rows[m] for m in MECH_ORDER]
assert sim['reps'] == 2000 and sim['seed'] == 20261116

PANELS3 = [('dAUC', 'ΔAUC', BLUE), ('dSlope', 'Δrecalibration slope', BLUE),
           ('dBrier', 'ΔBrier score', BLUE), ('dCITL', 'Δcalibration-in-the-large', ORANGE)]

def sim_panel(ax, key, name, color):
    pts = [m[key][0] for m in mechs]
    lo = [m[key][0] - m[key][1] for m in mechs]
    hi = [m[key][2] - m[key][0] for m in mechs]
    ax.errorbar(range(5), pts, yerr=[lo, hi], fmt='o', ms=4, capsize=2.5,
                lw=1.0, color=color, zorder=3)
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_xticks(range(5))
    ax.set_xticklabels(TICKS, fontsize=6.5)
    ax.set_ylabel(f'{name}\n(designated − alternative)', fontsize=7)

# independent panels
for key, name, color in PANELS3:
    fig, ax = plt.subplots(figsize=(3.4, 2.8), facecolor='white')
    sim_panel(ax, key, name, color)
    save(fig, f"fig3_{key.replace('d', '', 1).lower()}")

# combined Figure 3 (A–D)
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), facecolor='white')
for ax, (key, name, color) in zip(axes.ravel(), PANELS3):
    sim_panel(ax, key, name, color)
for ax, letter in zip(axes.ravel(), 'ABCD'):
    panel_letter(ax, letter)
fig.suptitle('Five reclassification mechanisms (n = 2,000; base AUC 0.63; 2,000 repetitions)',
             fontsize=8.5)
fig.text(0.01, 0.005,
         'M1–M3 and M5 target a +22.3-point prevalence increase; *M4 is a separate misclassification '
         'scenario (Se = Sp = 0.90) in which prevalence increased by ≈2.9 points; †M5 achieved prevalence '
         'varies with outcome re-draws.\nPoints and bars: medians and 2.5–97.5 percentiles of paired '
         'changes (designated − alternative).', fontsize=6.3, color='#444444')
fig.tight_layout(rect=[0, 0.07, 1, 0.965], h_pad=2.4)
save_combined(fig, 'Figure3_simulation_v15')

# ================================== Supplementary Figure S1: spline ORs (cohort A)
def spline_or(ax, var, unit):
    # Frozen convention (round4_analysis.py §B): UNADJUSTED cr(x, df=3) vs linear-term
    # logistic comparison on cohort A — this is the source of P = 0.029 (HDL) / 0.017 (age).
    d = A[[var, 'outcome_CAD']].dropna()
    full = dmatrix(f'cr({var}, df=3)', d, return_type='dataframe')
    m_full = sm.Logit(d['outcome_CAD'].astype(float), full).fit(disp=0)
    m_red = sm.Logit(d['outcome_CAD'].astype(float), sm.add_constant(d[[var]].astype(float))).fit(disp=0)
    lrt = 2 * (m_full.llf - m_red.llf)
    pv = float(stats.chi2.sf(lrt, 2))
    grid = np.linspace(np.percentile(d[var], 2), np.percentile(d[var], 98), 120)
    ref = float(np.median(d[var]))

    def pred(xv):
        frame = pd.DataFrame({var: xv})
        X = dmatrix(full.design_info, frame, return_type='dataframe')
        return np.asarray(m_full.predict(X))

    ors = pred(grid) / float(pred(np.array([ref])))
    ax.plot(grid, ors, color=BLUE, lw=1.4)
    ax.axhline(1, color='k', lw=0.6, ls='--')
    ax.axvline(ref, color=GREY, lw=0.6, ls=':')
    xlabel = f'{var} ({unit})' if unit else var
    ax.set_xlabel(xlabel)
    ax.set_ylabel('OR vs median')
    ax.set_title(f'Non-linearity P = {pv:.3f}', loc='left', fontsize=7)
    return pv

pv_hdl = None  # computed inside the paired panel below
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.2, 2.7), facecolor='white')
pv_hdl = spline_or(a1, 'HDL', 'mmol/L')
pv_age = spline_or(a2, 'age', 'years')
a1.set_title('HDL-C', loc='left')
a2.set_title('Age', loc='left')
fig.text(0.02, 0.93, f'Non-linearity P: HDL-C {pv_hdl:.3f}, age {pv_age:.3f} '
         '(2-df LRT, natural cubic spline, cohort A)', fontsize=6.3, color='#444444')
fig.tight_layout(rect=[0, 0, 1, 0.90])
save(fig, 'figS1_splines_pair')

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), facecolor='white')
pv_hdl = spline_or(a1, 'HDL', 'mmol/L')
pv_age = spline_or(a2, 'age', 'years')
a1.set_title('HDL-C', loc='left')
a2.set_title('Age', loc='left')
panel_letter(a1, 'A'); panel_letter(a2, 'B')
fig.suptitle('Natural cubic spline odds ratios (df = 3) versus cohort-A medians', fontsize=8.5)
fig.text(0.01, 0.005, f'2-df likelihood-ratio tests: HDL-C P = {pv_hdl:.3f}; age P = {pv_age:.3f}. '
         'Dotted lines mark cohort-A medians.', fontsize=6.3, color='#444444')
fig.tight_layout(rect=[0, 0.06, 1, 0.94])
save_combined(fig, 'FigureS1_splines_v15')
assert abs(pv_hdl - 0.029) < 0.01 and abs(pv_age - 0.017) < 0.01, (pv_hdl, pv_age)

# ============================== Supplementary Figure S2: distribution comparison
def dist_hist(ax, var, label, unit):
    lo = min(A[var].min(), B[var].min()); hi = max(A[var].max(), B[var].max())
    bins = np.linspace(lo, hi, 26)
    ax.hist(A[var].dropna(), bins=bins, alpha=0.6, color=BLUE, label='Cohort A (n = 682)', density=True)
    ax.hist(B[var].dropna(), bins=bins, alpha=0.6, color=VERM, label='Cohort B (n = 463)', density=True)
    ax.set_xlabel(f'{label} ({unit})' if unit else label)
    ax.set_title(label, loc='left')
    if var == 'age':
        ax.set_ylabel('Density')
    ax.legend(frameon=False, fontsize=6)

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.4), facecolor='white')
dist_hist(a1, 'age', 'Age', 'years')
dist_hist(a2, 'HbA1c', 'HbA1c', '%')
dist_hist(a3, 'HDL', 'HDL-C', 'mmol/L')
fig.tight_layout()
save(fig, 'figS2_distributions_abc')

fig, ax = plt.subplots(figsize=(2.6, 3.0), facecolor='white')
mi = B['maxIMT'].dropna()
bp = ax.boxplot([mi.values], vert=True, widths=0.45, patch_artist=True,
                boxprops=dict(facecolor='#cfe3f0', color=GREY),
                medianprops=dict(color='#17202a'), whiskerprops=dict(color=GREY),
                capprops=dict(color=GREY), flierprops=dict(marker='o', ms=3, mfc=VERM, mec=VERM))
for y in mi[mi > 4].values:
    ax.plot(1, y, '.', color=VERM, ms=4)
ax.annotate(f'max {mi.max():.1f} mm', (1, mi.max()), textcoords='offset points',
            xytext=(10, -2), fontsize=6.5)
ax.set_xticks([])
ax.set_ylabel('Designated maximum IMT (mm)')
ax.set_title(f'IMT extremes > 4 mm (n = {int((mi > 4).sum())})', loc='left', fontsize=7)
save(fig, 'figS2d_imt_box')

fig = plt.figure(figsize=(7.2, 2.5), facecolor='white')
gs = fig.add_gridspec(1, 4, wspace=0.42)
for k, (var, label, unit) in enumerate([('age', 'Age', 'years'),
                                        ('HbA1c', 'HbA1c', '%'),
                                        ('HDL', 'HDL-C', 'mmol/L')]):
    ax = fig.add_subplot(gs[0, k])
    dist_hist(ax, var, label, unit)
    ax.text(-0.18, 1.10, 'ABCD'[k], transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top', ha='left')
ax4 = fig.add_subplot(gs[0, 3])
mi = B['maxIMT'].dropna()
ax4.boxplot([mi.values], vert=True, widths=0.45, patch_artist=True,
            boxprops=dict(facecolor='#cfe3f0', color=GREY),
            medianprops=dict(color='#17202a'), whiskerprops=dict(color=GREY),
            capprops=dict(color=GREY), flierprops=dict(marker='o', ms=3, mfc=VERM, mec=VERM))
for y in mi[mi > 4].values:
    ax4.plot(1, y, '.', color=VERM, ms=4)
ax4.annotate(f'max {mi.max():.1f} mm', (1, mi.max()), textcoords='offset points',
             xytext=(10, -2), fontsize=6.5)
ax4.set_xticks([])
ax4.set_ylabel('Designated maximum IMT (mm)')
ax4.set_title('Designated IMT', loc='left')
ax4.text(-0.25, 1.10, 'D', transform=ax4.transAxes, fontsize=10,
         fontweight='bold', va='top', ha='left')
fig.suptitle('Cohort distributions and the extreme tail of the designated IMT field',
             fontsize=8.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save_combined(fig, 'FigureS2_distributions_v15')

# ------------------------------------------------------------------ packaging
readme = """# JCE v15 figure package

- `panels/` — one file per panel, independent (no panel letters): 600 dpi PNG + vector PDF.
- `combined/` — submission composites with panel letters A/B/C…: 600 dpi PNG + vector PDF.
- `source/make_figures_jce_v15.py` — plotting source; rerun from the repository root with
  `python scripts/make_figures_jce_v15.py`.
- Fonts: Arial 8 pt (panel letters 10 pt bold); PDF text embedded as Type 42 (editable).
- All values are read from frozen analysis outputs (`analysis_results_v14_corrected.json`,
  `analysis_results_v11_sim.json`) or recomputed directly from the harmonised public data
  (`harmonized_A.csv`, `harmonized_B.csv`); no smoothing or manual adjustment.
"""
(ROOT / 'figures' / 'README.md').write_text(readme, encoding='utf-8')
shutil.copy(Path(__file__), SOURCE / 'make_figures_jce_v15.py')
print('README + source copied; figure suite complete')
