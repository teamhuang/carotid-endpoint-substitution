# Endpoint substitution in prediction-model evaluation — replication package

Code and machine-readable results for:

> Operational endpoint definitions alter the calibration assessment of a fixed prediction model:
> a paired empirical and simulation study in type 2 diabetes (JCE submission, v15)

This repository contains **analysis code and results only**. The two individual-patient source
datasets are publicly archived and are **not** redistributed here:

- Cohort A (Chinese T2D carotid registry, 682 released rows): Figshare supplement,
  DOI 10.3389/fendo.2023.1131430.s001 (resolves to Feng et al., Front Endocrinol 2023;14:1131430).
- Cohort B (PROLOGUE RCT baseline, Japanese T2D, 463 participants): Dryad, DOI 10.5061/dryad.qt743
  (file `PROLOGUE_raw+data+for+submission+20160506.xlsx`).

## Setup

Python 3.10+ with: numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib, patsy, openpyxl.
Place the two downloaded source files in the repository root under the exact names above. To regenerate the figures you also need the intermediate `harmonized_A.csv` / `harmonized_B.csv`, which step 1 of the manuscript pipeline (`main_analysis.py`) writes next to the results; or reuse the copies shipped with the submission reproducibility ZIP.

## Run order

| Step | Script | Output |
|---|---|---|
| 1 | `scripts/rebuild_jce_v14.py` | `analysis_results_v14_corrected.json` (written to the repository root) — the frozen paired endpoint-substitution analysis (agreement matrix, paired bootstrap of ΔAUC/ΔBrier/ΔCITL/Δslope, threshold and subset sensitivities, four fixed scores) |
| 2 | `scripts/verify_perrep_identity.py` | `analysis_results_v14_perrep.json` — per-replication verification that the offset-CITL identity holds to solver tolerance (max residual 1.8×10⁻¹⁵) |
| 3 | `scripts/round11_simulation.py` | `analysis_results_v11_sim.json` — the 12-cell × 5-mechanism simulation (seed 20261116, 2,000 reps/cell) |

After steps 1–3, move the regenerated JSONs into `results/` (they are written to the
repository root) before running step 4.
| 4 | `scripts/make_figures_jce_v15.py` | `figures/panels/`, `figures/combined/` — 600 dpi PNG + vector PDF (Type 42), no SVG |
| 5 | `scripts/audit_jce_v15.py` | package audit: bold discipline, three-line tables, citation order, numbering (requires the manuscript Markdown files) |

`sim_beta_calibration.json` pins the bisection-calibrated score effect used by the simulation
(realised base AUC 0.6300/0.7500 on 400,000 draws).

## Key frozen results (from `analysis_results_v14_corrected.json`)

- Agreement on 453 both-field records: 68.2%, κ = 0.209; prevalence 63.1% vs 87.4%;
  designated-only 17, segment-only 127.
- Paired analysis on 400 patients (counts 259 vs 351): ΔAUC −0.030 (95% CI −0.124 to +0.060);
  ΔBrier +0.049 (+0.024 to +0.074); ΔCITL −1.575 (−1.952 to −1.225); Δslope −0.023 (−0.385 to +0.318).
- The offset-CITL difference equals a(259) − a(351) exactly for this fixed prediction vector.

## Provenance notes

- The cohort-A released file (682 rows) cannot be reconciled to the source publication's
  1,049 (733/316) counts and appears to be a single completed imputation; the fixed score is
  exploratory and its absolute performance should not be generalised.
- The two PROLOGUE IMT fields cannot be adjudicated against a gold standard; the package
  quantifies the consequence of choosing between them, not which is correct.

## Licence

Code: MIT (add a LICENSE file before first formal release if desired).
Data are not included; use them under the licences of the original archives.
