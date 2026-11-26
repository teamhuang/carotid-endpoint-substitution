# Carotid ultrasound endpoint substitution

Reproduction package for the v16 paired cross-sectional analysis and simulation. A frozen ten-predictor logistic score is evaluated against two baseline carotid IMT definitions on the same 400 PROLOGUE participants. Metric differences are designated minus segment-derived in the empirical analysis, and baseline minus alternative in the simulation. Simulated prevalence change uses alternative minus baseline.

## Obtain the inputs

Download the following unmodified files from the original archives. Keep them in a local data directory. Individual-level input and harmonized records are excluded from this repository and the release ZIP.

| Cohort | Source file | Archive MD5 |
| --- | --- | --- |
| A | [Frontiers Figshare record 22108052](https://figshare.com/articles/22108052), DataSheet_1_Development and validation of a nomogram for evaluating the incident risk of carotid atherosclerosis in patients with type 2 diabetes.csv | 0aee880339c157620d520d7df9656a4e |
| B | [PROLOGUE deposit](https://doi.org/10.5061/dryad.qt743), PROLOGUE_raw data for submission 20160506.xlsx | ecab14df1570d1924db161ba663ff78b |

The default input directory is data/. Set the CAROTID_DATA_DIR environment variable to use another directory. The plus-separated legacy PROLOGUE filename is also accepted.

## Run

Use Python 3.12 and install the pinned packages with python -m pip install -r requirements.txt. Then run python run_all.py from the repository root. The 12-cell simulation takes several minutes. The scripts can also be run individually in the order listed in run_all.py. Figure generation uses the two original inputs and paired.json, support.json, and simulation.json. Microsoft Word is unnecessary for the analysis and the figures.

Optional: run python scripts/calibrate_sim_beta.py to check the supplied simulation coefficients against the original seeded reference draw. The full grid reports usable repetitions per metric. Calibration slope summaries use converged fits.

| Script | Output |
| --- | --- |
| analyze_paired.py | Analysis flow, fixed score, paired metrics, intervals, thresholds, and sensitivity sets in paired.json |
| analyze_support.py | Agreement, harmonization summaries, coefficients, spline odds ratios, comparator results, and coding sensitivities in support.json |
| analyze_nomogram.py | Internal optimism assessment of the published and ten-predictor specifications in nomogram.json |
| analyze_imputation.py | Association sensitivity with posterior imputation and Rubin pooling in imputation.json |
| simulate.py | Twelve cells by five mechanisms in simulation.json |
| verify_identity.py | Independent binomial-offset check for all five mechanisms in the representative cell in identity.json |
| make_figures.py | Combined and individual 600 dpi PNG and vector PDF figures |

Harmonized participant-level tables are generated under derived/ for local verification. The released aggregate results are stored under results/.

## Interpretation and provenance

- The 453 participants with both IMT fields had 68.2% agreement. The 400 scorable participants with both fields had 259 versus 351 positive labels and an offset-intercept difference of −1.575. Ambiguous smoking codes were included in the zero category in the historical binary score and were evaluated in two sensitivity analyses.
- The cohort A release contains 682 rows and one exact duplicate. It cannot be reconciled with the 1,049 participants described in the source article. Cohort A sex coding and the mapping between cohort B IMT fields have limited documentation.
- M4 decreases mean prevalence by 2.907 percentage points in the representative cell. The simulation results include signed realized changes and separate grid targets.
- Spline odds ratios are calculated from log-odds contrasts. The corrected internal AUC uses a bootstrap refit evaluated on bootstrap and original samples.
- v15 material remains available in the Git history. Values and figures used in v16 are generated with the scripts and results in this revision.
