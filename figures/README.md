# JCE v15 figure package

- `panels/` — one file per panel, independent (no panel letters): 600 dpi PNG + vector PDF.
- `combined/` — submission composites with panel letters A/B/C…: 600 dpi PNG + vector PDF.
- `source/make_figures_jce_v15.py` — plotting source; rerun from the repository root with
  `python scripts/make_figures_jce_v15.py`.
- Fonts: Arial 8 pt (panel letters 10 pt bold); PDF text embedded as Type 42 (editable).
- All values are read from frozen analysis outputs (`analysis_results_v14_corrected.json`,
  `analysis_results_v11_sim.json`) or recomputed directly from the harmonised public data
  (`harmonized_A.csv`, `harmonized_B.csv`); no smoothing or manual adjustment.
