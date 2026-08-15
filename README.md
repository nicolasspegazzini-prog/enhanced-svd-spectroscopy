# Enhanced SVD for Signal–Noise Separation in Spectroscopic Data

Supplemental code archive for the Note by N. Spegazzini, *Applied Spectroscopy*
(manuscript ASP-26-0104). This archive is the version of the code as submitted;
subsequent development is available at
https://github.com/nicolasspegazzini-prog/enhanced-svd-spectroscopy

## Contents

| File | Description |
|---|---|
| `svd_baseline_removal.py` | Enhanced SVD implementation (expert module) |
| `svd_baseline_interactive.py` | Interactive script for non-expert users |
| `generate_synthetic_data.py` | Synthetic dataset used for Fig. 1 |
| `generate_figures_final.py` | Reproduces Figs. 1–4 |
| `savgol_compare.py` | Savitzky–Golay comparison (Algorithm Properties) |
| `rank_analysis.py` | Rank determination and subspace projections |
| `savgol_results.txt` | Precomputed output of `savgol_compare.py` |
| `rank_results.txt` | Precomputed output of `rank_analysis.py` |

## Reproducibility

The synthetic dataset is generated with a fixed random seed (42). Figure 1 and
every synthetic value reported in the Note are therefore reproduced exactly by

```
python generate_synthetic_data.py
```

with no external data file.

`savgol_compare.py`, `rank_analysis.py` and Figs. 2–4 of
`generate_figures_final.py` operate on the experimental FTIR data recorded
during the synthesis of diphenylurethane. Those data were reported previously
(Spegazzini N, Siesler HW, Ozaki Y. *J Phys Chem A*. 2011; 115(32): 8832–8844)
and are available from the author on reasonable request. The output of both
scripts on that dataset is included here as `savgol_results.txt` and
`rank_results.txt`, so that every number quoted in the Note can be traced to
the code that produced it.

## Requirements

Python 3.9 or later, with `numpy`, `scipy`, `pandas`, `matplotlib` and
`pillow`.

## Contact

Nicolas Spegazzini
Optical Measurements, VTT Technical Research Centre of Finland Ltd
Kaitoväylä 1, FI-90571 Oulu, Finland
nicolas.spegazzini@vtt.fi
