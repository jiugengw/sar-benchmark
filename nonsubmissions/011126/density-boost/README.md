# Density-head re-ranking (1 Nov, 23:20–23:36)

The +65.85 extras ZIP with 1,107 bridge and playground boxes (score ≥ 0.001) re-ranked by a small density head trained
on the frozen final detector (`sbatch/staging/job_dh.py`, job 348529; merged onto the extras ZIP by
`sbatch/staging/job_dh_extras.py`). Boxes, labels, hedge copies, tail and extras unchanged. No holdout check.

| ZIP | Re-ranking strength g | Δm | F1 | mAP |
|---|---|---|---|---|
| +65.85 extras (final) | — | +65.85 | 95.40 | 59.96 |
| density-boost-g1-extras.zip | 1 | +65.47 | 95.40 | 59.76 |
| density-boost-g2-extras.zip | 2 | +65.37 | 95.40 | 59.71 |

Both worse; not used.
