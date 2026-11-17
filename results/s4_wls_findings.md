# S4 — Variance-Robust (WLS) Strength Inference

Rows: 9009 · time cutoff: 2024-01-01 · firm seed: 7 · generated: 2026-11-17

> **Caveat:** the panel is already spent (2024+ was examined during exploration). The time holdout below is a replication/stability check, not a pristine out-of-sample test. WLS shifts the estimand toward low-variance firms; equal-weight OLS is printed alongside as the conservative bound.

## Strength coefficient by estimator (full sample)

| h | WLS (primary) | OLS (bound) | z-scored DV |
|---|---|---|---|
| 21 | -0.0645 (p=0.004) | -0.0777 (p=0.029) | -0.7671 (p=0.005) |
| 63 | -0.1835 (p=0.000) | -0.1542 (p=0.143) | -0.9847 (p=0.004) |
| 126 | -0.2388 (p=0.000) | -0.1722 (p=0.073) | -1.0380 (p=0.001) |
| 252 | -0.2300 (p=0.000) | -0.2212 (p=0.044) | -1.2762 (p=0.000) |

## Time holdout — WLS strength (weights frozen from train)

| h | train (pre-cutoff) | test (post-cutoff) |
|---|---|---|
| 21 | -0.0019 (p=0.958) | -0.0099 (p=0.757) |
| 63 | -0.1831 (p=0.233) | -0.1394 (p=0.160) |
| 126 | -0.1873 (p=0.028) | -0.1961 (p=0.044) |
| 252 | -0.2018 (p=0.003) | -0.2807 (p=0.041) |

## Firm holdout — WLS strength (secondary, sign check)

| h | half A | half B |
|---|---|---|
| 21 | -0.0349 (p=0.248) | -0.0936 (p=0.016) |
| 63 | -0.0917 (p=0.104) | -0.2967 (p=0.000) |
| 126 | -0.1938 (p=0.000) | -0.2908 (p=0.000) |
| 252 | -0.3033 (p=0.000) | -0.2381 (p=0.000) |

## Full WLS regressor table

### h = 21

| term | coef | se | t | p | 95% CI | n | n_firms |
|------|------|----|---|---|--------|---|---------|
| strength | -0.0645 | 0.0226 | -2.85 | 0.004 | [-0.1088, -0.0202] | 8646 | 47 |
| discretionary | -0.0169 | 0.0091 | -1.87 | 0.061 | [-0.0347, +0.0008] | 8646 | 47 |
| opp_plan | +0.0081 | 0.0067 | 1.22 | 0.223 | [-0.0049, +0.0211] | 8646 | 47 |
| log_value | -0.0007 | 0.0012 | -0.54 | 0.588 | [-0.0030, +0.0017] | 8646 | 47 |

### h = 63

| term | coef | se | t | p | 95% CI | n | n_firms |
|------|------|----|---|---|--------|---|---------|
| strength | -0.1835 | 0.0405 | -4.53 | 0.000 | [-0.2629, -0.1041] | 8026 | 47 |
| discretionary | -0.0472 | 0.0307 | -1.54 | 0.125 | [-0.1073, +0.0130] | 8026 | 47 |
| opp_plan | +0.0192 | 0.0167 | 1.15 | 0.249 | [-0.0135, +0.0519] | 8026 | 47 |
| log_value | +0.0002 | 0.0021 | 0.12 | 0.908 | [-0.0038, +0.0043] | 8026 | 47 |

### h = 126

| term | coef | se | t | p | 95% CI | n | n_firms |
|------|------|----|---|---|--------|---|---------|
| strength | -0.2388 | 0.0348 | -6.86 | 0.000 | [-0.3070, -0.1706] | 6972 | 47 |
| discretionary | -0.0497 | 0.0307 | -1.62 | 0.105 | [-0.1098, +0.0105] | 6972 | 47 |
| opp_plan | -0.0124 | 0.0087 | -1.42 | 0.155 | [-0.0296, +0.0047] | 6972 | 47 |
| log_value | -0.0005 | 0.0017 | -0.33 | 0.744 | [-0.0038, +0.0027] | 6972 | 47 |

### h = 252

| term | coef | se | t | p | 95% CI | n | n_firms |
|------|------|----|---|---|--------|---|---------|
| strength | -0.2300 | 0.0524 | -4.39 | 0.000 | [-0.3326, -0.1273] | 5060 | 45 |
| discretionary | +0.0109 | 0.0249 | 0.44 | 0.662 | [-0.0380, +0.0598] | 5060 | 45 |
| opp_plan | +0.0396 | 0.0299 | 1.32 | 0.186 | [-0.0191, +0.0982] | 5060 | 45 |
| log_value | -0.0027 | 0.0026 | -1.04 | 0.298 | [-0.0079, +0.0024] | 5060 | 45 |

## Discretionary channel (does not survive)

The S3 writeup attributed the signal to an opportunistic/discretionary flag. In the WLS regressor tables above, `discretionary` is not significant at the focal horizons — the earlier result was a two-firm (PLTR, RBLX) concentration artifact, not a cross-firm effect. Focal-horizon WLS `discretionary` estimates:

- h=63: discretionary -0.0472 (p=0.125)
- h=126: discretionary -0.0497 (p=0.105)

## Verdict

- **h=21: strength not robust** — full-sample sig-neg: True, test-window sig-neg: False, firm-halves sign-consistent: True.
- **h=63: strength not robust** — full-sample sig-neg: True, test-window sig-neg: False, firm-halves sign-consistent: True.
- **h=126: strength ROBUST** — full-sample sig-neg: True, test-window sig-neg: True, firm-halves sign-consistent: True.
- **h=252: strength ROBUST** — full-sample sig-neg: True, test-window sig-neg: True, firm-halves sign-consistent: True.

At the focal 126-day horizon, the variance-robust WLS estimate of sell-into-strength -> forward excess is robust by the pre-registered criteria.
