# Measured Baselines

Walk-forward, no lookahead. Ratings rebuilt each week from prior weeks only.
1,615 games, 2020-2025, graded against real closing lines.

| Market | Record | Win % | Break-even |
|---|---|---|---|
| Spread | 772-810 | 48.8% | 52.4% |
| Total | 808-791 | 50.5% | 52.4% |
| Moneyline | 636-974 | 39.5% | -12.7% ROI |

## By disagreement size — spreads

| Min edge | Record | Win % |
|---|---|---|
| 0+ | 772-810 | 48.8% |
| 2+ | 384-433 | 47.0% |
| 3+ | 259-270 | 49.0% |
| 5+ | 106-100 | 51.5% |

Non-monotonic. No relationship between disagreement size and accuracy.

## Accuracy vs the market

| Predicting actual margin | Mean absolute error |
|---|---|
| Market closing line | 9.74 pts |
| Our model | 10.34 pts |

Totals: market 10.28, model 10.74.

## Calibration

Fitted, not assumed:
- P(home win) = Phi(-0.0123 + model_spread / 14.98)
- P(cover) = Phi(-0.0089 - 0.00638 x edge)  <- slope is NEGATIVE
- Market line implied SD: 10.93 (much sharper than ours)

## Approaches tested and rejected

- Blending toward the market: identical 48.1% at every weight 100% to 30%
- Totals dispersion shrink: noise
- 57.2% totals signal at 5+ edges: reversed to 47.4% out-of-sample
- QB offseason adjustment, raw EPA: correlation 0.567 -> 0.533
- QB offseason adjustment, portable metrics (CPOE, pressure-to-sack): 0.567 -> 0.388
- Purpose-built totals model from pace/drives/PPD: 50.4%, declining with edge size
- Props projection vs naive baselines: +0.6%, +0.1%, -0.2%, +0.3%

## Props accuracy

| Market | Model MAE | Season avg | Last 3 | Model vs best |
|---|---|---|---|---|
| QB pass yds | 64.78 | 65.15 | 69.06 | +0.6% |
| RB rush yds | 22.91 | 22.94 | 24.10 | +0.1% |
| WR/TE rec yds | 22.26 | 22.21 | 23.85 | -0.2% |
| WR/TE recs | 1.50 | 1.50 | 1.57 | +0.3% |
