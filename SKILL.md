---
name: nfl-game-card
description: Produce NFL pre-game analysis — a full-slate card covering every game and every market (spread, moneyline, total, team totals, first half, first quarter), plus player prop projections. Covers venue and weather, rest differential, injury reports, an independent Elo cross-check, spread/moneyline split detection, and a sportsbook self-consistency check. Runs the whole week at once by default; can also do a single-game deep dive. Use this whenever the user asks for NFL analysis, a breakdown, a preview, a read, a card, or picks — for a specific matchup, a week, or the slate. Use it even when the request is short or casual ("thoughts on Chiefs-Bills?", "what's the Week 4 card look like?") — the point is that the format and the honesty about edge stay consistent every time.
---

# NFL Game Card

Full-slate NFL analysis. Every game, every market, plus player projections.

## Read this before producing any output

This model **has no demonstrated edge**. It was tested walk-forward on 1,615 games across 2020–2025, and beat the market in no category:

| Market | Measured | Break-even |
|---|---|---|
| Spread | 48.8% | 52.4% |
| Total | 50.5% | 52.4% |
| Moneyline | −12.7% ROI | 0% |

Model spread implied SD is **14.98** against the market's **10.93** — the market line is substantially sharper. The relationship between our disagreement size and cover probability has a **negative slope**: a 3-point "edge" implies a 48.9% cover rate, not the 59% a naive normal mapping suggests.

**Never present this as a betting system.** Present it as analysis. Tiers describe the size of a model–market disagreement, not confidence in a win. Historically the largest disagreements performed worst, and the card must say so.

Player props were also tested: the projection engine beat naive baselines by +0.6%, +0.1%, −0.2%, +0.3% across four markets — all inside noise. Present projections as reference, never as an edge over posted prop lines.

## Running it

```bash
cd scripts

# one-time / weekly refresh
python3 build_ratings.py          # opponent-adjusted EPA power ratings
python3 elo.py                    # independent Elo cross-check

# the card
python3 nfl_card.py                       # next upcoming week, full slate
python3 nfl_card.py --week 6
python3 nfl_card.py --game "KC @ BUF"     # single-game deep dive
python3 nfl_card.py --no-weather          # skip if network blocked
python3 nfl_card.py --no-props
```

Outputs `Week<N>-Card.html` (open in a browser) and `ledger_week<N>.csv`.

Re-running **preserves** any results already filled into the ledger.

Rebuild ratings weekly. The card warns when they are more than 7 days old.

## What the card contains

**1 · Summary table** — one row per game, split into two visually distinct groups:
- *Market-priced*: spread, moneyline, total, plus the book-consistency check
- *Derived*: team totals, first-half total, first-quarter total — computed from the full-game number, not separately priced. Flag these; books may post different numbers or none at all.

**2 · Spread/moneyline splits** — games where the spread lean and moneyline lean point to opposite sides. This is legitimate: a team can be likely to win without covering. The two markets measure different quantities.

**3 · Game notes** — per game: spread, total and moneyline reads; venue, roof, surface, home-field value; weather with its point adjustment; rest differential; injury report; Elo cross-check.

**4 · Player projections** — volume × shrunk efficiency, top 40 by projection.

## Model components

**Power ratings** — opponent-adjusted EPA and success rate via ridge regression, split pass/rush and offense/defense, weighted 40/25/12/8/5/10 per component, exponential recency decay with an 8-game half-life. Validated at r = 0.935 against actual point differentials. Prior season regressed 35% (offense) and 45% (defense) toward the mean.

**Home field** — venue-specific, **not** a flat 3. League HFA has compressed to roughly 1.0–1.7. Above baseline: SEA, KC, DEN, BUF, NO, GB, BAL, PIT (2.0–2.2). Below: LA, LAC, JAX, LV, ATL (0.8–1.0). Neutral-site games get 0.

**Weather** — Open-Meteo, free, no key, by stadium coordinates. Wind is the only variable that consistently matters: 10–15 mph drops the total 1.5, 15–20 drops 3.0, 20+ drops 5.0. Precipitation above 40% drops 1.0; below 20°F drops 1.5. Domes get skipped.

**Rest** — differential in days since each team last played, capped at ±1.5 points.

**Variance** — the standard deviation used for probability conversion is adjusted per game. High-total games get more, high-wind games get less. This is what produces legitimate spread/moneyline splits: lower variance raises a favorite's win probability while lowering its cover probability.

**Book consistency** — converts the moneyline into an implied spread and compares it to the posted spread. A gap over 1.0 point means the sportsbook's own two prices disagree with each other. That is a pricing inconsistency rather than a handicapping opinion, and it is the most interesting signal on the card.

**Elo cross-check** — built from game results only, deliberately using no EPA, so disagreement is informative rather than circular. Correlates 0.906 with the main model. Games diverging by 3+ points are flagged for review.

## Data sources

All free, no API keys:
- **nflverse** — play-by-play, schedules, **betting lines** (spread, total, moneyline for the current season), rosters, depth charts, injuries. Reachable from restricted environments because it is hosted on GitHub.
- **Open-Meteo** — weather.

Note: nflverse carries **no preseason data** at all. Preseason cards require lines and rotation reporting gathered by web search instead.

## Preseason

If asked for a preseason card, the framework changes:
- **Power ratings do not apply.** They describe starters who mostly will not play. Exclude them.
- The driver is **rotation asymmetry** — one team's starters against the other's backups. This comes from beat reporting, not any dataset.
- **First-half derivation inverts.** Regular season uses 0.55× because favorites separate late; preseason concentrates the edge in the first half, so use ~1.0×.
- Both teams in full backup mode means low variance, which compresses margins and favors underdogs against the spread.
- Grade in a **separate PRESEASON ledger**, never merged with regular season.

## Honesty requirements

1. Never claim an edge. Tiers describe disagreement size, not expected win rate.
2. Never invent a line, injury status, weather reading, or stat. Mark `DATA GAP`.
3. Surface every failure on the card itself, in an issues panel — not just the console.
4. Flag derived markets as derived.
5. State that largest disagreements historically performed worst.
6. No bet sizing, no bankroll, no unit recommendations, ever.

## The one open question

`track_gaps.py` tests whether large model–market disagreements outperform small ones. Every backtest so far says no. If that ever reverses over a meaningful sample, the confidence model earns a revision — and that comparison is only possible because the full record is kept, including low-tier picks.

Needs roughly 100 graded picks before it means anything.
