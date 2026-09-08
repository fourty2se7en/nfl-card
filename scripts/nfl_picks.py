"""
nfl_picks.py — grading, moved out of nfl_card.py and put behind the record.

WHAT THIS REPLACES
    This card used to grade inline, with one composite score out of 100:
    70 points for the simulated win probability and 30 for the value over the
    price, then bands and three minimum-value gates. That scheme was invented
    here and the college card copied it, so it is the ancestor of both.

    It has a defect that only shows once there is a backtest. On a spread or a
    total the feed carries no price, so the "value" term was the win probability
    restated in different units and the score counted one number twice. Worse,
    nothing capped it: a market could grade A on a strategy that had never been
    measured, or one that had been measured and lost.

    The composite is still computed and still shown, as raw_score, because the
    college port was verified against this card's own worked example and that
    anchor is worth keeping. It is no longer the grade.

WHAT GRADES NOW, AND WHY IT IS TWO READINGS
    Confidence: how likely this is to happen, after calibration. Nothing to do
    with the price.
    Value: how far that calibrated probability clears what the price demands.
    One letter from the two, weighted, with the weights in model_state.json
    beside the measured numbers so which is which stays visible.

THE CAPS, WHICH ARE THE POINT
    nfl_backtest.py fits, per strategy, the line that maps what the model
    claimed onto what actually happened:

        P(win) = Phi(a + k * Phi^-1(p_sim))

    k is measured, never chosen, and is not constrained to be positive. On
    3,407 games of real closing lines the answer is stark and it replicates the
    college card's:

        outright (who wins)   k = +0.86, clear of zero. Real information.
        spread, weeks 1-4     k = +0.16, not distinguishable from zero
        spread, week 5 on     k = -0.11, not distinguishable from zero
        totals                k = -0.01, not distinguishable from zero

    So the model is a good predictor of football and carries no information
    about who covers. A letter is a claim about evidence. A market whose
    strategy backtested with its whole interval below break-even cannot grade
    above C; one never measured cannot grade above B; one measured and found
    inconclusive cannot grade above B either, because "no evidence either way"
    is not permission for an A.

    Every cap is reported, not applied silently: the card prints a star and the
    reason. The caps lift on their own. Re-run the nfl-backtest workflow and
    every grade here recalculates from the record it commits, with nothing
    pasted anywhere.

    Read the current figures from scripts/model_state.json, not from this
    comment. The comment is what the numbers were when this was written.
"""
import math, os, sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_state as MS

BREAK_EVEN = float(MS.TH["break_even"])

# --- the old composite scale, kept for the raw score only ---
P_FLOOR, P_CEIL = float(MS.SIM["p_floor"]), float(MS.SIM["p_ceil"])
VALUE_CEIL = float(MS.SIM["value_ceil"])
WIN_POINTS, VALUE_POINTS = float(MS.SIM["win_points"]), float(MS.SIM["value_points"])

# --- the grading bands and merge weights. CHOSEN, not measured, and the same
# --- values the college card uses so the two grade alike.
_G = MS.GRADING
CONF_A, CONF_B, CONF_C = float(_G["conf_a"]), float(_G["conf_b"]), float(_G["conf_c"])
VALUE_A, VALUE_B = float(_G["value_a"]), float(_G["value_b"])
W_CONF, W_VALUE = float(_G["w_confidence"]), float(_G["w_value"])

# A call this unlikely cannot be lifted above D by the price alone. Measured
# where the evidence supports one: nfl_backtest.py buckets moneyline bets by
# calibrated probability and asks whether each bucket made money. Where the
# bottom buckets lost with their whole 95% interval below zero, the top of that
# run is the floor. Where none did, there is no measured floor and the chosen
# 40 stands, which is the honest fallback: absence of evidence for a floor is
# not evidence against one.
#
# .get, not [], because model_state.py falls back to values compiled into
# itself when the JSON is missing or broken. A gate that raised KeyError would
# cost the whole card, which is what the fallback exists to prevent.
_CHOSEN_FLOOR = float(_G.get("conf_floor", 40.0))
_MEASURED_FLOOR = (MS.BACKTEST.get("conf_floor") or {}).get("measured")
CONF_FLOOR = float(_MEASURED_FLOOR) if _MEASURED_FLOOR is not None else _CHOSEN_FLOOR
CONF_FLOOR_MEASURED = _MEASURED_FLOOR is not None
CONF_FLOOR_NOTE = ((MS.BACKTEST.get("conf_floor") or {}).get("note")
                   or f"chosen at {_CHOSEN_FLOOR:.0f}%, never measured")

# strategy -> (wins, losses), written by nfl_backtest.py into model_state.json.
RECORD = MS.record()
# Strategies judged on RETURN rather than win rate. A moneyline cannot be judged
# on how often it wins: the price differs on every bet, so 40% of underdogs can
# pay and 85% of favourites can lose money. For these the interval is compared
# against a zero return instead of against the break-even win rate.
RETURNS = MS.BACKTEST.get("returns") or {}
CAL = MS.BACKTEST.get("calibration") or {}

MARKET_STRATEGY = {"spread": None, "moneyline": "moneyline_fav", "total": "total_model"}
ORDER = ["A", "B", "C", "D"]


def _clamp(x):
    return max(0.0, min(1.0, float(x)))


def wilson(wins, losses, z=1.96):
    n = wins + losses
    if n == 0:
        return 0.0, 100.0
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (centre - half), 100 * (centre + half)


# This season's settled picks, folded into the tiers so a cap can move week to
# week instead of only when the backtest is re-run. Filled by nfl_card.py once
# it has read the picks ledger; empty here so the module still works alone.
LIVE = {}


def set_live(d):
    LIVE.clear()
    LIVE.update(d or {})


def _pool_returns(bt, rets):
    """Combine the backtest's mean return with this season's, honestly.

    The backtest stores mean, standard error and n. se*sqrt(n) recovers the
    sample standard deviation, which is what lets the two be pooled rather than
    one simply overwriting the other.
    """
    n0, m0, se0 = int(bt["n"]), float(bt["roi"]), float(bt["se"])
    if not rets:
        return n0, m0, float(bt["lo"]), float(bt["hi"])
    n1 = len(rets)
    m1 = 100.0 * sum(rets) / n1
    sd0 = se0 * math.sqrt(n0)
    ss0 = (n0 - 1) * sd0 ** 2 if n0 > 1 else 0.0
    ss1 = sum((100.0 * r - m1) ** 2 for r in rets)
    n = n0 + n1
    m = (n0 * m0 + n1 * m1) / n
    between = n0 * (m0 - m) ** 2 + n1 * (m1 - m) ** 2      # disagreement widens it
    var = (ss0 + ss1 + between) / max(n - 1, 1)
    se = math.sqrt(var / n)
    return n, m, m - 1.96 * se, m + 1.96 * se


def tier_of(strategy):
    """PLAY, LEAN, PASS or AVOID, decided by the interval and nothing else."""
    live = LIVE.get(strategy) or {}
    r = RETURNS.get(strategy)
    if isinstance(r, dict) and int(r.get("n", 0)) >= 100:
        n, roi, lo, hi = _pool_returns(r, live.get("rets") or [])
        if lo > 0.0:
            return "PLAY", roi, (lo, hi), n
        if hi < 0.0:
            return "AVOID", roi, (lo, hi), n
        if roi > 0.0:
            return "LEAN", roi, (lo, hi), n
        return "PASS", roi, (lo, hi), n
    if strategy not in RECORD:
        return "PASS", 0.0, (0.0, 100.0), 0
    w, l = RECORD[strategy]
    w += int(live.get("w", 0))
    l += int(live.get("l", 0))
    n = w + l
    lo, hi = wilson(w, l)
    pct = 100 * w / n if n else 0.0
    if n < 100:
        return "PASS", pct, (lo, hi), n
    if lo > BREAK_EVEN:
        return "PLAY", pct, (lo, hi), n
    if hi < BREAK_EVEN:
        return "AVOID", pct, (lo, hi), n
    if pct > BREAK_EVEN:
        return "LEAN", pct, (lo, hi), n
    return "PASS", pct, (lo, hi), n


def evidence_line(strategy):
    tier, pct, (lo, hi), n = tier_of(strategy)
    if n < 100:
        return f"{tier}. Not enough backtested games to say anything."
    live = LIVE.get(strategy) or {}
    nl = len(live.get("rets") or []) or (int(live.get("w", 0)) + int(live.get("l", 0)))
    tail = (f" Includes {nl} settled pick{'' if nl == 1 else 's'} from this season."
            if nl else "")
    if strategy in RETURNS:
        return (f"{tier}. Return {pct:+.1f}% per unit staked over {n:,} bets, "
                f"95% interval {lo:+.1f} to {hi:+.1f}, against zero.{tail}")
    return (f"{tier}. {pct:.1f}% over {n:,} games, 95% interval "
            f"{lo:.1f} to {hi:.1f}, against a {BREAK_EVEN}% break-even.{tail}")


def spread_strategy(week):
    """Weeks 1 to 4 lean hardest on the preseason number, so they are measured
    apart from the rest of the season. Same split as the college card."""
    return "model_gap_early" if int(week) <= 4 else "model_gap_late"


# ------------------------------------------------------------------ score
def raw_score(p_win, edge_pts):
    """The old composite, kept so the How to read tab can show what the grades
    used to be built from. Nothing grades on it any more."""
    win = WIN_POINTS * _clamp((p_win - P_FLOOR) / (P_CEIL - P_FLOOR))
    val = VALUE_POINTS * _clamp(edge_pts / VALUE_CEIL)
    return win + val


def band(score):
    return "A" if score >= 70 else "B" if score >= 55 else "C" if score >= 40 else "D"


def worse(a, b):
    return a if ORDER.index(a) >= ORDER.index(b) else b


# ------------------------------------------------------------ calibration
def _z(p):
    return float(norm.ppf(min(max(float(p), 1e-6), 1 - 1e-6)))


def calibrate(p_sim, strategy):
    """The calibrated probability, and whether a real fit was used."""
    c = CAL.get(strategy)
    if not isinstance(c, dict) or ("a" not in c and "k" not in c):
        return float(p_sim), False
    return float(norm.cdf(float(c.get("a", 0.0)) + float(c.get("k", 1.0)) * _z(p_sim))), True


def calibration_note(strategy):
    c = CAL.get(strategy)
    if not isinstance(c, dict):
        return "never calibrated, so this probability is unverified"
    if c.get("level_only"):
        return f"level only, measured over {c.get('n', 0):,} games, no slope to fit"
    k, n = c.get("k"), c.get("n", 0)
    if k is None:
        return "never calibrated"
    if c.get("k_clears_zero"):
        return (f"fitted slope {k:+.2f} over {n:,} games, clear of zero, so this "
                f"probability carries real information")
    return (f"fitted slope {k:+.2f} over {n:,} games, not distinguishable from zero, "
            f"so this reads as close to a coin flip whatever the model says")


def break_even(price=None):
    """The bar to clear, from the actual price where there is one.

    Spreads and totals arrive with a number and no price in this feed, so they
    take the standard -110 figure. A moneyline carries a real price and uses it.
    """
    if price is None:
        return BREAK_EVEN / 100.0
    p = float(price)
    return 100.0 / (p + 100.0) if p > 0 else (-p) / ((-p) + 100.0)


# ---------------------------------------------------------- the two grades
def confidence_grade(p_cal):
    """How likely this is to happen. Nothing to do with the price."""
    pct = 100.0 * p_cal
    if pct >= CONF_A: return "A"
    if pct >= CONF_B: return "B"
    if pct >= CONF_C: return "C"
    return "D"


def value_grade(margin_pp):
    """How far the calibrated probability clears what the price demands."""
    if margin_pp >= VALUE_A: return "A"
    if margin_pp >= VALUE_B: return "B"
    if margin_pp >= 0.0: return "C"
    return "D"


def merge(conf, val):
    n = {"A": 4, "B": 3, "C": 2, "D": 1}
    score = W_CONF * n[conf] + W_VALUE * n[val]
    return {4: "A", 3: "B", 2: "C", 1: "D"}[int(round(_clamp((score - 1) / 3.0) * 3 + 1))]


def grade_market(p_win, edge_pts, strategy, price=None, conf_strategy=None):
    """Confidence, value, and one merged letter, capped by the record.

    p_win         the simulated probability this pick wins
    edge_pts      kept for the ledger and the raw composite, not graded on
    strategy      which backtested strategy this belongs to
    price         the posted price, where the market has one
    conf_strategy which calibration answers the confidence question. A moneyline
                  asks who wins outright, which on this card is the one
                  prediction whose slope clears zero.
    """
    cs = conf_strategy or strategy
    p_cal, fitted = calibrate(p_win, cs)
    be = break_even(price)
    margin = 100.0 * (p_cal - be)

    conf = confidence_grade(p_cal)
    val = value_grade(margin)
    letter = merge(conf, val)
    capped_by = None

    if not fitted:
        # No measured scale means the probability is a claim about nothing.
        new = worse(letter, "C")
        if new != letter:
            capped_by = "this market has never been calibrated"
        letter = new

    if 100.0 * p_cal < CONF_FLOOR:
        new = worse(letter, "D")
        if new != letter:
            kind = "measured" if CONF_FLOOR_MEASURED else "chosen"
            capped_by = (f"only {100 * p_cal:.0f}% likely, below the {kind} "
                         f"{CONF_FLOOR:.0f}% floor, so the price cannot lift it")
        letter = new

    tier, _pct, _iv, n_bt = tier_of(strategy) if strategy else ("PASS", 0.0, (0.0, 100.0), 0)
    if tier == "AVOID":
        new = worse(letter, "C")
        if new != letter:
            capped_by = "this strategy backtested below break-even"
        letter = new
    elif n_bt < 100:
        # The confidence half is measured. Whether BETTING this market makes
        # money has never been tested, and an A would claim it had been.
        new = worse(letter, "B")
        if new != letter:
            capped_by = "no backtested record for this market yet"
        letter = new
    elif tier == "PASS":
        new = worse(letter, "B")
        if new != letter:
            capped_by = "backtested, but with no evidence of an edge either way"
        letter = new

    rs = raw_score(p_win, edge_pts)
    return dict(score=round(rs, 1), raw=band(rs),
                grade=letter, conf=conf, value=val, capped=capped_by,
                p=p_win, p_cal=round(p_cal, 4), conf_pct=round(100 * p_cal, 1),
                break_even=round(100 * be, 1), margin_pp=round(margin, 2),
                calibrated=fitted, calibration=calibration_note(cs),
                edge=edge_pts, tier=tier, strategy=(strategy or ""),
                conf_strategy=cs,
                evidence=evidence_line(strategy) if strategy else "")
