"""nfl_backtest.py — what, if anything, this card can honestly claim.

Run by hand from the Actions tab. It refits the model as of each week of each
past season, using only what was known at the time, prices every game against
the closing line, and writes what it finds back into model_state.json so the
card grades on a record instead of on a decision somebody made.

Three rules it exists to keep:

  It measures THE MODEL THE CARD PUBLISHES. The fit lives in nfl_fit.py and both
  files call it. On the college card the same algorithm was written out twice,
  drifted, and the backtest spent its time measuring a model the page did not
  publish. See 4.3n.

  It refuses to run on the fallback constants. Every other script survives a
  broken model_state.json by using the copy compiled into model_state.py. This
  one must not: its whole job is to describe the live model, and describing the
  wrong one is worse than not running.

  It reports what it finds, including nothing. "No evidence either way" is not
  permission to grade a pick.
"""
import os, sys, json, time
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import model_state as MS
import nfl_fit as FIT

if MS.FELL_BACK:
    sys.exit(f"ERROR: {MS.FELL_BACK}. The backtest describes the live model, so it "
             f"will not run on the values compiled into model_state.py.")

R = MS.RATINGS
HALF_LIFE   = float(R["half_life"])
RIDGE_ALPHA = float(R["ridge_alpha"])
EARLY_DOWN  = float(R["early_down_weight"])
EPA_W       = float(R["epa_weight"])
PLAYS_PG    = float(R["plays_per_game"])
COMPOSITE   = dict(R["composite"])
OFF_KEEP    = 1.0 - float(R["off_regress"])
DEF_KEEP    = 1.0 - float(R["def_regress"])

# ---- from nfl_card.py. Not constants of the fit, but of the line the card
# ---- publishes, so the backtest has to use the same ones or it is measuring
# ---- something else. Test 8 asks whether the home-field table earns its place.
HFA = {'SEA':2.2,'KC':2.2,'DEN':2.2,'BUF':2.2,'NO':2.0,'GB':2.0,'BAL':2.0,'PIT':2.0,
       'LA':1.0,'LAC':0.8,'JAX':1.0,'LV':0.8,'ATL':1.2}
DEF_HFA, AVG_PTS = 1.5, 22.4

# The simulation constants. These were literals in this file AND in nfl_card.py,
# which is 4.3h: one number, two files, nothing failing when a copy drifts. They
# now live in model_state.json and both read them from there.
LINE_SD = float(MS.SIM["line_sd"])            # our uncertainty about the true line
MARGIN_SD = float(MS.SIM["margin_sd"])        # the game around that line
TOTAL_LINE_SD = float(MS.SIM["total_line_sd"])
TOTAL_SD = float(MS.SIM["total_sd"])
TOTAL_BUMP = tuple(float(x) for x in MS.SIM["total_bump"])
BREAK_EVEN = float(MS.TH["break_even"])       # what -110 needs

# ---- backtest-only ----
FIRST_PRIOR = 2012                       # fitted only to give 2013 a prior
SEASONS = list(range(2013, 2026))
BLEND_GRID = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 30, None]   # None = never update

print("constants in use:")
print(f"  half-life {HALF_LIFE}  ridge {RIDGE_ALPHA}  early-down {EARLY_DOWN}  "
      f"epa weight {EPA_W}  {PLAYS_PG} plays")
print(f"  regress {1-OFF_KEEP:.2f}/{1-DEF_KEEP:.2f}   line sd {LINE_SD}   "
      f"margin sd {MARGIN_SD}   break-even {BREAK_EVEN}%")
print()


# ============================================================ statistics
def wilson(wins, n):
    """95% interval on a win rate, so a lucky bucket cannot masquerade as edge."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - m), 100 * (c + m)


def verdict(wins, losses):
    n = wins + losses
    if n < 100:
        return "too few games to say"
    lo, hi = wilson(wins, n)
    if lo > BREAK_EVEN:
        return f"CLEARS break-even ({lo:.1f} to {hi:.1f} at 95%)"
    if hi < BREAK_EVEN:
        return f"loses at -110 ({lo:.1f} to {hi:.1f} at 95%)"
    return f"no evidence either way ({lo:.1f} to {hi:.1f} at 95%)"


def tier(wins, losses):
    """The tier a strategy earns from its record. Never from how big a number looks."""
    n = wins + losses
    if n < 100:
        return "PASS"
    lo, hi = wilson(wins, n)
    if lo > BREAK_EVEN:
        return "PLAY"
    if hi < BREAK_EVEN:
        return "AVOID"
    if wins / n > BREAK_EVEN / 100:
        return "LEAN"
    return "PASS"


def report(mask, win, label):
    """One printed line. Both arguments are positional arrays, never Series.

    These frames are filtered differently from one another, so a pandas boolean
    Series carries an index that need not line up with the one it is masking.
    """
    mask = np.asarray(mask, dtype=bool)
    win = np.asarray(win, dtype=bool)
    n = int(mask.sum())
    if n < 40:
        return None
    w = int(win[mask].sum())
    l = n - w
    print(f"  {label:<26} {n:>6} {f'{w}-{l}':>13} {100*w/n:>7.1f}%   {verdict(w, l)}")
    return w, l


def paired_t(base_err, new_err):
    """The per-game difference, not two separate error bars.

    Comparing two versions of the model on the same games has to be done game by
    game. The unpaired ruler is about seven times wider here and hides every real
    effect this test is looking for.
    """
    d = np.asarray(base_err) - np.asarray(new_err)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return 0.0, 0.0
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


# ============================================================ data
print("loading play-by-play and schedules from nflverse ...")
import nflreadpy as nfl

PBP_COLS = ["season", "week", "season_type", "game_id", "posteam", "defteam",
            "epa", "pass", "rush", "down"]
_frames = []
for _s in range(FIRST_PRIOR, SEASONS[-1] + 1):
    _d = nfl.load_pbp([_s]).to_pandas()
    _d = _d[_d.season_type == "REG"][PBP_COLS]
    _frames.append(_d[_d.posteam.notna() & _d.defteam.notna()])
PBP = pd.concat(_frames, ignore_index=True)
del _frames

SCH = nfl.load_schedules().to_pandas()
SCH = SCH[(SCH.season >= SEASONS[0]) & (SCH.season <= SEASONS[-1])
          & (SCH.game_type == "REG") & SCH.result.notna()].copy()

# nflverse play-by-play carries today's abbreviation for a team that has moved;
# the schedule file carries the one in use that season. Joined without this,
# 224 team-games disappear in silence and the run still reports success. 4.3m:
# resolve it or report it, never approximate, and never leave it unchecked.
MOVED = {"OAK": "LV", "STL": "LA", "SD": "LAC", "LAR": "LA", "JAC": "JAX"}
for _c in ("home_team", "away_team"):
    SCH[_c] = SCH[_c].replace(MOVED)
_known = set(PBP.posteam.dropna()) | set(PBP.defteam.dropna())
_unknown = sorted((set(SCH.home_team) | set(SCH.away_team)) - _known)
if _unknown:
    sys.exit("ERROR: these schedule teams have no play-by-play under that name: "
             + ", ".join(_unknown) + ". Add them to MOVED rather than letting the "
             "join drop their games.")
print(f"  plays {len(PBP):,}   games {len(SCH):,}   seasons {SEASONS[0]}-{SEASONS[-1]}")


def _fit(frame, max_week):
    return FIT.fit_ratings(frame, half_life=HALF_LIFE, ridge_alpha=RIDGE_ALPHA,
                           early_down_weight=EARLY_DOWN, epa_weight=EPA_W,
                           plays_per_game=PLAYS_PG, composite=COMPOSITE,
                           max_week=max_week)


def build_fits():
    """Every fit the tests need, each seeing only what was known at the time.

    PRIOR[s] is where a team starts season s: last season's fit regressed toward
    the mean, which is what build_ratings.py writes and the card carries in.
    INS[(s, w)] is the fit on season s through week w-1 and nothing later.
    """
    t0 = time.time()
    prior, ins, played = {}, {}, {}
    for s in range(FIRST_PRIOR, SEASONS[-1] + 1):
        f = _fit(PBP[PBP.season == s], None)
        prior[s + 1] = pd.DataFrame({"off_pt": f.off_pt * OFF_KEEP,
                                     "def_pt": f.def_pt * DEF_KEEP})
    for s in SEASONS:
        d = PBP[PBP.season == s]
        for w in sorted(d.week.unique()):
            past = d[d.week < w]
            played[(s, w)] = past.groupby("posteam").game_id.nunique()
            if past.game_id.nunique() < 16:      # a full slate before fitting
                ins[(s, w)] = None
                continue
            try:
                ins[(s, w)] = _fit(past, w - 1)[["off_pt", "def_pt"]]
            except Exception as e:
                print(f"  fit failed at {s} week {w}: {e}")
                ins[(s, w)] = None
    print(f"  walk-forward fits: {len(ins)} weeks in {time.time()-t0:.1f}s")
    return prior, ins, played


PRIOR, INS, PLAYED = build_fits()


def rating_as_of(season, week, k):
    """Offence and defence in points, as the card would have had them.

    k is how many games of results it takes to move halfway off the preseason
    number: the in-season fit gets weight n/(n+k) after n games. k=0 throws the
    prior away the moment a game is played, k=None never looks at results at
    all, which is what this card does today. Test 7 measures it.
    """
    pr = PRIOR[season]
    off, dfn = pr.off_pt.copy(), pr.def_pt.copy()
    cur = INS.get((season, week))
    if cur is None or k is None:
        return off, dfn
    n = PLAYED[(season, week)].reindex(off.index).fillna(0).astype(float)
    lam = pd.Series(1.0, index=off.index) if k == 0 else n / (n + float(k))
    c_off = cur.off_pt.reindex(off.index)
    c_def = cur.def_pt.reindex(off.index)
    lam = lam.where(c_off.notna(), 0.0)
    return ((1 - lam) * off + lam * c_off.fillna(0.0),
            (1 - lam) * dfn + lam * c_def.fillna(0.0))


def price_slate(k=None, hfa_fn=None):
    """Our number for every game in the window, on one setting of the model."""
    if hfa_fn is None:
        hfa_fn = lambda season, home: HFA.get(home, DEF_HFA)
    rows = []
    for (s, w), grp in SCH.groupby(["season", "week"]):
        off, dfn = rating_as_of(s, w, k)
        for g in grp.itertuples():
            h, a = g.home_team, g.away_team
            rest = 0
            if pd.notna(g.home_rest) and pd.notna(g.away_rest):
                rest = int(g.home_rest - g.away_rest)
            margin = ((off[h] + dfn[h]) - (off[a] + dfn[a]) + hfa_fn(s, h)
                      + float(np.clip(rest * 0.25, -1.5, 1.5)))
            total = (2 * AVG_PTS + (off[h] - dfn[a]) + (off[a] - dfn[h])
                     + (1.5 if str(g.roof) == "dome" else 0.0))
            rows.append((s, w, h, a, float(g.result), float(g.spread_line),
                         float(g.total), float(g.total_line), total, margin,
                         g.home_moneyline, g.away_moneyline, bool(g.div_game)))
    return pd.DataFrame(rows, columns=["season", "week", "home", "away", "margin",
                                       "mkt", "total", "total_line", "model_total",
                                       "model", "hml", "aml", "div"])


# ============================================================ Test 7 first:
# every other test needs to know how the ratings are built, and today the card
# has no answer. Nothing downstream is meaningful until this one is settled.
def test_blend():
    print("\n" + "=" * 84)
    print("TEST 7 - HOW FAST SHOULD RESULTS TAKE OVER FROM THE PRESEASON NUMBER?")
    print("=" * 84)
    print("  The card has no answer today: it sits on the preseason ratings all year.")
    print("  Each row gives the in-season fit weight n/(n+k) after n games.")
    print(f"  {'k':>6} {'half-way at':>12} {'MAE':>8} {'vs never':>10} {'paired t':>9} {'ATS%':>7}")
    base = price_slate(None)
    base_err = (base.margin - base.model).abs().values
    best = (None, 1e9)
    out = {}
    for k in BLEND_GRID:
        p = price_slate(k)
        err = (p.margin - p.model).abs().values
        gain, t = paired_t(base_err, err)
        q = p[p.margin != p.mkt]
        cov = (q.margin > q.mkt).values
        e = (q.model - q.mkt).values
        win = ((e > 0) & cov) | ((e <= 0) & ~cov)
        half = "never" if k is None else ("at once" if k == 0 else f"{k} games")
        print(f"  {str(k):>6} {half:>12} {err.mean():>8.3f} {gain:>+10.3f} "
              f"{t:>9.2f} {100*win.mean():>7.2f}")
        out[str(k)] = dict(mae=float(err.mean()), gain=gain, t=t)
        if err.mean() < best[1]:
            best = (k, err.mean())
    mkt = float((base.margin - base.mkt).abs().mean())
    print(f"\n  the market's own error on the same games: {mkt:.3f}")
    print(f"  best k: {best[0]}   error {best[1]:.3f}   still {best[1]-mkt:+.3f} worse than the market")
    print("  READ THIS AS: blending results in makes the model a better predictor of")
    print("  football. It does not make it a better predictor of what the market got")
    print("  wrong, which is a different thing and is what Test 1 measures.")
    print("  CAVEAT: historical sportsbook win totals are not available, so the prior")
    print("  measured here is last season regressed toward the mean, while the card")
    print("  carries the market's preseason number. k describes how fast a dozen games")
    print("  of results outweigh ONE prior number, which does not depend on which")
    print("  prior it was. It would if the market prior were much the better of the two.")
    return best[0], out


BEST_K, BLEND_TABLE = test_blend()
P = price_slate(BEST_K)
print(f"\npricing every game with k={BEST_K}: {len(P):,} games")


# ============================================================ Test 8
def test_home_field():
    print("\n" + "=" * 84)
    print("TEST 8 - DOES THE PER-TEAM HOME FIELD TABLE EARN ITS PLACE?")
    print("=" * 84)
    zero = price_slate(BEST_K, hfa_fn=lambda s, h: 0.0)
    zero = zero.assign(resid=lambda x: x.margin - x.model)
    measured_edge = float(zero.resid.mean())
    print(f"  home edge measured from the residuals: {measured_edge:+.2f} points")
    print(f"  the market's average home number:      {zero.mkt.mean():+.2f} points")
    for lo, hi in ((2013, 2016), (2017, 2019), (2020, 2020), (2021, 2025)):
        z = zero[(zero.season >= lo) & (zero.season <= hi)]
        tag = "  (no crowds)" if lo == 2020 else ""
        print(f"    {lo}-{hi}: ours {z.resid.mean():+.2f}  market {z.mkt.mean():+.2f}"
              f"  n={len(z)}{tag}")

    # Measured per team, from earlier seasons only, shrunk toward the league
    # number. Sixty home games is roughly four seasons.
    tables = {}
    for s in SEASONS:
        h = zero[zero.season < s]
        if len(h) < 800:
            tables[s] = None
            continue
        g = h.groupby("home").resid.agg(["mean", "size"])
        lg = float(h.resid.mean())
        tables[s] = ((g["size"] * g["mean"] + 60 * lg) / (g["size"] + 60), lg)

    def measured(season, home):
        t = tables.get(season)
        return DEF_HFA if t is None else float(t[0].get(home, t[1]))

    def flat(season, home):
        t = tables.get(season)
        return DEF_HFA if t is None else float(t[1])

    variants = [("the card's table", None),
                ("one flat number, measured", flat),
                ("per team, measured", measured),
                ("no home field at all", lambda s, h: 0.0)]
    base_err = None
    print(f"\n  {'variant':<28} {'MAE':>8} {'vs the table':>13} {'paired t':>9}")
    for name, fn in variants:
        p = price_slate(BEST_K, hfa_fn=fn)
        err = (p.margin - p.model).abs().values
        if base_err is None:
            base_err = err
        gain, t = paired_t(base_err, err)
        print(f"  {name:<28} {err.mean():>8.3f} {gain:>+13.3f} {t:>9.2f}")
    print("  A table that cannot be told apart from one flat number is not adding")
    print("  information. It is not hurting either, so it stays; it is not evidence.")
    return measured_edge


HOME_FIELD_MEASURED = test_home_field()


# ============================================================ probabilities
# The card turns our number into a probability by drawing twice: our own
# uncertainty about the true line, then the game around that line. Two normals
# add, so the closed form below is the same thing without 20,000 draws, and it
# is what makes a calibration fit possible at all.
def sd_margin(total_line):
    """Margin scatter, widened for a shootout exactly as the card widens it."""
    t = np.where(np.isfinite(total_line), total_line, 45.0)
    return MARGIN_SD + np.where(t > TOTAL_BUMP[0], TOTAL_BUMP[1], 0.0)


def add_probabilities(frame):
    f = frame.copy()
    sdq = np.sqrt(LINE_SD ** 2 + sd_margin(f.total_line.values) ** 2)
    sdt = np.sqrt(TOTAL_LINE_SD ** 2 + TOTAL_SD ** 2)
    f["sdq"] = sdq
    f["p_cover_home"] = norm.cdf((f.model.values - f.mkt.values) / sdq)
    f["p_home_win"] = norm.cdf(f.model.values / sdq)
    f["p_over"] = norm.cdf((f.model_total.values - f.total_line.values) / sdt)
    return f


P = add_probabilities(P)


def fit_probit(p_sim, won, min_n=150):
    """P(win) = Phi(a + k * Phi^-1(p_sim)), by maximum likelihood.

    k is deliberately NOT constrained to be positive. On both cards the measured
    slope has come out at or below zero, meaning a bigger disagreement did not
    win more often, and a constrained fit would hide exactly that.

    Written to match cfb_backtest.py line for line, so the two cards' numbers
    mean the same thing and can be read side by side.
    """
    p = np.clip(np.asarray(p_sim, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(won, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(y) < min_n or y.max() == y.min():
        return None
    z = norm.ppf(p)

    def nll(th):
        q = np.clip(norm.cdf(th[0] + th[1] * z), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    r = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    a, k = float(r.x[0]), float(r.x[1])
    try:
        se = np.sqrt(np.diag(r.hess_inv))
        se_a, se_k = float(se[0]), float(se[1])
    except Exception:
        se_a = se_k = float("nan")
    return dict(a=round(a, 4), k=round(k, 4), se_a=round(se_a, 4), se_k=round(se_k, 4),
                n=int(len(y)), k_clears_zero=bool(k - 1.96 * se_k > 0))


def show_fit(name, fit):
    if not fit:
        print(f"  {name:<18} not enough games to fit")
        return
    if fit.get("level_only"):
        print(f"  {name:<18} a {fit['a']:+.4f}  no slope: this strategy has no model "
              f"input to calibrate, only a level   n {fit['n']:,}")
        return
    se = fit.get("se_k")
    se_txt = f"(se {se:.4f})" if isinstance(se, float) and np.isfinite(se) else "(se unavailable)"
    print(f"  {name:<18} a {fit['a']:+.4f}  k {fit['k']:+.4f} {se_txt}  n {fit['n']:,}   "
          f"{'k clears zero' if fit['k_clears_zero'] else 'k does not clear zero'}")


def keep(name, fn):
    """Run one measurement and carry on if it fails.

    A single bad frame would otherwise lose the whole write, leaving the card
    grading on the previous record while this output said otherwise. Each piece
    stands or falls on its own, and says which.
    """
    try:
        return fn()
    except Exception as ex:
        print(f"  SKIPPED {name}: {ex}")
        return None


# ============================================================ the market tests
# Strategy names are the college card's, deliberately. The two cards measure
# the same things and a reader should not have to translate between them.
#   model_gap_early   our number against the spread, weeks 1-4
#   model_gap_late    the same, week 5 onward
#   total_model       our total against the market's total
#   total_over        take every over; no model input, a level only
#   outright          who wins the game, which is a different and better
#                     measured prediction than who covers
#   moneyline*        judged on return per unit staked, never on win rate
RECORD, CALIB, RETURNS = {}, {}, {}


def test_spread():
    print("\n" + "=" * 84)
    print(f"TEST 1 - AGAINST THE CLOSING SPREAD   (break-even {BREAK_EVEN}%)")
    print("=" * 84)
    d = P[P.margin != P.mkt].copy()
    cov = (d.margin > d.mkt).values
    e = (d.model - d.mkt).values
    win = ((e > 0) & cov) | ((e <= 0) & ~cov)
    print(f"  {'disagreement':<26} {'games':>6} {'record':>13} {'win%':>8}   verdict")
    for lo, hi, lab in ((0, 99, "any"), (0, 3, "0 to 3"), (3, 6, "3 to 6"),
                        (6, 10, "6 to 10"), (10, 99, "10 or more")):
        report((np.abs(e) >= lo) & (np.abs(e) < hi), win, lab)
    print(f"  our margin misses by {float((P.margin-P.model).abs().mean()):.2f} points, "
          f"the market's by {float((P.margin-P.mkt).abs().mean()):.2f}")

    early = d.week.values <= 4
    for name, mask in (("model_gap_early", early), ("model_gap_late", ~early)):
        w = int(win[mask].sum())
        RECORD[name] = (w, int(mask.sum()) - w)
        # p_sim is the probability OUR side covers, which is what the card
        # prints, so that is what has to be calibrated.
        p_home = d.p_cover_home.values[mask]
        p_ours = np.where(p_home > 0.5, p_home, 1 - p_home)
        CALIB[name] = fit_probit(p_ours, win[mask])
    return d, e, win


def test_totals():
    print("\n" + "=" * 84)
    print("TEST 2 - TOTALS, against the closing total")
    print("=" * 84)
    t = P[P.total != P.total_line].copy()
    over = (t.total > t.total_line).values
    e = (t.model_total - t.total_line).values
    win = ((e > 0) & over) | ((e <= 0) & ~over)
    print(f"  {'disagreement':<26} {'games':>6} {'record':>13} {'win%':>8}   verdict")
    for lo, hi, lab in ((0, 99, "any"), (0, 3, "0 to 3"), (3, 6, "3 to 6"), (6, 99, "6 or more")):
        report((np.abs(e) >= lo) & (np.abs(e) < hi), win, lab)
    print(f"  our total misses by {float((t.total-t.model_total).abs().mean()):.2f} points, "
          f"the market's by {float((t.total-t.total_line).abs().mean()):.2f}")

    w = int(win.sum())
    RECORD["total_model"] = (w, len(win) - w)
    p_o = t.p_over.values
    CALIB["total_model"] = fit_probit(np.where(p_o > 0.5, p_o, 1 - p_o), win)

    wo = int(over.sum())
    RECORD["total_over"] = (wo, len(over) - wo)
    print(f"  every over: {wo}-{len(over)-wo} ({100*over.mean():.1f}%)   "
          f"{verdict(wo, len(over)-wo)}")
    # Taking every over uses no model input, so there is no slope to fit. Only
    # the level is measurable, and saying so is more honest than fitting a line
    # through a constant and reporting whatever number falls out.
    CALIB["total_over"] = dict(a=round(float(norm.ppf(np.clip(over.mean(), 1e-6, 1 - 1e-6))), 4),
                               k=0.0, se_a=None, se_k=None, n=int(len(over)),
                               k_clears_zero=False, level_only=True)


def test_subgroups(d, e, win):
    print("\n" + "=" * 84)
    print("TEST 3 - SUBGROUPS. Printed for a reader, not stored as strategies.")
    print("=" * 84)
    print(f"  {'group':<26} {'games':>6} {'record':>13} {'win%':>8}   verdict")
    wk = d.week.values
    for mask, lab in (((wk > 4) & (wk <= 9), "weeks 5-9"),
                      (wk > 9, "weeks 10+"),
                      (d["div"].values, "divisional"),
                      (~d["div"].values, "non-divisional"),
                      (d.mkt.abs().values <= 3, "line 3 or less"),
                      (d.mkt.abs().values >= 7, "line 7 or more"),
                      ((np.abs(e) >= 3) & (wk > 4), "3+ gap, week 5+")):
        report(mask, win, lab)
    print("  Only model_gap_early and model_gap_late are stored. A subgroup that")
    print("  looks good here is a slice chosen after seeing the answer, and the card")
    print("  must not grade on one.")


def amp(x):
    """A moneyline as an implied probability, vig included."""
    return 100 / (x + 100) if x > 0 else -x / (-x + 100)


def dec(x):
    """A moneyline as a decimal payout."""
    return 1.0 + x / 100.0 if x > 0 else 1.0 + 100.0 / (-x)


def test_moneyline():
    """Betting the straight winner, measured by RETURN.

    A moneyline cannot be judged on win rate: a 35% underdog at +250 makes money
    and a 90% favourite at -1200 does not. It is walk-forward twice over. The
    ratings see only earlier weeks, AND the calibration used to choose a side is
    fitted only on seasons before the one being bet, so no bet is picked using
    its own outcome.
    """
    print("\n" + "=" * 84)
    print("TEST 4 - MONEYLINE, measured by what it returns")
    print("=" * 84)
    m = P[P.hml.notna() & P.aml.notna()].copy()
    ph, pa = m.hml.map(amp), m.aml.map(amp)
    m["mkt_home"] = np.clip(ph / (ph + pa), 0.001, 0.999)
    m["dec_home"], m["dec_away"] = m.hml.map(dec), m.aml.map(dec)
    m["home_won"] = (m.margin.values > 0).astype(float)
    rows = []
    seasons = sorted(m.season.unique())
    for s in seasons[1:]:
        past, fut = m[m.season < s], m[m.season == s]
        if len(past) < 400 or fut.empty:
            continue
        fit = fit_probit(past.p_home_win.values, past.home_won.values, min_n=400)
        a_, k_ = (0.0, 1.0) if not fit else (fit["a"], fit["k"])
        p_cal = norm.cdf(a_ + k_ * norm.ppf(np.clip(fut.p_home_win.values, 1e-6, 1 - 1e-6)))
        take_home = p_cal > fut.mkt_home.values
        d_ = np.where(take_home, fut.dec_home.values, fut.dec_away.values)
        won = np.where(take_home, fut.home_won.values > 0, fut.home_won.values == 0)
        rows.append(pd.DataFrame(dict(season=s, won=won.astype(float),
                                      ret=np.where(won, d_ - 1.0, -1.0),
                                      edge=np.abs(p_cal - fut.mkt_home.values) * 100,
                                      p_cal=np.where(take_home, p_cal, 1 - p_cal),
                                      dog=(d_ > 2.0))))
    if not rows:
        print("  not enough seasons to run this walk-forward")
        return None
    R = pd.concat(rows, ignore_index=True)

    def line(label, sub, key=None):
        if len(sub) < 100:
            return
        w = int(sub.won.sum())
        l = len(sub) - w
        roi = float(sub.ret.mean()) * 100
        se = float(sub.ret.std(ddof=1)) / np.sqrt(len(sub)) * 100
        lo, hi = roi - 1.96 * se, roi + 1.96 * se
        v = ("CLEARS, the whole interval is above zero" if lo > 0 else
             "loses, the whole interval is below zero" if hi < 0 else
             "no evidence either way")
        print(f"  {label:<26} {len(sub):>6} {f'{w}-{l}':>13} {roi:>+7.1f}%   "
              f"({lo:+.1f} to {hi:+.1f})  {v}")
        if key:
            RECORD[key] = (w, l)
            RETURNS[key] = dict(n=int(len(sub)), roi=round(roi, 2), se=round(se, 2),
                                lo=round(lo, 2), hi=round(hi, 2), w=w, l=l)

    print(f"  {'group':<26} {'bets':>6} {'record':>13} {'return':>8}   95% interval")
    line("every game", R, "moneyline")
    line("we take the dog", R[R.dog], "moneyline_dog")
    line("we take the favourite", R[~R.dog], "moneyline_fav")
    for lo_ in (2, 5, 10):
        line(f"edge {lo_}+ points", R[R.edge >= lo_])
    CALIB["outright"] = fit_probit(P.p_home_win.values, (P.margin.values > 0).astype(float))
    CALIB["moneyline"] = CALIB["moneyline_fav"] = CALIB["moneyline_dog"] = CALIB["outright"]
    return R


def test_price_consistency():
    print("\n" + "=" * 84)
    print("TEST 5 - PRICE CONSISTENCY. The moneyline against the posted spread.")
    print("=" * 84)
    c = P[P.hml.notna() & P.aml.notna() & (P.margin != P.mkt)].copy()
    ph, pa = c.hml.map(amp), c.aml.map(amp)
    fair = np.clip(ph / (ph + pa), 0.001, 0.999)
    gap = (norm.ppf(fair) * c.sdq.values - c.mkt.values)
    print(f"  quotes: {len(c):,}   average gap {gap.mean():+.2f} points, "
          f"scatter {gap.std():.2f}")
    cov = (c.margin > c.mkt).values
    win = ((gap > 0) & cov) | ((gap <= 0) & ~cov)
    print(f"  {'threshold':<26} {'bets':>6} {'record':>13} {'win%':>8}   verdict")
    for thr in (1.0, 1.5, 2.0, 3.0):
        report(np.abs(gap) >= thr, win, f"{thr} points or more")
    print("  NOT STORED AS A STRATEGY, and the card should not show it.")
    print("  On the college card this is the one angle that survives, because there it")
    print("  compares EACH BOOK's moneyline with THAT BOOK's own spread and a stale one")
    print("  shows up. nflverse publishes one consensus number per market, and a")
    print(f"  consensus is internally consistent by construction: the gap scatters only")
    print(f"  {gap.std():.2f} points. Testing this here needs per-book prices, which this")
    print("  card does not have.")


D, E, WIN1 = test_spread()
keep("totals", test_totals)
keep("subgroups", lambda: test_subgroups(D, E, WIN1))
R7 = keep("moneyline", test_moneyline)
keep("price consistency", test_price_consistency)

# A section that produced nothing must say so. The college run has this guard,
# and it is the only reason a whole test that had silently stopped running was
# ever noticed: the job still reported success. See 4.3j.
_expected = {"model_gap_early", "model_gap_late", "total_model", "moneyline"}
_missing = sorted(_expected - set(RECORD))
if _missing:
    sys.exit(f"ERROR: these strategies produced no record at all: {', '.join(_missing)}. "
             f"A test that quietly stops running still leaves the run green, so this "
             f"stops instead of writing a partial record over a good one.")


# ============================================================ calibration
def calibration_report():
    """Does the model's stated confidence mean anything?

    The card grades on a simulated probability. If that probability carries no
    information about what actually happened, every letter on the page is a
    claim with nothing behind it. This is what caps the grades, and it is the
    reason the college card needed a fourth gate.
    """
    print("\n" + "=" * 84)
    print("CALIBRATION - what the model claims against what happened")
    print("=" * 84)
    print("  P(win) = Phi(a + k * Phi-inverse(p_sim)).  k = 1 is perfect, k = 0 is")
    print("  no information at all, and k is not constrained to be positive.")
    for name in ("model_gap_early", "model_gap_late", "total_model", "total_over", "outright"):
        show_fit(name, CALIB.get(name))

    d = P[P.margin != P.mkt]
    claimed = d.p_cover_home.values
    hit = (d.margin.values > d.mkt.values).astype(float)
    print("\n  the spread, in plain numbers:")
    print(f"    {'model says':>14} {'games':>7} {'actually':>10}")
    for lo, hi in ((0, .35), (.35, .45), (.45, .50), (.50, .55), (.55, .65), (.65, 1.0)):
        sel = (claimed >= lo) & (claimed < hi)
        if sel.sum() < 40:
            continue
        print(f"    {100*claimed[sel].mean():>13.1f}% {int(sel.sum()):>7} "
              f"{100*hit[sel].mean():>9.1f}%")


keep("calibration", calibration_report)


# ============================================================ confidence floor
def confidence_floor():
    """Is there a probability below which the price cannot rescue a call?

    Measured, where the evidence supports one: moneyline bets bucketed by
    calibrated probability and judged on return. The floor is the top of the
    longest opening run of buckets whose WHOLE interval sits below zero. The
    moment a bucket is merely bad, or undecided, the run stops. "Probably
    loses" is not evidence enough to cap a grade at D.
    """
    if R7 is None or len(R7) < 100:
        return {}
    buckets = []
    edges = [0, 35, 40, 45, 50, 100]
    pc = R7.p_cal.values * 100
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = R7[(pc >= lo) & (pc < hi)]
        if len(s) < 60:
            continue
        roi = float(s.ret.mean()) * 100
        se = float(s.ret.std(ddof=1)) / np.sqrt(len(s)) * 100
        w = int(s.won.sum())
        buckets.append(dict(lo=lo, hi=hi, n=int(len(s)), w=w, l=int(len(s) - w),
                            roi=round(roi, 2),
                            interval=[round(roi - 1.96 * se, 2), round(roi + 1.96 * se, 2)]))
    measured = None
    for b in buckets:
        if b["interval"][1] < 0:
            measured = b["hi"]
        else:
            break
    losers = ["%d-%d%%" % (b["lo"], b["hi"]) for b in buckets if b["interval"][1] < 0]
    if measured is None:
        note = ("no bucket at the BOTTOM lost with its whole interval below zero, so "
                "there is no measured floor and the chosen one stands")
        if losers:
            note += (". The buckets that did lose outright were " + ", ".join(losers)
                     + ", which a floor cannot reach")
    else:
        note = f"every bucket below {measured}% lost with its whole 95% interval below zero"
    print("\n  confidence floor. Moneyline bets by calibrated probability, on return:")
    print(f"    {'bucket':<14} {'bets':>6} {'record':>12} {'return':>8}   95% interval")
    for b in buckets:
        print(f"    {'%d-%d%%' % (b['lo'], b['hi']):<14} {b['n']:>6} "
              f"{'%d-%d' % (b['w'], b['l']):>12} {b['roi']:>+7.1f}%   "
              f"({b['interval'][0]:+.1f} to {b['interval'][1]:+.1f})")
    print(f"    -> {note}")
    return dict(buckets=buckets, measured=measured, note=note,
                basis="moneyline bets, bucketed by calibrated probability, judged on return")


CONF_FLOOR = keep("confidence floor", confidence_floor) or {}


# ============================================================ what it all means
def summarise():
    print("\n" + "=" * 84)
    print("WHAT THE CARD IS ALLOWED TO DO")
    print("=" * 84)
    for name in sorted(RECORD):
        w, l = RECORD[name]
        if name in RETURNS:
            r = RETURNS[name]
            t = ("PLAY" if r["lo"] > 0 else "AVOID" if r["hi"] < 0
                 else "LEAN" if r["roi"] > 0 else "PASS")
            print(f"  {name:<20} {w}-{l:<6} {r['roi']:>+6.1f}% a unit  "
                  f"{r['lo']:>+6.1f} to {r['hi']:<+6.1f}  {t}")
        else:
            lo, hi = wilson(w, w + l)
            print(f"  {name:<20} {w}-{l:<6} {100*w/max(w+l,1):>6.1f}%          "
                  f"{lo:>5.1f} to {hi:<5.1f}  {tier(w, l)}")
    plays = [n for n, (w, l) in RECORD.items()
             if (RETURNS[n]["lo"] > 0 if n in RETURNS else tier(w, l) == "PLAY")]
    print()
    if plays:
        print(f"  Strategies that may be graded as picks: {', '.join(plays)}")
    else:
        print("  NOTHING clears break-even. No strategy on this card may be graded as a")
        print("  pick. The card keeps showing its numbers and makes no claim about them,")
        print("  which is the honest outcome and not a failure of the build.")
    c = CALIB.get("model_gap_late") or {}
    if c and not c.get("k_clears_zero"):
        print()
        print(f"  The spread confidence fits a slope of {c['k']:+.3f}, which does not clear")
        print("  zero. A model whose stated confidence is uncorrelated with what happens")
        print("  cannot be allowed to print an A. nfl_picks.py caps every grade by the")
        print("  record behind it and the card shows the cap rather than hiding it.")


keep("summary", summarise)


# ============================================================ write it back
def write_state():
    payload = {
        "run_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": f"{SEASONS[0]}-{SEASONS[-1]} regular season, {len(P):,} games, "
                  f"walk-forward against real closing lines",
        "record": {k: [int(v[0]), int(v[1])] for k, v in RECORD.items()},
        "returns": RETURNS,
        "calibration": {k: v for k, v in CALIB.items() if v},
        "conf_floor": CONF_FLOOR,
        "record_notes": {
            "model_gap_early": "our number against the closing spread, weeks 1 to 4",
            "model_gap_late": "our number against the closing spread, week 5 onward",
            "total_model": "our total against the closing total",
            "total_over": "taking every over, which uses no model input",
            "moneyline": "betting the outright winner, judged on return per unit staked",
            "moneyline_fav": "the same, only where we take the favourite",
            "moneyline_dog": "the same, only where we take the underdog",
        },
        "headline": {
            "margin_mae": round(float((P.margin - P.model).abs().mean()), 3),
            "market_margin_mae": round(float((P.margin - P.mkt).abs().mean()), 3),
            "total_mae": round(float((P.total - P.model_total).abs().mean()), 3),
            "market_total_mae": round(float((P.total - P.total_line).abs().mean()), 3),
            "blend_k": BEST_K,
            "blend_grid": BLEND_TABLE,
            "home_field_measured": round(float(HOME_FIELD_MEASURED), 3),
        },
    }
    # replace, not merge: a strategy that stops being measured must not keep an
    # old record in the file that nothing updates. See model_state.save.
    MS.save("backtest", payload, replace=True)
    if BEST_K is not None:
        MS.save("ratings", {"in_season_k": int(BEST_K)})
        print(f"  in-season blend k = {BEST_K}: the in-season fit carries weight "
              f"n/(n+{BEST_K}) after n games")


write_state()
print("\ndone.")
