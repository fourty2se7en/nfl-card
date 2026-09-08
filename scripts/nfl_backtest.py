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
DEF_HFA, AVG_PTS, SD_BASE = 1.5, 22.4, 13.2
TOTAL_SD = 17.0          # scatter of an actual total around a projected one
BREAK_EVEN = 52.38       # what -110 needs

# ---- backtest-only ----
FIRST_PRIOR = 2012                       # fitted only to give 2013 a prior
SEASONS = list(range(2013, 2026))
BLEND_GRID = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 30, None]   # None = never update

print("constants in use:")
print(f"  half-life {HALF_LIFE}  ridge {RIDGE_ALPHA}  early-down {EARLY_DOWN}  "
      f"epa weight {EPA_W}  {PLAYS_PG} plays")
print(f"  regress {1-OFF_KEEP:.2f}/{1-DEF_KEEP:.2f}   game sd {SD_BASE}   "
      f"break-even {BREAK_EVEN}%")
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
    print(f"  home edge measured from the residuals: {zero.resid.mean():+.2f} points")
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


test_home_field()


# ============================================================ the market tests
def test_spread():
    print("\n" + "=" * 84)
    print(f"TEST 1 - AGAINST THE CLOSING SPREAD   (break-even {BREAK_EVEN}%)")
    print("=" * 84)
    d = P[P.margin != P.mkt]
    cov = (d.margin > d.mkt).values
    e = (d.model - d.mkt).values
    win = ((e > 0) & cov) | ((e <= 0) & ~cov)
    print(f"  {'disagreement':<26} {'games':>6} {'record':>13} {'win%':>8}   verdict")
    rec = {}
    for lo, hi, lab, key in ((0, 99, "any", "spread_any"), (0, 3, "0 to 3", None),
                             (3, 6, "3 to 6", "spread_3"), (6, 10, "6 to 10", "spread_6"),
                             (10, 99, "10 or more", "spread_10")):
        r = report((np.abs(e) >= lo) & (np.abs(e) < hi), win, lab)
        if r and key:
            rec[key] = r
    print(f"  our margin misses by {float((P.margin-P.model).abs().mean()):.2f} points, "
          f"the market's by {float((P.margin-P.mkt).abs().mean()):.2f}")
    return d, e, win, rec


def test_totals():
    print("\n" + "=" * 84)
    print("TEST 2 - TOTALS, against the closing total")
    print("=" * 84)
    t = P[P.total != P.total_line]
    over = (t.total > t.total_line).values
    e = (t.model_total - t.total_line).values
    win = ((e > 0) & over) | ((e <= 0) & ~over)
    print(f"  {'disagreement':<26} {'games':>6} {'record':>13} {'win%':>8}   verdict")
    rec = {}
    for lo, hi, lab, key in ((0, 99, "any", "total_any"), (0, 3, "0 to 3", None),
                             (3, 6, "3 to 6", "total_3"), (6, 99, "6 or more", "total_6")):
        r = report((np.abs(e) >= lo) & (np.abs(e) < hi), win, lab)
        if r and key:
            rec[key] = r
    print(f"  our total misses by {float((t.total-t.model_total).abs().mean()):.2f} points, "
          f"the market's by {float((t.total-t.total_line).abs().mean()):.2f}")
    print(f"  every over, for reference: {int(over.sum())}-{int((~over).sum())} "
          f"({100*over.mean():.1f}%)")
    return rec


def test_subgroups(d, e, win):
    print("\n" + "=" * 84)
    print("TEST 3 - SUBGROUPS, against the closing spread")
    print("=" * 84)
    print(f"  {'group':<26} {'games':>6} {'record':>13} {'win%':>8}   verdict")
    rec = {}
    wk = d.week.values
    for mask, lab, key in ((wk <= 4, "weeks 1-4", "spread_wk1_4"),
                           ((wk > 4) & (wk <= 9), "weeks 5-9", None),
                           (wk > 9, "weeks 10+", "spread_wk10"),
                           (d["div"].values, "divisional", "spread_div"),
                           (~d["div"].values, "non-divisional", None),
                           (d.mkt.abs().values <= 3, "line 3 or less", None),
                           (d.mkt.abs().values >= 7, "line 7 or more", None),
                           ((np.abs(e) >= 3) & (wk > 4), "3+ gap, week 5+", None)):
        r = report(mask, win, lab)
        if r and key:
            rec[key] = r
    return rec


def amp(x):
    """A moneyline as an implied probability, vig included."""
    return 100 / (x + 100) if x > 0 else -x / (-x + 100)


def test_moneyline():
    print("\n" + "=" * 84)
    print("TEST 4 - MONEYLINE, measured by what it returns")
    print("=" * 84)
    m = P[P.hml.notna() & P.aml.notna()].copy()
    ph, pa = m.hml.map(amp), m.aml.map(amp)
    fair = ph / (ph + pa)
    ours = 1 - norm.cdf(0, m.model, SD_BASE)
    val = (ours - fair).values
    rec = {}
    print(f"  {'our edge over the price':<26} {'bets':>6} {'record':>13} {'win%':>8}   return")
    for lo, key in ((0.02, "ml_2"), (0.05, "ml_5"), (0.10, "ml_10")):
        sel = np.abs(val) >= lo
        if sel.sum() < 60:
            continue
        s = m[sel]
        back_home = val[sel] > 0
        line = np.where(back_home, s.hml.values, s.aml.values)
        won = np.where(back_home, s.margin.values > 0, s.margin.values < 0)
        pay = np.where(line > 0, line / 100.0, 100.0 / (-line))
        roi = np.where(won, pay, -1.0)
        w, l = int(won.sum()), int(len(s) - won.sum())
        print(f"  {'>= ' + str(int(lo*100)) + ' points of probability':<26} {len(s):>6} "
              f"{f'{w}-{l}':>13} {100*w/len(s):>7.1f}%   {100*roi.mean():+.1f}% a unit")
        rec[key] = (w, l)
    print("  A moneyline is not a coin flip, so win% alone says nothing here; the")
    print("  return is the number that matters.")
    return rec


def test_price_consistency():
    print("\n" + "=" * 84)
    print("TEST 5 - PRICE CONSISTENCY. The moneyline against the posted spread.")
    print("=" * 84)
    c = P[P.hml.notna() & P.aml.notna() & (P.margin != P.mkt)].copy()
    ph, pa = c.hml.map(amp), c.aml.map(amp)
    fair = np.clip(ph / (ph + pa), 0.001, 0.999)
    gap = (norm.ppf(fair) * SD_BASE - c.mkt).values
    print(f"  quotes: {len(c):,}   average gap {gap.mean():+.2f} points, "
          f"scatter {gap.std():.2f}")
    cov = (c.margin > c.mkt).values
    win = ((gap > 0) & cov) | ((gap <= 0) & ~cov)
    print(f"  {'threshold':<26} {'bets':>6} {'record':>13} {'win%':>8}   verdict")
    rec = {}
    for thr in (1.0, 1.5, 2.0, 3.0):
        r = report(np.abs(gap) >= thr, win, f"{thr} points or more")
        if r and thr == 1.5:
            rec["price_consistency"] = r
    print("  This is the one angle that survived on the college card, and it does not")
    print("  transfer. There it compares EACH BOOK's moneyline with THAT BOOK's own")
    print("  spread, and a stale one shows up. nflverse publishes one consensus number")
    print("  for each market, and a consensus is internally consistent by construction:")
    print(f"  the gap scatters only {gap.std():.2f} points. Testing this properly needs")
    print("  per-book prices, which this card does not have.")
    return rec


D, E, WIN1, REC = test_spread()
REC.update(test_totals())
REC.update(test_subgroups(D, E, WIN1))
REC.update(test_moneyline())
REC.update(test_price_consistency())

# A section that produced nothing must say so. The college run has this guard,
# and it is the only reason a whole test that had silently stopped running was
# ever noticed: the job still reported success. See 4.3j.
_expected = {"spread_any", "total_any", "spread_wk10", "ml_5"}
_missing = sorted(_expected - set(REC))
if _missing:
    sys.exit(f"ERROR: these strategies produced no record at all: {', '.join(_missing)}. "
             f"A test that quietly stops running still leaves the run green, so this "
             f"stops instead of writing a partial record over a good one.")


# ============================================================ calibration
def calibrate():
    """Does the model's stated confidence mean anything?

    The card grades mostly on a simulated win probability. If that probability
    carries no information about what actually happened, every letter on the
    page is a claim with nothing behind it, and the grades have to be capped.
    The college card needed exactly this and calls it the fourth gate.
    """
    print("\n" + "=" * 84)
    print("CALIBRATION - what the model claims against what happened")
    print("=" * 84)
    out = {}
    for name, frame, line, sd, hit_col, mod_col in (
            ("spread", P[P.margin != P.mkt], "mkt", SD_BASE, "margin", "model"),
            ("total", P[P.total != P.total_line], "total_line", TOTAL_SD, "total", "model_total")):
        claimed = 1 - norm.cdf(frame[line].values, frame[mod_col].values, sd)
        hit = (frame[hit_col].values > frame[line].values).astype(float)
        print(f"\n  {name}:")
        print(f"    {'model says':>14} {'games':>7} {'actually':>10}")
        for lo, hi in ((0, .35), (.35, .45), (.45, .50), (.50, .55), (.55, .65), (.65, 1.0)):
            sel = (claimed >= lo) & (claimed < hi)
            if sel.sum() < 40:
                continue
            print(f"    {100*claimed[sel].mean():>13.1f}% {int(sel.sum()):>7} "
                  f"{100*hit[sel].mean():>9.1f}%")
        x = norm.ppf(np.clip(claimed, .01, .99))

        def nll(t):
            p = np.clip(norm.cdf(t[0] + t[1] * x), 1e-9, 1 - 1e-9)
            return -float(np.sum(hit * np.log(p) + (1 - hit) * np.log(1 - p)))

        fit = minimize(nll, [0.0, 1.0], method="Nelder-Mead")
        a, b = float(fit.x[0]), float(fit.x[1])
        h = 1e-4
        H = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                e1 = np.zeros(2); e1[i] = h
                e2 = np.zeros(2); e2[j] = h
                H[i, j] = (nll(fit.x + e1 + e2) - nll(fit.x + e1 - e2)
                           - nll(fit.x - e1 + e2) + nll(fit.x - e1 - e2)) / (4 * h * h)
        try:
            se = float(np.sqrt(np.diag(np.linalg.inv(H)))[1])
        except Exception:
            se = float("nan")
        print(f"    slope {b:+.3f}  (95% {b-1.96*se:+.3f} to {b+1.96*se:+.3f}; "
              f"1.0 is perfectly calibrated, 0.0 carries no information)")
        out[name] = {"intercept": round(a, 4), "slope": round(b, 4),
                     "slope_se": round(se, 4), "n": int(len(x))}
    return out


CALIB = calibrate()


# ============================================================ what it all means
def summarise():
    print("\n" + "=" * 84)
    print("WHAT THE CARD IS ALLOWED TO DO")
    print("=" * 84)
    tiers = {k: tier(*v) for k, v in REC.items()}
    for name, (w, l) in sorted(REC.items()):
        lo, hi = wilson(w, w + l)
        print(f"  {name:<22} {w}-{l:<6} {100*w/max(w+l,1):>6.1f}%  "
              f"{lo:>5.1f} to {hi:<5.1f}  {tiers[name]}")
    plays = [k for k, v in tiers.items() if v == "PLAY"]
    print()
    if plays:
        print(f"  Strategies that may be graded as picks: {', '.join(plays)}")
    else:
        print("  NOTHING clears break-even. No strategy on this card may be graded as a")
        print("  pick. The card keeps showing its numbers and makes no claim about them,")
        print("  which is the honest outcome and not a failure of the build.")
    s = CALIB.get("spread", {})
    if s and abs(s.get("slope", 0)) < 0.3:
        print()
        print(f"  The simulated win probability has a calibration slope of "
              f"{s['slope']:+.3f}. A model whose stated confidence is uncorrelated with")
        print("  what happens cannot be allowed to print an A. Every grade must be")
        print("  capped by the record of the strategy behind it, exactly as the college")
        print("  card does it, and the cap must be visible on the page.")
    return tiers


TIERS = summarise()


# ============================================================ write it back
def write_state(tiers):
    payload = {
        "run_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "window": f"{SEASONS[0]}-{SEASONS[-1]} regular season, {len(P):,} games, "
                  f"walk-forward against the closing line",
        "record": {k: [int(v[0]), int(v[1])] for k, v in REC.items()},
        "tier": tiers,
        "record_notes": {
            "spread_any": "our number against the closing spread, any disagreement",
            "spread_3": "disagreement of 3 to 6 points",
            "spread_6": "disagreement of 6 to 10 points",
            "spread_10": "disagreement of 10 points or more",
            "spread_wk1_4": "weeks 1 to 4, when the preseason number carries most weight",
            "spread_wk10": "week 10 onward",
            "spread_div": "divisional games",
            "total_any": "our total against the closing total, any disagreement",
            "ml_5": "moneyline, 5 or more points of probability over the price",
            "price_consistency": "consensus moneyline against the consensus spread; "
                                 "needs per-book prices to mean anything",
        },
        "calibration": CALIB,
        "headline": {
            "margin_mae": round(float((P.margin - P.model).abs().mean()), 3),
            "market_margin_mae": round(float((P.margin - P.mkt).abs().mean()), 3),
            "total_mae": round(float((P.total - P.model_total).abs().mean()), 3),
            "market_total_mae": round(float((P.total - P.total_line).abs().mean()), 3),
            "blend_k": BEST_K,
            "blend_grid": BLEND_TABLE,
        },
    }
    MS.save("backtest", payload)
    if BEST_K is not None:
        MS.save("ratings", {"in_season_k": int(BEST_K)})
        print(f"  model_state.json: in-season blend k = {BEST_K} "
              f"(the in-season fit gets weight n/(n+{BEST_K}) after n games)")


write_state(TIERS)
print("\ndone.")
