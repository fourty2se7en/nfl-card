"""
build_ratings.py — opponent-adjusted power ratings from nflverse play-by-play.
Implements Section 1 of the master model instructions.
"""
import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import os, sys
BASE = os.environ.get("NFL_DIR") or os.path.dirname(os.path.abspath(__file__))
P = lambda x: os.path.join(BASE, x)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_state as MS
import nfl_fit as FIT

# Every constant comes from model_state.json. They used to be literals here, and
# two of them were ALSO written into nfl_card.py as 0.65 and 0.55, which is
# 1 - OFF_REGRESS and 1 - DEF_REGRESS. That file recomputed the Prior2026 column
# this one already writes, from its own copy of the numbers. Nothing would have
# failed if the two drifted; the run goes green and the page is just wrong.
SEASONS = [int(x) for x in MS.RATINGS["seasons"]]
HALF_LIFE = float(MS.RATINGS["half_life"])        # weeks, decay within a season
OFF_REGRESS = float(MS.RATINGS["off_regress"])    # regress toward mean for the prior
DEF_REGRESS = float(MS.RATINGS["def_regress"])
PLAYS_PER_GAME = float(MS.RATINGS["plays_per_game"])
RIDGE_ALPHA = float(MS.RATINGS["ridge_alpha"])
EARLY_DOWN_W = float(MS.RATINGS["early_down_weight"])
EPA_W = float(MS.RATINGS["epa_weight"])           # success rate takes the rest
WEIGHTS = dict(MS.RATINGS["composite"])
print(f"constants from model_state: seasons {SEASONS}, half-life {HALF_LIFE}, "
      f"regress {OFF_REGRESS}/{DEF_REGRESS}, {PLAYS_PER_GAME} plays, "
      f"ridge {RIDGE_ALPHA}"
      + (f"   WARNING: {MS.FELL_BACK}" if MS.FELL_BACK else ""))

pbp = nfl.load_pbp(SEASONS).to_pandas()
df = pbp[(pbp.season_type == "REG")].copy()

# The fit itself lives in nfl_fit.py, shared with nfl_backtest.py. It was moved
# there BEFORE the backtest was written, while there was still only one copy.
# On the college card the same algorithm ended up in two files, drifted, and the
# backtest spent a season measuring a model the card did not publish. See 4.3n.
R = FIT.fit_ratings(df, half_life=HALF_LIFE, ridge_alpha=RIDGE_ALPHA,
                    early_down_weight=EARLY_DOWN_W, epa_weight=EPA_W,
                    plays_per_game=PLAYS_PER_GAME, composite=WEIGHTS)
share_pass = float(R.share_pass.iloc[0])
R = R.rename(columns={"off_pt": "off_epa_pt", "def_pt": "def_epa_pt",
                      "rating": "rating_2025"})
# the same subset the fit used, kept for the pass-rate and turnover columns below
df = df[(df.posteam.notna()) & (df.epa.notna())]
df = df[(df["pass"] == 1) | (df["rush"] == 1)]

# --- 2026 priors: regress toward mean ---
R["off_2026"] = R.off_epa_pt * (1 - OFF_REGRESS)
R["def_2026"] = R.def_epa_pt * (1 - DEF_REGRESS)
R["rating_2026_prior"] = R.off_2026 + R.def_2026

# scheme: pass rate on early downs
neu = df[df.down.isin([1,2])]
R["pass_rate"] = neu.groupby("posteam").apply(lambda d: (d["pass"]==1).mean()).reindex(R.index)

# unit splits in points (for pass/rush offense and defense ranks)
R["pass_off_pt"] = R.pass_off_epa * PLAYS_PER_GAME * share_pass
R["rush_off_pt"] = R.rush_off_epa * PLAYS_PER_GAME * (1 - share_pass)
R["pass_def_pt"] = -R.pass_def_epa * PLAYS_PER_GAME * share_pass
R["rush_def_pt"] = -R.rush_def_epa * PLAYS_PER_GAME * (1 - share_pass)

# turnovers: giveaways per game (offense) and takeaways per game (defense)
full = pbp[(pbp.season_type=="REG") & pbp.posteam.notna()].copy()
full["to"] = ((full.interception.fillna(0)==1) | (full.fumble_lost.fillna(0)==1)).astype(int)
gpt = full.groupby("posteam").game_id.nunique()
give = full.groupby("posteam")["to"].sum() / gpt
take = full.groupby("defteam")["to"].sum() / full.groupby("defteam").game_id.nunique()
R["giveaways"] = give.reindex(R.index)
R["takeaways"] = take.reindex(R.index)

out = R[["rating_2025","off_epa_pt","def_epa_pt","rating_2026_prior","pass_rate",
         "pass_off_pt","rush_off_pt","pass_def_pt","rush_def_pt","giveaways","takeaways"]].copy()
out.columns = ["Rating2025","Off2025","Def2025","Prior2026","PassRate",
               "PassOff","RushOff","PassDef","RushDef","Giveaways","Takeaways"]
out = out.sort_values("Prior2026", ascending=False).round(4)
out.to_csv(P("power_ratings.csv"))

print(f"plays used: {len(df):,}   pass share: {share_pass:.3f}")
print(f"2025 rating spread: {R.rating_2025.min():.1f} to {R.rating_2025.max():.1f}")
print(f"2026 prior spread:  {R.rating_2026_prior.min():.1f} to {R.rating_2026_prior.max():.1f}\n")
print(out.to_string())
