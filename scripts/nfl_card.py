"""
nfl_card.py — NFL slate card + ledger on one page with a view toggle.

Usage:
    python3 nfl_card.py                     # next upcoming week
    python3 nfl_card.py --week 6
    python3 nfl_card.py --game "KC @ BUF"
    python3 nfl_card.py --notes notes.json  # attach researched news / expert picks

notes.json (all keys optional), keyed by "AWAY@HOME":
{
  "KC@BUF": {
     "returning":  ["WR Jones - ACL, first game back"],
     "trades":     ["Acquired OT Brown from CLE on Tuesday"],
     "coaching":   ["OC Miller fired Monday; QB coach calling plays"],
     "suspensions":["CB Davis - 2 games, 1 remaining"],
     "birthdays":  ["RB Allen turns 26 on gameday"],
     "movement":   ["Opened KC -2.5, now -4 on Buffalo OL news"],
     "experts":    [{"name":"J. Smith (Action Network)","record":"31-24 ATS","pick":"KC -4"}]
  }
}
"""
import os, sys, json, argparse, datetime as dt
import numpy as np, pandas as pd
from scipy.stats import norm
import nflreadpy as nfl
import warnings; warnings.filterwarnings("ignore")

BASE = os.environ.get("NFL_DIR") or os.path.dirname(os.path.abspath(__file__))
P = lambda f: os.path.join(BASE, f)

ISSUES = []
def issue(lvl, what, why, fix, tab="card"):
    ISSUES.append(dict(lvl=lvl, what=what, why=why, fix=fix, tab=tab)); print(f"  [{lvl}] {what}")

ap = argparse.ArgumentParser()
ap.add_argument("--week", type=int, default=None)
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--game", type=str, default=None)
ap.add_argument("--notes", type=str, default=None)
ap.add_argument("--no-weather", action="store_true")
A = ap.parse_args()
SEASON = A.season

MEASURED = {"spread": 48.8, "total": 50.5, "ml_roi": -12.7, "be": 52.4}
# Public repo URLs only. No token is ever embedded — a token in a public page
# is readable by anyone who views it.
REPO = os.environ.get("NFL_REPO", "fourty2se7en/nfl-card")
WORKFLOW = os.environ.get("NFL_WORKFLOW", "nfl-card.yml")
SD_BASE = 13.2
N_SIMS = 20000
RATING_SE = 3.0   # uncertainty in our own line, in points
RNG = np.random.default_rng(20260904)
HFA = {'SEA':2.2,'KC':2.2,'DEN':2.2,'BUF':2.2,'NO':2.0,'GB':2.0,'BAL':2.0,'PIT':2.0,
       'LA':1.0,'LAC':0.8,'JAX':1.0,'LV':0.8,'ATL':1.2}
DEF_HFA, AVG_PTS = 1.5, 22.4
LL = {'ARI':(33.528,-112.263),'ATL':(33.755,-84.401),'BAL':(39.278,-76.623),'BUF':(42.774,-78.787),
 'CAR':(35.226,-80.853),'CHI':(41.863,-87.617),'CIN':(39.095,-84.516),'CLE':(41.506,-81.699),
 'DAL':(32.748,-97.093),'DEN':(39.744,-105.020),'DET':(42.340,-83.046),'GB':(44.501,-88.062),
 'HOU':(29.685,-95.411),'IND':(39.760,-86.164),'JAX':(30.324,-81.637),'KC':(39.049,-94.484),
 'LA':(33.953,-118.339),'LAC':(33.953,-118.339),'LV':(36.091,-115.184),'MIA':(25.958,-80.239),
 'MIN':(44.974,-93.258),'NE':(42.091,-71.264),'NO':(29.951,-90.081),'NYG':(40.814,-74.074),
 'NYJ':(40.814,-74.074),'PHI':(39.901,-75.168),'PIT':(40.447,-80.016),'SEA':(47.595,-122.332),
 'SF':(37.403,-121.970),'TB':(27.976,-82.503),'TEN':(36.166,-86.771),'WAS':(38.908,-76.864)}
amp = lambda o: 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)

TEAMCOLOR = {
 'ARI':'#97233F','ATL':'#A71930','BAL':'#241773','BUF':'#00338D','CAR':'#0085CA','CHI':'#0B162A',
 'CIN':'#FB4F14','CLE':'#E75B00','DAL':'#041E42','DEN':'#FB4F14','DET':'#0076B6','GB':'#203731',
 'HOU':'#03202F','IND':'#002C5F','JAX':'#006778','KC':'#E31837','LA':'#003594','LAC':'#0080C6',
 'LV':'#333333','MIA':'#008E97','MIN':'#4F2683','NE':'#002244','NO':'#9F8958','NYG':'#0B2265',
 'NYJ':'#125740','PHI':'#004C54','PIT':'#B8901A','SEA':'#0C5C86','SF':'#AA0000','TB':'#D50A0A',
 'TEN':'#0C2340','WAS':'#5A1414'}

# secondary/alternate colors — used in dark mode where the primary is too dark to glow
TEAMCOLOR2 = {
 'ARI':'#FFB612','ATL':'#A71930','BAL':'#9E7C0C','BUF':'#C60C30','CAR':'#0085CA','CHI':'#C83803',
 'CIN':'#FB4F14','CLE':'#FF3C00','DAL':'#0B4C9E','DEN':'#FB4F14','DET':'#0076B6','GB':'#FFB612',
 'HOU':'#A71930','IND':'#0B63B8','JAX':'#D7A22A','KC':'#FFB81C','LA':'#FFA300','LAC':'#FFC20E',
 'LV':'#A5ACAF','MIA':'#00C4CF','MIN':'#FFC62F','NE':'#C60C30','NO':'#D3BC8D','NYG':'#A71930',
 'NYJ':'#1E9E6A','PHI':'#00A19B','PIT':'#FFB612','SEA':'#69BE28','SF':'#C9A860','TB':'#FF7900',
 'TEN':'#4B92DB','WAS':'#B33A4E'}

def _rgb(hexc):
    h = hexc.lstrip('#'); return tuple(int(h[i:i+2],16) for i in (0,2,4))

def _lum(rgb):
    """WCAG relative luminance."""
    def ch(c):
        c = c/255.0
        return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4
    r_,g_,b_ = (ch(v) for v in rgb)
    return 0.2126*r_ + 0.7152*g_ + 0.0722*b_

def _blend(rgb, amt):
    return tuple(int(v + (255-v)*amt) for v in rgb)

def _for_bg(hexc, target_lum, lighten=True):
    """Lift (or darken) a color until it reads clearly against the background."""
    rgb = _rgb(hexc)
    for i in range(21):
        amt = i/20.0
        cand = _blend(rgb, amt) if lighten else tuple(int(v*(1-amt)) for v in rgb)
        if (_lum(cand) >= target_lum) if lighten else (_lum(cand) <= target_lum):
            return '#%02X%02X%02X' % cand
    return '#%02X%02X%02X' % (_blend(rgb, 1.0) if lighten else (0,0,0))

import colorsys as _cs
def _set_light(hexc, L, min_sat=0.45, min_lum=None):
    """Move a color to a target lightness in HSL, keeping its hue.
    Near-greyscale colors stay grey. Optionally raise L until a luminance floor is met."""
    r_,g_,b_ = (v/255 for v in _rgb(hexc))
    h_, l0, s0 = _cs.rgb_to_hls(r_, g_, b_)
    s_ = s0 if s0 < 0.12 else max(s0, min_sat)      # keep greys grey
    def mk(LL):
        r2,g2,b2 = _cs.hls_to_rgb(h_, LL, s_)
        return (int(r2*255), int(g2*255), int(b2*255))
    rgb = mk(L)
    if min_lum is not None:
        LL = L
        while _lum(rgb) < min_lum and LL < 0.92:
            LL += 0.02; rgb = mk(LL)
    return '#%02X%02X%02X' % rgb

def _neon(hexc, L=0.60, sat=0.88, min_lum=0.30):
    """Vivid dark-mode treatment: keep the hue, push saturation, set a bright lightness.
    Greys stay grey. Matches the glow of naturally bright colors like Denver orange."""
    r_,g_,b_ = (v/255 for v in _rgb(hexc))
    h_, l0, s0 = _cs.rgb_to_hls(r_, g_, b_)
    s_ = s0 if s0 < 0.12 else max(s0, sat)
    def mk(LL):
        r2,g2,b2 = _cs.hls_to_rgb(h_, LL, s_)
        return (int(r2*255), int(g2*255), int(b2*255))
    rgb = mk(L); LL = L
    while _lum(rgb) < min_lum and LL < 0.90:
        LL += 0.02; rgb = mk(LL)
    return '#%02X%02X%02X' % rgb

def _sat(hexc):
    r_,g_,b_ = (v/255 for v in _rgb(hexc))
    return _cs.rgb_to_hls(r_,g_,b_)[2]

def _vivid(hexc, L, sat=0.90):
    """Set lightness and push saturation, keeping hue. Greys stay grey."""
    r_,g_,b_ = (v/255 for v in _rgb(hexc))
    h_, l0, s0 = _cs.rgb_to_hls(r_, g_, b_)
    s_ = s0 if s0 < 0.12 else max(s0, sat)
    r2,g2,b2 = _cs.hls_to_rgb(h_, L, s_)
    return '#%02X%02X%02X' % (int(r2*255), int(g2*255), int(b2*255))

def _pick(team, L, min_lum=None, max_lum=None):
    """Choose whichever of the two team colors renders most vividly at this lightness.
    The two modes are picked independently — a team can use its primary on one
    background and its secondary on the other."""
    # primary first — it is the team's identity. Secondary only if the primary
    # cannot be made readable at this lightness on this background.
    cands = [TEAMCOLOR[team], TEAMCOLOR2.get(team, TEAMCOLOR[team])]
    best = None
    for c in cands:
        v = _vivid(c, L)
        if min_lum is not None and _lum(_rgb(v)) < min_lum: continue
        if max_lum is not None and _lum(_rgb(v)) > max_lum: continue
        best = v; break
    if best is None:
        # neither color cleared the bar — walk the primary's lightness until it does
        best, LL = _vivid(cands[0], L), L
        if min_lum is not None:
            while _lum(_rgb(best)) < min_lum and LL < 0.90:
                LL += 0.02; best = _vivid(cands[0], LL)
        if max_lum is not None:
            while _lum(_rgb(best)) > max_lum and LL > 0.14:
                LL -= 0.02; best = _vivid(cands[0], LL)
    return best

# Thresholds come from contrast ratio against the actual panel color, not raw
# luminance — red is inherently low-luminance and a flat floor rejects it unfairly.
_BG_DARK, _BG_LIGHT = 0.014, 0.97          # luminance of the dark and light panels
_MIN_DARK  = 4.2*(_BG_DARK+0.05) - 0.05    # >= 4.2:1 against the dark panel
_MAX_LIGHT = (_BG_LIGHT+0.05)/4.2 - 0.05   # >= 4.2:1 against the light panel

# Each mode picks independently: primary first, secondary only if it cannot be read.
TEAMCOLOR_DARK  = {k: _pick(k, 0.63, min_lum=_MIN_DARK)  for k in TEAMCOLOR}
TEAMCOLOR_LIGHT = {k: _pick(k, 0.36, max_lum=_MAX_LIGHT) for k in TEAMCOLOR}

DIV = {
 'BUF':'AFC East','MIA':'AFC East','NE':'AFC East','NYJ':'AFC East',
 'BAL':'AFC North','CIN':'AFC North','CLE':'AFC North','PIT':'AFC North',
 'HOU':'AFC South','IND':'AFC South','JAX':'AFC South','TEN':'AFC South',
 'DEN':'AFC West','KC':'AFC West','LAC':'AFC West','LV':'AFC West',
 'DAL':'NFC East','NYG':'NFC East','PHI':'NFC East','WAS':'NFC East',
 'CHI':'NFC North','DET':'NFC North','GB':'NFC North','MIN':'NFC North',
 'ATL':'NFC South','CAR':'NFC South','NO':'NFC South','TB':'NFC South',
 'ARI':'NFC West','LA':'NFC West','SEA':'NFC West','SF':'NFC West'}
TEAMNAME = {
 'ARI':'Cardinals','ATL':'Falcons','BAL':'Ravens','BUF':'Bills','CAR':'Panthers','CHI':'Bears',
 'CIN':'Bengals','CLE':'Browns','DAL':'Cowboys','DEN':'Broncos','DET':'Lions','GB':'Packers',
 'HOU':'Texans','IND':'Colts','JAX':'Jaguars','KC':'Chiefs','LA':'Rams','LAC':'Chargers',
 'LV':'Raiders','MIA':'Dolphins','MIN':'Vikings','NE':'Patriots','NO':'Saints','NYG':'Giants',
 'NYJ':'Jets','PHI':'Eagles','PIT':'Steelers','SEA':'Seahawks','SF':'49ers','TB':'Buccaneers',
 'TEN':'Titans','WAS':'Commanders'}

# Different sources abbreviate teams differently — LAR vs LA, WSH vs WAS, JAC vs
# JAX, plus legacy codes like OAK and SD. A notes key that does not match a game
# would otherwise vanish with no warning, so every reasonable spelling is mapped
# to the canonical code and anything still unmatched is reported on the card.
TEAM_ALIAS = {}
_ALIASES = {
 'ARI':['ARI','ARZ','AZ','ARIZONA','CARDINALS','ARIZONA CARDINALS'],
 'ATL':['ATL','ATLANTA','FALCONS','ATLANTA FALCONS'],
 'BAL':['BAL','BLT','BALTIMORE','RAVENS','BALTIMORE RAVENS'],
 'BUF':['BUF','BUFFALO','BILLS','BUFFALO BILLS'],
 'CAR':['CAR','CAROLINA','PANTHERS','CAROLINA PANTHERS'],
 'CHI':['CHI','CHICAGO','BEARS','CHICAGO BEARS'],
 'CIN':['CIN','CINCINNATI','BENGALS','CINCINNATI BENGALS'],
 'CLE':['CLE','CLV','CLEVELAND','BROWNS','CLEVELAND BROWNS'],
 'DAL':['DAL','DALLAS','COWBOYS','DALLAS COWBOYS'],
 'DEN':['DEN','DENVER','BRONCOS','DENVER BRONCOS'],
 'DET':['DET','DETROIT','LIONS','DETROIT LIONS'],
 'GB':['GB','GNB','GBP','GREEN BAY','PACKERS','GREEN BAY PACKERS'],
 'HOU':['HOU','HST','HOUSTON','TEXANS','HOUSTON TEXANS'],
 'IND':['IND','INDIANAPOLIS','COLTS','INDIANAPOLIS COLTS'],
 'JAX':['JAX','JAC','JACKSONVILLE','JAGUARS','JACKSONVILLE JAGUARS'],
 'KC':['KC','KAN','KCC','KANSAS CITY','CHIEFS','KANSAS CITY CHIEFS'],
 'LA':['LA','LAR','RAM','RAMS','LOS ANGELES RAMS','STL','ST LOUIS RAMS'],
 'LAC':['LAC','SD','SDG','CHARGERS','LOS ANGELES CHARGERS','SAN DIEGO CHARGERS'],
 'LV':['LV','LVR','OAK','RAI','RAIDERS','LAS VEGAS','LAS VEGAS RAIDERS','OAKLAND RAIDERS'],
 'MIA':['MIA','MIAMI','DOLPHINS','MIAMI DOLPHINS'],
 'MIN':['MIN','MINNESOTA','VIKINGS','MINNESOTA VIKINGS'],
 'NE':['NE','NWE','NEP','NEW ENGLAND','PATRIOTS','NEW ENGLAND PATRIOTS'],
 'NO':['NO','NOR','NOS','NEW ORLEANS','SAINTS','NEW ORLEANS SAINTS'],
 'NYG':['NYG','NEW YORK GIANTS','GIANTS'],
 'NYJ':['NYJ','NEW YORK JETS','JETS'],
 'PHI':['PHI','PHILADELPHIA','EAGLES','PHILADELPHIA EAGLES'],
 'PIT':['PIT','PITTSBURGH','STEELERS','PITTSBURGH STEELERS'],
 'SEA':['SEA','SEATTLE','SEAHAWKS','SEATTLE SEAHAWKS'],
 'SF':['SF','SFO','SAN FRANCISCO','49ERS','NINERS','SAN FRANCISCO 49ERS'],
 'TB':['TB','TAM','TBB','TAMPA BAY','BUCCANEERS','BUCS','TAMPA BAY BUCCANEERS'],
 'TEN':['TEN','TENNESSEE','TITANS','TENNESSEE TITANS'],
 'WAS':['WAS','WSH','WFT','WASHINGTON','COMMANDERS','WASHINGTON COMMANDERS'],
}
for _canon, _alist in _ALIASES.items():
    for _a in _alist:
        TEAM_ALIAS[_a.upper().replace(".", "").replace("-", " ").strip()] = _canon

def canon_team(t):
    if t is None: return None
    k = str(t).upper().replace(".", "").replace("-", " ").strip()
    return TEAM_ALIAS.get(k, TEAM_ALIAS.get(k.replace(" ", ""), None))

def canon_key(k):
    """Normalise a notes key to AWAY@HOME using canonical codes.
    Accepts @, vs, at and various separators."""
    raw = str(k).upper()
    for sep in ["@", " VS ", " V ", " AT ", "VS.", "--", "_"]:
        if sep in raw:
            parts = raw.split(sep, 1); break
    else:
        return None
    a, b = (canon_team(parts[0]), canon_team(parts[1]))
    return f"{a}@{b}" if a and b else None

NOTES = {}
if A.notes:
    if os.path.exists(P(A.notes)):
        try:
            _raw = json.load(open(P(A.notes)))
            NOTES = {}
            _unmatched = []
            for _k, _v in _raw.items():
                if _k.startswith("_"): continue
                _c = canon_key(_k)
                if _c: NOTES[_c] = _v
                else: _unmatched.append(_k)
            if _unmatched:
                issue("WARN", f"{len(_unmatched)} note(s) could not be matched to a game",
                      "These entries were researched but will not appear on any card, because "
                      f"the team names did not resolve: {', '.join(_unmatched[:6])}.",
                      "Key each game as AWAY@HOME. Most spellings are accepted automatically; "
                      "check for typos or a team that is not playing this week.", tab="card")
        except Exception as e:
            issue("WARN", f"Could not read {A.notes}",
                  "Game notes will show only automated data such as injuries, weather and rest.",
                  "Check the file is valid JSON, or omit --notes.", tab="card")
    else:
        issue("WARN", f"{A.notes} not found",
              "Game notes will show only automated data.",
              f"Create {A.notes} or omit --notes.", tab="card")

if not os.path.exists(P("power_ratings.csv")):
    sys.exit("ERROR: power_ratings.csv missing. Run: python3 build_ratings.py")
r = pd.read_csv(P("power_ratings.csv"), index_col=0)
age = (dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(P("power_ratings.csv")))).days
if age > 7:
    issue("WARN", f"Power ratings are {age} days old",
          "Ratings still reflect games from over a week ago, so every model number on this card is stale.",
          "Run: python3 build_ratings.py", tab="all")
r["off"], r["def"] = r.Off2025*0.65, r.Def2025*0.55
r["rating"] = r["off"] + r["def"]
r["off_rank"] = r["off"].rank(ascending=False).astype(int)
r["def_rank"] = r["def"].rank(ascending=False).astype(int)
r["pwr_rank"] = r["rating"].rank(ascending=False).astype(int)
if "PassRate" in r.columns:
    lo, hi = r.PassRate.quantile(.30), r.PassRate.quantile(.70)
    r["scheme"] = np.where(r.PassRate>=hi,"pass-heavy",np.where(r.PassRate<=lo,"run-heavy","balanced"))
else:
    r["PassRate"] = np.nan; r["scheme"] = "n/a"
for c,rk in [("PassOff","po_rank"),("RushOff","ro_rank"),("PassDef","pd_rank"),("RushDef","rd_rank")]:
    r[rk] = r[c].rank(ascending=False).astype(int) if c in r.columns else 0
if "Giveaways" in r.columns:
    r["gv_rank"] = r.Giveaways.rank(ascending=True).astype(int)
    r["tk_rank"] = r.Takeaways.rank(ascending=False).astype(int)
    r["to_diff"] = (r.Takeaways - r.Giveaways).round(2)
else:
    r["Giveaways"]=np.nan; r["Takeaways"]=np.nan; r["gv_rank"]=0; r["tk_rank"]=0; r["to_diff"]=np.nan
_gap = r.pd_rank - r.rd_rank
r["dsch"] = np.where(_gap <= -8, "pass-stopping", np.where(_gap >= 8, "run-stopping", "balanced D"))

try:
    ELO = pd.read_csv(P("elo_ratings.csv"), index_col=0).iloc[:,0]; ELO = (ELO-ELO.mean())/25.0
except Exception:
    ELO = None
    issue("INFO", "Elo cross-check unavailable",
          "The independent second-opinion model is not loaded, so games where the two models disagree cannot be flagged.",
          "Run: python3 elo.py", tab="card")

sch = nfl.load_schedules([SEASON]).to_pandas(); sch = sch[sch.game_type=="REG"]
if A.week is None:
    pl = sch[sch.result.notna()]
    WEEK = int(pl.week.max())+1 if len(pl) else int(sch.week.min())
else: WEEK = A.week
REC = {}
_done = sch[sch.result.notna() & (sch.week < WEEK)]
for t in DIV:
    hg = _done[_done.home_team==t]; ag = _done[_done.away_team==t]
    w = int((hg.home_score>hg.away_score).sum() + (ag.away_score>ag.home_score).sum())
    l = int((hg.home_score<hg.away_score).sum() + (ag.away_score<ag.home_score).sum())
    t_ = int((hg.home_score==hg.away_score).sum() + (ag.away_score==ag.home_score).sum())
    REC[t] = f"{w}-{l}" + (f"-{t_}" if t_ else "")

gw = sch[(sch.week==WEEK) & sch.spread_line.notna()].copy()
if len(gw)==0: sys.exit(f"No lines posted for week {WEEK} yet.")

try:
    INJ = nfl.load_injuries([SEASON]).to_pandas()
    INJ = INJ[(INJ.week==WEEK) & INJ.report_status.isin(["Out","Doubtful","Questionable"])]
    if len(INJ)==0:
        INJ=None
        issue("INFO", f"No injury report published for Week {WEEK} yet",
              "Injuries are the largest single driver of line movement, so the card is missing its most important input.",
              "Reports post Wednesday through Friday. Re-run the card on Friday.", tab="card")
except Exception:
    INJ=None
    issue("INFO", f"Injury data not yet available for {SEASON}",
          "nflverse publishes injuries once the season is underway. Until then game notes show no injury lines.",
          "No action needed. This resolves automatically once Week 1 is played.", tab="card")

def inj_list(t):
    if INJ is None: return []
    x = INJ[INJ.team==t]; out=[]
    for _,v in x[x.report_status=="Out"].head(6).iterrows():
        det = f" ({v.report_primary_injury})" if pd.notna(v.get('report_primary_injury')) else ""
        out.append(f"OUT — {v.position} {v.last_name}{det}")
    for _,v in x[x.report_status=="Questionable"].head(4).iterrows():
        det = f" ({v.report_primary_injury})" if pd.notna(v.get('report_primary_injury')) else ""
        out.append(f"Questionable — {v.position} {v.last_name}{det}")
    return out

_D=A.no_weather; _C={}
def wx(t, roof):
    global _D
    if _D or str(roof) in ("dome","closed") or t not in LL: return None
    if t in _C: return _C[t]
    try:
        import requests
        la,lo = LL[t]
        j = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude":la,"longitude":lo,
            "hourly":"temperature_2m,wind_speed_10m,precipitation_probability","temperature_unit":"fahrenheit",
            "wind_speed_unit":"mph","forecast_days":7}, timeout=6).json()["hourly"]
        i=min(len(j["time"])-1,12)
        d=dict(t=j["temperature_2m"][i],w=j["wind_speed_10m"][i],p=j["precipitation_probability"][i])
    except Exception:
        _D=True
        issue("WARN","Weather forecast unavailable",
              "Wind is the only weather variable that consistently moves totals, so totals may be overstated in windy games.",
              "Usually a network restriction. Works on a normal machine; re-run without --no-weather.", tab="card")
        return None
    _C[t]=d; return d

rows=[]
for _,g in gw.iterrows():
    h,a = g.home_team, g.away_team
    if h not in r.index or a not in r.index:
        issue("ERROR", f"{a} @ {h} could not be priced",
              "This game is missing from the card entirely because one team has no power rating.",
              "Usually a team abbreviation change. Re-run build_ratings.py.", tab="all"); continue
    hfa = 0.0 if str(g.location)=="Neutral" else HFA.get(h,DEF_HFA)
    rd = int(g.home_rest - g.away_rest) if (pd.notna(g.home_rest) and pd.notna(g.away_rest)) else 0
    rest_adj = float(np.clip(rd*0.25,-1.5,1.5))
    w = wx(h,g.roof); wxt=0.0
    if w:
        wxt -= 5.0 if w["w"]>=20 else 3.0 if w["w"]>=15 else 1.5 if w["w"]>=10 else 0.0
        if w["p"] and w["p"]>=40: wxt-=1.0
        if w["t"]<=20: wxt-=1.5
    ms = r.loc[h,"rating"]-r.loc[a,"rating"]+hfa+rest_adj
    mt = 2*AVG_PTS + (r.loc[h,"off"]-r.loc[a,"def"]) + (r.loc[a,"off"]-r.loc[h,"def"]) + wxt
    if str(g.roof)=="dome": mt+=1.5
    sd = SD_BASE + (1.0 if g.total_line>47 else 0.0) - (1.0 if (w and w["w"]>=15) else 0.0)
    ph,pa = amp(g.home_moneyline), amp(g.away_moneyline); nv = ph/(ph+pa)
    ml_imp = norm.ppf(np.clip(nv,.001,.999))*SD_BASE
    cons = ml_imp - g.spread_line
    # ---- Monte Carlo: draw our true line from its own uncertainty, then draw the game ----
    true_line = RNG.normal(ms, RATING_SE, N_SIMS)
    margin = RNG.normal(true_line, sd)                      # home margin
    tot_true = RNG.normal(mt, RATING_SE*0.8, N_SIMS)
    total_sim = RNG.normal(tot_true, 10.4)
    pwin = float((margin > 0).mean() + 0.5*(margin == 0).mean())
    pcov = float((margin > g.spread_line).mean())
    pover = float((total_sim > g.total_line).mean())
    sp_side = h if pcov>.5 else a; ml_side = h if pwin>nv else a
    def combo(edge_pp, winp):
        """Composite score, 0-100. Simulated win probability carries 70 points,
        value over the price carries 30. A minimum value gate stops heavily
        priced favorites grading well on almost no edge.
        edge_pp = simulated probability minus de-vigged market probability, in points
        winp    = simulated probability this specific pick wins, 0-1"""
        w = 100*winp
        W = float(np.clip((w - 40.0)/20.0, 0, 1)) * 70.0     # 40% -> 0, 60%+ -> 70
        E = float(np.clip(edge_pp/8.0, 0, 1)) * 30.0          # 0pp -> 0, 8pp+ -> 30
        score = W + E
        g = "A" if score >= 70 else "B" if score >= 55 else "C" if score >= 40 else "D"
        # value gate: cannot grade above these without a minimum edge
        if edge_pp < 2.0 and g == "A": g = "B"
        if edge_pp < 1.0 and g in ("A","B"): g = "C"
        if edge_pp < 0.5: g = "D"
        return g
    def combo_score(edge_pp, winp):
        w = 100*winp
        return round(float(np.clip((w-40.0)/20.0,0,1))*70.0 + float(np.clip(edge_pp/8.0,0,1))*30.0, 1)
    def tier(e):
        e=abs(e); return "A" if e>=5.5 else "B" if e>=3.5 else "C" if e>=2.0 else "—"
    sp_conf = round(100*max(pcov,1-pcov),1)
    ml_conf = round(100*(pwin if ml_side==h else 1-pwin),1)
    tot_conf = round(100*max(pover,1-pover),1)
    # book-consistency read, expressed as which market is the better price
    better = ""
    if abs(cons) >= 1.0:
        home_cheaper_on_spread = cons > 0
        if sp_side == h: better = "spread" if home_cheaper_on_spread else "moneyline"
        else:            better = "moneyline" if home_cheaper_on_spread else "spread"
    rows.append(dict(away=a,home=h,gid=f"{a}@{h}",date=str(g.gameday),day=str(g.weekday),
        time=str(g.gametime),venue=str(g.stadium),roof=str(g.roof),surface=str(g.surface),
        hfa=hfa,rest=rd,div=bool(g.div_game),wx=w,wxt=round(wxt,1),
        mk_sp=g.spread_line,our_sp=round(ms,1),sp_gap=round(ms-g.spread_line,1),
        sp_side=sp_side,sp_tier=combo(100*(max(pcov,1-pcov)-0.5), max(pcov,1-pcov)),
        sp_edge=round(100*(max(pcov,1-pcov)-0.5),1),
        sp_score=combo_score(100*(max(pcov,1-pcov)-0.5), max(pcov,1-pcov)),
        mk_ml_h=g.home_moneyline,mk_ml_a=g.away_moneyline,nv=round(100*nv,1),
        our_ph=round(100*pwin,1),ml_side=ml_side,
        ml_tier=combo(abs(100*(pwin-nv)), (pwin if ml_side==h else 1-pwin)),
        ml_score=combo_score(abs(100*(pwin-nv)), (pwin if ml_side==h else 1-pwin)),
        ml_edge=round(100*(pwin-nv),1),ml_mkt=round(100*(nv if ml_side==h else 1-nv),1),
        ml_imp=round(ml_imp,1),cons=round(cons,1),
        mk_tot=g.total_line,our_tot=round(mt,1),tot_gap=round(mt-g.total_line,1),
        tot_side=("OVER" if pover>0.5 else "UNDER"),
        tot_tier=combo(100*(max(pover,1-pover)-0.5), max(pover,1-pover)),
        tot_score=combo_score(100*(max(pover,1-pover)-0.5), max(pover,1-pover)),
        tot_edge=round(100*(max(pover,1-pover)-0.5),1), pover=round(100*pover,1),
        elo_sp=(round(ELO[h]-ELO[a]+hfa,1) if (ELO is not None and h in ELO.index and a in ELO.index) else None),
        split=(sp_side!=ml_side), better=better, sp_conf=sp_conf, ml_conf=ml_conf, tot_conf=tot_conf,
        h_off=int(r.loc[h,"off_rank"]), h_def=int(r.loc[h,"def_rank"]), h_pwr=int(r.loc[h,"pwr_rank"]),
        a_off=int(r.loc[a,"off_rank"]), a_def=int(r.loc[a,"def_rank"]), a_pwr=int(r.loc[a,"pwr_rank"]),
        h_sch=str(r.loc[h,"scheme"]), a_sch=str(r.loc[a,"scheme"]),
        h_dsch=str(r.loc[h,"dsch"]), a_dsch=str(r.loc[a,"dsch"]),
        h_po=int(r.loc[h,"po_rank"]), h_ro=int(r.loc[h,"ro_rank"]),
        h_pd=int(r.loc[h,"pd_rank"]), h_rd=int(r.loc[h,"rd_rank"]),
        a_po=int(r.loc[a,"po_rank"]), a_ro=int(r.loc[a,"ro_rank"]),
        a_pd=int(r.loc[a,"pd_rank"]), a_rd=int(r.loc[a,"rd_rank"]),
        h_gv=(float(r.loc[h,"Giveaways"]) if pd.notna(r.loc[h,"Giveaways"]) else None),
        a_gv=(float(r.loc[a,"Giveaways"]) if pd.notna(r.loc[a,"Giveaways"]) else None),
        h_tk=(float(r.loc[h,"Takeaways"]) if pd.notna(r.loc[h,"Takeaways"]) else None),
        a_tk=(float(r.loc[a,"Takeaways"]) if pd.notna(r.loc[a,"Takeaways"]) else None),
        h_pr=(float(r.loc[h,"PassRate"]) if pd.notna(r.loc[h,"PassRate"]) else None),
        a_pr=(float(r.loc[a,"PassRate"]) if pd.notna(r.loc[a,"PassRate"]) else None),
        inj_h=inj_list(h), inj_a=inj_list(a)))
D=pd.DataFrame(rows)
if len(D)==0: sys.exit("No games priced.")
D["elo_gap"]=D.apply(lambda x:(round(x.our_sp-x.elo_sp,1) if x.elo_sp is not None else np.nan),axis=1)
_eg = D.elo_gap.dropna()
ELO_SD = float(_eg.std()) if len(_eg)>3 else 3.5
ELO_THR = round(1.5*ELO_SD, 1)
if A.game:
    k=A.game.replace(" ","").upper(); D=D[D.gid.str.upper()==k]
    if len(D)==0: sys.exit(f"'{A.game}' not in week {WEEK}.")

if not NOTES:
    issue("INFO", "No researched news attached",
          "Trades, coaching changes, suspensions, line movement and expert picks are not in any dataset, so game notes show only injuries, weather and rest.",
          "Ask Claude to research the slate and produce notes.json, then re-run with --notes notes.json", tab="card")

led=[]
for _,x in D.iterrows():
    led += [dict(week=WEEK,date=x.date,day=x.day,game=x.gid,market="Spread",
                 pick=f"{x.sp_side} {(-x.mk_sp if x.sp_side==x.home else x.mk_sp):+.1f}",
                 our_num=x.our_sp,market_num=x.mk_sp,gap=x.sp_gap,tier=x.sp_tier,result=""),
            dict(week=WEEK,date=x.date,day=x.day,game=x.gid,market="Moneyline",pick=x.ml_side,
                 our_num=x.ml_conf,market_num=x.ml_mkt,gap=round(x.ml_conf-x.ml_mkt,1),win_pct=x.ml_conf,tier=x.ml_tier,result=""),
            dict(week=WEEK,date=x.date,day=x.day,game=x.gid,market="Total",
                 pick=f"{x.tot_side} {x.mk_tot}",our_num=x.our_tot,market_num=x.mk_tot,
                 gap=x.tot_gap,win_pct=x.tot_conf,tier=x.tot_tier,result="")]
L=pd.DataFrame(led); L["result"]=L["result"].astype("object")

# auto-grade any game that already has a final score
_fin = {f"{g.away_team}@{g.home_team}": g for _,g in
        sch[(sch.week==WEEK) & sch.result.notna()].iterrows()}
def _autograde(row):
    g = _fin.get(row.game)
    if g is None: return row.result
    hm, aw = g.home_score, g.away_score
    marg, tot = hm-aw, hm+aw
    if row.market == "Spread":
        side = row.pick.split()[0]
        if marg == g.spread_line: return "P"
        covered = (marg > g.spread_line)
        return "W" if (covered == (side == g.home_team)) else "L"
    if row.market == "Moneyline":
        if marg == 0: return "P"
        return "W" if ((marg > 0) == (row.pick == g.home_team)) else "L"
    if row.market == "Total":
        if tot == g.total_line: return "P"
        return "W" if ((tot > g.total_line) == row.pick.startswith("OVER")) else "L"
    return row.result
L["result"] = L.apply(_autograde, axis=1)
_graded = int((L.result.isin(["W","L","P"])).sum())
lp=P(f"ledger_week{WEEK}.csv"); pres=0
if os.path.exists(lp):
    try:
        old=pd.read_csv(lp,dtype={"result":str},keep_default_na=False)
        keep=old[old.result.str.upper().str.strip().isin(["W","L","P"])]
        if len(keep):
            L=L.merge(keep[["game","market","result"]],on=["game","market"],how="left",suffixes=("","_o"))
            L["result"]=L["result_o"].fillna(L["result"]); L=L.drop(columns=["result_o"]); pres=len(keep)
    except Exception as e:
        issue("ERROR","Existing ledger could not be read",
              "Results you already filled in were NOT carried over and may be overwritten.",
              f"Back up ledger_week{WEEK}.csv before re-running. ({e})", tab="ledger")
L.to_csv(lp,index=False)

def tc(t): return {"A":"t-a","B":"t-b","C":"t-c","D":"t-d"}.get(t,"t-n")
def ln(team,home,sp): return f"{team} {(-sp if team==home else sp):+.1f}"
days = list(dict.fromkeys(D.day.tolist()))

import glob as _g, re as _re2
_weeks = set()
for _p in _g.glob(P("Week*-Card.html")) + _g.glob(os.path.join(BASE, "..", "site", "Week*-Card.html")):
    _m = _re2.search(r"Week(\\d+)-Card\\.html$", os.path.basename(_p))
    if _m: _weeks.add(int(_m.group(1)))
_weeks.add(WEEK)
WEEKS = sorted(_weeks)
WEEKOPTS = "".join(
    f'<option value="Week{w}-Card.html"{" selected" if w==WEEK else ""}>Week {w}'
    f'{" (current)" if w==WEEK else ""}</option>' for w in WEEKS)
H=[]
H.append(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NFL Week {WEEK}</title><style>
:root{{--ink:#12161c;--mute:#5d6875;--line:#dce1e7;--bg:#fbfcfd;--pan:#ffffff;--hd:#f4f6f8;--hov:#f8fafb;
--a:#0d7a4f;--abg:#e8f5ee;--b:#1a5f9e;--bbg:#e7f0f9;--c:#7a6a1a;--cbg:#f7f2df;
--n:#8b95a1;--nbg:#f1f3f5;--fl:#a8541a;--flbg:#fdf0e4;--sp:#553a7a;--spbg:#efe7f7;--sep:#aab4c0}}
body.dark{{--ink:#e6eaf0;--mute:#9aa5b3;--line:#2b333d;--bg:#12161c;--pan:#1a1f27;--hd:#222833;--hov:#20262f;
--a:#4ade80;--abg:#12321f;--b:#7cb8f0;--bbg:#0f2740;--c:#e0c766;--cbg:#332c12;
--n:#8b95a1;--nbg:#252b34;--fl:#e0913f;--flbg:#3a2712;--sp:#c4a8ee;--spbg:#2c2140;--sep:#5a6675}}
*{{box-sizing:border-box}}body{{margin:0;padding:26px 18px 70px;background:var(--bg);color:var(--ink);transition:background .15s;
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:1280px;margin:0 auto}}
.top{{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:6px}}
h1{{font-size:25px;margin:0;letter-spacing:-.01em}}
select,button{{font:inherit;font-size:13px;padding:6px 10px;border:1px solid var(--line);
border-radius:6px;background:var(--pan);color:var(--ink);cursor:pointer}}
a.btn{{font-size:13px;padding:6px 11px;border:1px solid var(--line);border-radius:6px;
background:var(--pan);color:var(--ink);text-decoration:none;white-space:nowrap}}
a.btn:hover{{border-color:var(--mute)}}
.sub{{color:var(--mute);font-size:13px;margin-bottom:18px}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--mute);
margin:30px 0 11px;padding-bottom:6px;border-bottom:1px solid var(--line);font-weight:600}}
.iss{{background:var(--pan);border:1px solid var(--line);border-left:4px solid var(--fl);
border-radius:6px;padding:12px 14px;margin-bottom:10px}}
.iss .h{{font-weight:600;font-size:13.5px;margin-bottom:3px}}
.iss .w{{font-size:13px;color:var(--mute);margin-bottom:5px}}
.iss .f{{font-size:13px;background:var(--flbg);color:var(--fl);padding:5px 9px;border-radius:4px;display:inline-block}}
.filters{{background:var(--pan);border:1px solid var(--line);border-radius:8px;padding:11px 13px;margin-bottom:11px;
display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start}}
.fg{{font-size:12.5px}}.fg b{{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
color:var(--mute);margin-bottom:5px}}
label{{display:inline-flex;align-items:center;gap:4px;margin:0 9px 3px 0;cursor:pointer}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--pan)}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}
th{{background:var(--hd);text-align:left;padding:8px 9px;font-weight:600;font-size:10.5px;
text-transform:uppercase;letter-spacing:.05em;color:var(--mute);border-bottom:1px solid var(--line);white-space:nowrap}}
td{{padding:8px 9px;border-bottom:1px solid var(--line);white-space:nowrap}}
td.rd{{white-space:normal;font-size:12px;min-width:230px;line-height:1.4}}
tbody tr:hover{{background:var(--hov)}}.game{{font-weight:600}}
.num{{color:var(--mute);font-variant-numeric:tabular-nums}}.pick{{font-weight:600}}
.grp{{border-left:1px solid var(--line)}}
.tag{{display:inline-block;padding:2px 7px;border-radius:10px;font-size:10.5px;font-weight:600}}
.t-a{{background:var(--abg);color:var(--a)}}.t-b{{background:var(--bbg);color:var(--b)}}
.t-c{{background:var(--cbg);color:var(--c)}}.t-d{{background:var(--nbg);color:var(--n)}}
.pk-a{{color:var(--a);font-weight:700}}.pk-b{{color:var(--b);font-weight:700}}
.pk-c{{color:var(--c);font-weight:700}}.pk-d{{color:var(--n);font-weight:700}}.t-n{{background:var(--nbg);color:var(--n)}}
.sp{{background:var(--spbg);color:var(--sp);padding:2px 6px;border-radius:9px;font-size:10.5px;font-weight:600;margin-right:3px}}
.bad{{color:var(--fl);font-weight:600}}
.card{{background:var(--pan);border:1px solid var(--line);border-radius:8px;padding:13px 15px;margin-bottom:10px}}
.card h3{{margin:0 0 2px;font-size:15px}}
.card .meta{{font-size:12px;color:var(--mute);margin-bottom:8px;font-variant-numeric:tabular-nums}}
.nl{{margin:5px 0 0;padding-left:17px;font-size:13.5px}}.nl li{{margin:2px 0}}
.lbl{{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);font-weight:600;margin-top:10px}}
.none{{color:var(--mute);font-size:13px;font-style:italic}}
details{{background:var(--pan);border:1px solid var(--line);border-radius:8px;padding:12px 15px;margin-top:14px}}
details .scroll{{margin-top:9px}}
summary{{cursor:pointer;font-weight:600;font-size:14px}}
details h4{{font-size:15px;margin:28px 0 9px;padding:6px 0 6px 11px;
border-left:4px solid var(--hc);border-bottom:1px solid var(--line);color:var(--hc)}}
details h4:nth-of-type(1){{--hc:#1a5f9e}} details h4:nth-of-type(2){{--hc:#0d7a4f}}
details h4:nth-of-type(3){{--hc:#8a5a1a}} details h4:nth-of-type(4){{--hc:#7a3a8a}}
details h4:nth-of-type(5){{--hc:#0e6f79}} details h4:nth-of-type(6){{--hc:#a03a52}}
body.dark details h4:nth-of-type(1){{--hc:#7cb8f0}} body.dark details h4:nth-of-type(2){{--hc:#4ade80}}
body.dark details h4:nth-of-type(3){{--hc:#e0a44f}} body.dark details h4:nth-of-type(4){{--hc:#c9a0e8}}
body.dark details h4:nth-of-type(5){{--hc:#5fd0d8}} body.dark details h4:nth-of-type(6){{--hc:#f08fa4}}
details .sh{{border-left:3px solid var(--line);padding-left:8px}}
details .sh{{font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--mute);font-weight:600;margin:16px 0 5px}}
details .wt{{font-weight:400;font-size:11px;color:var(--mute);text-transform:uppercase;letter-spacing:.05em}}
details table{{margin-bottom:6px}}
details table{{font-size:13px;margin-top:4px}}details td,details th{{white-space:normal;vertical-align:top}}
details p{{font-size:13.5px}}
.nav{{position:sticky;top:0;z-index:20;background:var(--bg);padding:8px 0 10px;margin-bottom:6px;
border-bottom:1px solid var(--line);display:flex;gap:8px;flex-wrap:wrap}}
.secnav{{display:flex;gap:7px;flex-wrap:wrap;margin:-4px 0 12px}}
.secnav a{{font-size:11.5px;padding:3px 9px;border:1px solid var(--line);border-radius:20px;
color:var(--mute);text-decoration:none;background:var(--pan)}}
.secnav a:hover{{color:var(--ink)}}
.nav a{{font-size:12px;padding:4px 10px;border:1px solid var(--line);border-radius:20px;
color:var(--mute);text-decoration:none;background:var(--pan)}}
.nav a:hover{{color:var(--ink);border-color:var(--mute)}}
h2{{scroll-margin-top:60px}}
.hide{{display:none}}
""" + "".join(f".tm-{k}{{color:{v}}}" for k,v in TEAMCOLOR_LIGHT.items())
  + "".join(f"body.dark .tm-{k}{{color:{v}}}" for k,v in TEAMCOLOR_DARK.items()) + f"""
.gc{{background:var(--pan);border:1px solid var(--line);border-radius:10px;margin-bottom:10px}}
.gc>summary{{cursor:pointer;padding:14px 16px;list-style:none;
display:grid;grid-template-columns:22px 1fr auto;gap:12px;align-items:start}}
.gc>summary::-webkit-details-marker{{display:none}}
summary:focus{{outline:none}}
summary:focus-visible{{outline:2px solid var(--b);outline-offset:-2px;border-radius:10px}}
.gc>summary:hover{{background:var(--hov);border-radius:10px}}
.gc[open]>summary{{border-bottom:1px solid var(--line);border-radius:10px 10px 0 0}}
.gc .car{{color:var(--mute);font-size:15px;line-height:1.25;transition:transform .15s;
display:inline-block;text-align:center}}
.gc[open] .car{{transform:rotate(90deg)}}
.gcH{{font-size:15.5px;font-weight:600;line-height:1.3;letter-spacing:-.01em}}
.gcH .vs{{color:var(--mute);font-weight:400;font-size:13px;margin:0 6px}}
.gcM{{font-size:11.5px;color:var(--mute);margin-top:3px;font-variant-numeric:tabular-nums}}
.gcP{{font-size:12.5px;color:var(--mute);margin-top:5px;line-height:1.5}}
.gcP b{{font-weight:600}}
.pvRow{{display:flex;flex-wrap:wrap;align-items:baseline;margin-top:4px}}
.pvRow>span{{padding:2px 12px;border-left:2px solid var(--sep);line-height:1.35}}
.pvRow>span:first-child{{padding-left:0;border-left:none}}
.pvRow{{gap:0}}
.pvT{{font-weight:700;font-size:13px;min-width:104px}}
.pvS{{font-variant-numeric:tabular-nums;min-width:42px}}
.pvU b{{font-size:10px;letter-spacing:.06em;color:var(--mute);font-weight:700}}
.pvU i{{font-style:normal;color:var(--mute)}}
@media(max-width:640px){{.pvRow>span{{padding:2px 8px}}.pvT{{min-width:84px}}.pvS{{min-width:34px}}}}
.gcF{{display:flex;flex-direction:column;gap:4px;align-items:flex-end;padding-top:2px}}
.gcX{{font-size:10.5px;font-weight:600;color:var(--fl);background:var(--flbg);
border-radius:20px;padding:2px 9px;white-space:nowrap}}
.gc .body{{padding:14px 16px 16px}}
.gc .body .lbl:first-child{{margin-top:0}}
.cmp{{width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:4px}}
.cmp th{{background:transparent;border-bottom:1px solid var(--line);padding:5px 8px;
font-size:12px;text-transform:none;letter-spacing:0;font-weight:600}}
.cmp th.ca,.cmp th.ch{{text-align:center;width:29%}}
.cmp td{{padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}}
.cmp td.cl{{color:var(--mute)}}
.cmp td.ca,.cmp td.ch{{text-align:center;font-variant-numeric:tabular-nums}}
.cmp tr.grp2 td{{font-weight:700;background:var(--hd);font-size:10.5px;letter-spacing:.08em;color:var(--mute)}}
.cmp td.bet{{background:rgba(13,122,79,.13);font-weight:600}}
body.dark .cmp td.bet{{background:rgba(74,222,128,.15)}}
.mkc{{margin-bottom:14px;border:none;background:transparent;padding:0}}
summary.mk{{cursor:pointer;font-size:14px;font-weight:600;margin:18px 0 8px;padding-left:9px;
border-left:4px solid var(--b);color:var(--ink);list-style:none}}
summary.mk::-webkit-details-marker{{display:none}}
.mkn{{font-weight:400;font-size:11.5px;color:var(--mute);margin-left:7px}}
h3.sd{{font-size:14px;margin:6px 0 9px;padding-left:9px;border-left:4px solid var(--a);color:var(--ink)}}
tr.tot td{{font-weight:700;background:var(--hd)}}
.fpanel{{margin-bottom:10px}}.fpanel>summary{{cursor:pointer;font-size:13px;font-weight:600;padding:2px 0}}
.fpanel .filters{{margin-top:9px}}
td.rw{{color:var(--a);font-weight:700}}td.rl{{color:#b3313c;font-weight:700}}
body.dark td.rl{{color:#f08fa4}}td.rp{{color:var(--mute);font-weight:700}}
@media(max-width:640px){{.gc>summary{{grid-template-columns:20px 1fr}}.gcF{{display:none}}}}
@media print{{body{{background:#fff;padding:0}}.filters,select{{display:none}}}}
</style></head><body><div class="wrap">
<div class="top"><h1>NFL Week {WEEK}</h1>
<select id="view" onchange="setView()">
  <option value="card">Weekly Card</option>
  <option value="cheat">Team Cheat Sheet</option>
  <option value="ledger">Ledger &amp; Results</option>
</select>
<select id="wk" onchange="goWeek()" title="View another week">{WEEKOPTS}</select>
<a class="btn" href="https://github.com/{REPO}/actions/workflows/{WORKFLOW}"
   target="_blank" rel="noopener" title="Opens the run screen on GitHub">Update now &rsaquo;</a>
<button id="thm" onclick="tog()" style="margin-left:auto">Dark mode</button></div>
<div class="sub">{SEASON} season · {len(D)} games · generated {dt.date.today()}</div>
<div class="nav" id="nav"></div>""")

if ISSUES:
    H.append('<h2 id="s-issues" class="hide">Needs attention</h2>')
    for i in ISSUES:
        H.append(f'<div class="iss" data-tab="{i.get("tab","card")}"><div class="h">{i["lvl"]} — {i["what"]}</div>'
                 f'<div class="w"><b>Why it matters:</b> {i["why"]}</div>'
                 f'<div class="f"><b>Fix:</b> {i["fix"]}</div></div>')

H.append('<div id="v-card">')
H.append('<h2 id="s-summary">Summary — all markets</h2>')
H.append('<div class="secnav" data-own="s-summary"></div>')
H.append('<div class="filters"><div class="fg"><b>Day</b>')
for d in days: H.append(f'<label><input type="checkbox" class="fd" value="{d}" checked onchange="flt()">{d}</label>')
H.append('</div><div class="fg"><b>Game</b>')
for _,x in D.iterrows(): H.append(f'<label><input type="checkbox" class="fgm" value="{x.gid}" checked onchange="flt()">{x.away} @ {x.home}</label>')
H.append('</div><div class="fg"><b>&nbsp;</b><button onclick="allOn()">Select all</button> '
         '<button onclick="allOff()">Clear</button>'
         '<div id="fcount" class="none" style="margin-top:6px"></div></div></div>')
H.append('<div class="scroll"><table><thead><tr><th>Date</th><th>Day / Time</th><th>Game</th>'
         '<th class="grp">Spread</th><th>Grade</th>'
         '<th class="grp">Moneyline</th><th>Grade</th>'
         '<th class="grp">Total</th><th>Pick</th><th>Grade</th>'
         '<th class="grp">What this tells you</th></tr></thead><tbody>')
for _,x in D.iterrows():
    reads = []
    if isinstance(x.better,str) and x.better:
        other = "moneyline" if x.better=="spread" else "spread"
        fav = x.home if x.mk_sp>0 else x.away
        reads.append(f'The moneyline prices <b>{fav}</b> as a {abs(x.ml_imp):.1f}-point favorite, but the spread '
                     f'is only {abs(x.mk_sp):.1f}. Backing <b>{x.sp_side}</b>, the <b>{x.better}</b> is worth '
                     f'{abs(x.cons):.1f} points more than the {other}.')
    if x.split:
        reads.append(f'Markets point different ways. Best cover is <b>{x.sp_side}</b> ({x.sp_conf}% to cover). '
                     f'Best value to win outright is <b>{x.ml_side}</b> — we make them {x.ml_conf}%, '
                     f'the price implies {x.ml_mkt}%.')
    if pd.notna(x.elo_gap) and abs(x.elo_gap)>=ELO_THR:
        lean = x.home if x.elo_sp > x.our_sp else x.away
        reads.append(f'Our two models disagree by {abs(x.elo_gap):.1f} points on the spread: the EPA model has '
                     f'{x.our_sp:+.1f}, the Elo model {x.elo_sp:+.1f}. Elo is higher on <b>{lean}</b>. '
                     f'Low-information game.')
    def side_cell(fav_txt, dog_txt, fav_is_pick, grade):
        pc = {"A":"pk-a","B":"pk-b","C":"pk-c","D":"pk-d"}.get(grade,"pk-d")
        f = f'<span class="{pc}">{fav_txt}</span>' if fav_is_pick else f'<span class="num">{fav_txt}</span>'
        d = f'<span class="{pc}">{dog_txt}</span>' if not fav_is_pick else f'<span class="num">{dog_txt}</span>'
        return f + ' <span class="num">/</span> ' + d
    sp_home_txt = f"{x.home} {-x.mk_sp:+.1f}"; sp_away_txt = f"{x.away} {x.mk_sp:+.1f}"
    ml_home_txt = f"{x.home} {x.mk_ml_h:+.0f}"; ml_away_txt = f"{x.away} {x.mk_ml_a:+.0f}"
    tot_pc = {"A":"pk-a","B":"pk-b","C":"pk-c","D":"pk-d"}.get(x.tot_tier,"pk-d")
    H.append(f'<tr class="row" data-day="{x.day}" data-gid="{x.gid}">'
      f'<td class="num">{x.date[5:].replace("-","/")}</td>'
      f'<td class="num">{x.day[:3]} {x.time[:5]}</td>'
      f'<td class="game"><span class="tm-{x.away}">{x.away}</span> <span class="num">@</span> <span class="tm-{x.home}">{x.home}</span></td>'
      f'<td class="grp">{side_cell(sp_home_txt, sp_away_txt, x.sp_side==x.home, x.sp_tier)}</td>'
      f'<td><span class="tag {tc(x.sp_tier)}">{x.sp_tier}</span></td>'
      f'<td class="grp">{side_cell(ml_home_txt, ml_away_txt, x.ml_side==x.home, x.ml_tier)}</td>'
      f'<td><span class="tag {tc(x.ml_tier)}">{x.ml_tier}</span></td>'
      f'<td class="num grp">{x.mk_tot}</td>'
      f'<td><span class="{tot_pc}">{x.tot_side}</span></td>'
      f'<td><span class="tag {tc(x.tot_tier)}">{x.tot_tier}</span></td>'
      f'<td class="grp rd">{"<br>".join(reads) if reads else "<span class=\'none\'>nothing unusual</span>"}</td></tr>')
H.append('</tbody></table></div>')

H.append('<h2 id="s-notes">Game notes</h2>')
H.append('<div style="margin:-2px 0 10px"><button onclick="expAll(1)">Expand all</button> '
         '<button onclick="expAll(0)">Collapse all</button></div>')
H.append('<div class="secnav" data-own="s-notes"></div>')
for _,x in D.sort_values(["date","time"]).iterrows():
    n = NOTES.get(x.gid, {})
    wxs = (f' · {x.wx["t"]:.0f}°F, {x.wx["w"]:.0f} mph wind' + (" — wind flag" if x.wx["w"]>=15 else "")) if x.wx else ""
    dome = " · dome" if x.roof=="dome" else ""
    divg = " · division game" if x.div else ""
    rest = f' · rest {x.rest:+d} days' if x.rest else ''
    _flags = []
    if isinstance(x.better,str) and x.better: _flags.append("price gap")
    if x.split: _flags.append("markets split")
    if pd.notna(x.elo_gap) and abs(x.elo_gap)>=ELO_THR: _flags.append("models disagree")
    _fl = '<div class="gcF">' + "".join(f'<span class="gcX">{f}</span>' for f in _flags) + '</div>' if _flags else '<div class="gcF"></div>'
    _meta = f'{x.day[:3]} {x.date[5:].replace("-","/")} · {x.time[:5]} · {x.venue}{dome} · {x.surface}{divg}{wxs}{rest}'
    def _pline(t, name, rec, orank, osch, drank, dsch):
        return (f'<span class="pvT tm-{t}">{name}</span>'
                f'<span class="pvS">{rec}</span>'
                f'<span class="pvU"><b>OFF</b> #{orank} <i>{osch}</i></span>'
                f'<span class="pvU"><b>DEF</b> #{drank} <i>{dsch}</i></span>')
    _prev = ('<div class="pvRow">' + _pline(x.away, TEAMNAME.get(x.away,x.away), REC.get(x.away,"0-0"),
                                            x.a_off, x.a_sch, x.a_def, x.a_dsch) + '</div>'
             '<div class="pvRow">' + _pline(x.home, TEAMNAME.get(x.home,x.home), REC.get(x.home,"0-0"),
                                            x.h_off, x.h_sch, x.h_def, x.h_dsch) + '</div>')
    H.append(f'<details class="gc row" data-day="{x.day}" data-gid="{x.gid}"><summary>'
      f'<span class="car">&rsaquo;</span>'
      f'<span><span class="gcH"><span class="tm-{x.away}">{x.away}</span>'
      f'<span class="vs">at</span><span class="tm-{x.home}">{x.home}</span></span>'
      f'<div class="gcM">{_meta}</div><div class="gcP">{_prev}</div></span>'
      f'{_fl}</summary><div class="body">')
    def _cmp_row(label, a_val, h_val, a_num=None, h_num=None, lower_better=False):
        ac = hc = ""
        if a_num is not None and h_num is not None and a_num != h_num:
            a_better = (a_num < h_num) if lower_better else (a_num > h_num)
            ac, hc = (" bet", "") if a_better else ("", " bet")
        return (f'<tr><td class="cl">{label}</td><td class="ca{ac}">{a_val}</td>'
                f'<td class="ch{hc}">{h_val}</td></tr>')
    def _rk(n): return f'#{n}'
    _to_a = f'{x.a_tk:.1f} / {x.a_gv:.1f}' if x.a_tk is not None else '—'
    _to_h = f'{x.h_tk:.1f} / {x.h_gv:.1f}' if x.h_tk is not None else '—'
    _pr_a = f'{x.a_pr*100:.0f}%' if x.a_pr else '—'
    _pr_h = f'{x.h_pr*100:.0f}%' if x.h_pr else '—'
    blocks=[f'<table class="cmp"><thead><tr><th></th>'
            f'<th class="ca"><span class="tm-{x.away}">{TEAMNAME.get(x.away,x.away)}</span></th>'
            f'<th class="ch"><span class="tm-{x.home}">{TEAMNAME.get(x.home,x.home)}</span></th></tr></thead><tbody>'
            + _cmp_row("Record", REC.get(x.away,"0-0"), REC.get(x.home,"0-0"))
            + _cmp_row("Division", DIV.get(x.away,"—"), DIV.get(x.home,"—"))
            + _cmp_row("Power rank", _rk(x.a_pwr), _rk(x.h_pwr), x.a_pwr, x.h_pwr, True)
            + '<tr class="grp2"><td class="cl">OFFENSE</td><td class="ca">&nbsp;</td><td class="ch">&nbsp;</td></tr>'
            + _cmp_row("Overall", _rk(x.a_off), _rk(x.h_off), x.a_off, x.h_off, True)
            + _cmp_row("Pass", _rk(x.a_po), _rk(x.h_po), x.a_po, x.h_po, True)
            + _cmp_row("Rush", _rk(x.a_ro), _rk(x.h_ro), x.a_ro, x.h_ro, True)
            + _cmp_row("Scheme", x.a_sch, x.h_sch)
            + _cmp_row("Early-down pass rate", _pr_a, _pr_h)
            + _cmp_row("Giveaways / gm", (f"{x.a_gv:.1f}" if x.a_gv is not None else "—"),
                       (f"{x.h_gv:.1f}" if x.h_gv is not None else "—"), x.a_gv, x.h_gv, True)
            + '<tr class="grp2"><td class="cl">DEFENSE</td><td class="ca">&nbsp;</td><td class="ch">&nbsp;</td></tr>'
            + _cmp_row("Overall", _rk(x.a_def), _rk(x.h_def), x.a_def, x.h_def, True)
            + _cmp_row("Pass", _rk(x.a_pd), _rk(x.h_pd), x.a_pd, x.h_pd, True)
            + _cmp_row("Rush", _rk(x.a_rd), _rk(x.h_rd), x.a_rd, x.h_rd, True)
            + _cmp_row("Identity", x.a_dsch, x.h_dsch)
            + _cmp_row("Takeaways / gm", (f"{x.a_tk:.1f}" if x.a_tk is not None else "—"),
                       (f"{x.h_tk:.1f}" if x.h_tk is not None else "—"), x.a_tk, x.h_tk, False)
            + '</tbody></table>']
    def sec(title, items):
        if items: blocks.append(f'<div class="lbl">{title}</div><ul class="nl">' +
                                "".join(f"<li>{i}</li>" for i in items) + "</ul>")
    sec("Injury report", [f"{x.home} — {i}" for i in x.inj_h] + [f"{x.away} — {i}" for i in x.inj_a])
    sec("Returning from injury", n.get("returning"))
    sec("Trades &amp; roster moves", n.get("trades"))
    sec("Coaching &amp; suspensions", (n.get("coaching") or []) + (n.get("suspensions") or []))
    sec("Line movement", n.get("movement"))
    sec("Notable", n.get("birthdays"))
    ex = n.get("experts")
    if ex:
        blocks.append('<div class="lbl">Expert picks</div><ul class="nl">' + "".join(
            f"<li><b>{e.get('name','')}</b> ({e.get('record','record n/a')}) — {e.get('pick','')}</li>"
            for e in ex) + "</ul>")
    if pd.notna(x.elo_gap) and abs(x.elo_gap)>=ELO_THR:
        _l = x.home if x.elo_sp > x.our_sp else x.away
        blocks.append(f'<div class="lbl">Models disagree</div><ul class="nl"><li>The <b>EPA model</b> (play-by-play '
            f'efficiency, the main one) makes this {x.our_sp:+.1f}. The <b>Elo model</b> (game results only, no '
            f'play-by-play) makes it {x.elo_sp:+.1f} — {abs(x.elo_gap):.1f} points apart, with Elo higher on '
            f'{TEAMNAME.get(_l,_l)}. This week the two models differ by {ELO_SD:.1f} points on average, so anything '
            f'past {ELO_THR:.1f} is an outlier. One of the two is badly wrong here and there is no way to know '
            f'which.</li></ul>')
    if isinstance(x.better,str) and x.better:
        other = "moneyline" if x.better=="spread" else "spread"
        fav = x.home if x.mk_sp>0 else x.away
        blocks.append(f'<div class="lbl">Which ticket to buy</div><ul class="nl"><li>The posted spread makes '
            f'{fav} a {abs(x.mk_sp):.1f}-point favorite. Converting the moneyline into a spread gives '
            f'{abs(x.ml_imp):.1f} points instead — a {abs(x.cons):.1f} point disagreement between the book\'s own '
            f'two prices, so one of them is stale. <b>If you back {x.sp_side}, the {x.better} is worth '
            f'{abs(x.cons):.1f} points more</b> than the {other}. This does not change which side to take, '
            f'and does not affect the total.</li></ul>')
    H.append("".join(blocks))
    H.append('</div></details>')
H.append('</div>')


H.append('<div id="v-cheat" class="hide"><h2 id="s-cheat">Team cheat sheet</h2>')
H.append('<div class="secnav" data-own="s-cheat"></div>')
_cs = r.copy(); _cs["team"] = _cs.index
_cs["dvn"] = _cs.team.map(DIV); _cs = _cs[_cs.dvn.notna()]
_cs["conf"] = _cs.dvn.str[:3]
for cf in ["AFC","NFC"]:
    blk = _cs[_cs.conf==cf].copy()
    blk["crank"] = blk.rating.rank(ascending=False).astype(int)
    blk = blk.sort_values("crank")
    H.append(f'<details open><summary>{cf} — ranked 1 to {len(blk)}</summary>'
             '<div class="scroll"><table><thead><tr><th>#</th><th>Team</th><th>Div</th><th>Rec</th>'
             '<th>Off</th><th>Def</th><th>Off scheme</th><th>Pass rate</th>'
             '<th>Def identity</th><th>Take/gm</th><th>Give/gm</th><th>TO diff</th></tr></thead><tbody>')
    for _,t in blk.iterrows():
        prv = f"{t.PassRate*100:.0f}%" if pd.notna(t.PassRate) else "—"
        _tk = f'{t.Takeaways:.1f}' if pd.notna(t.Takeaways) else '—'
        _gv = f'{t.Giveaways:.1f}' if pd.notna(t.Giveaways) else '—'
        _td = f'{t.to_diff:+.1f}' if pd.notna(t.to_diff) else '—'
        H.append(f'<tr><td class="num">{int(t.crank)}</td>'
                 f'<td class="game"><span class="tm-{t.team}">{TEAMNAME.get(t.team,t.team)}</span></td>'
                 f'<td class="num">{t.dvn.split()[1]}</td>'
                 f'<td class="num">{REC.get(t.team,"0-0")}</td>'
                 f'<td class="num">#{int(t.off_rank)}</td><td class="num">#{int(t.def_rank)}</td>'
                 f'<td>{t.scheme}</td><td class="num">{prv}</td>'
                 f'<td>{t.dsch}</td><td class="num">{_tk}</td><td class="num">{_gv}</td>'
                 f'<td class="num">{_td}</td></tr>')
    H.append('</tbody></table></div></details>')
H.append('<h2 id="s-div">Rankings by division</h2>')
H.append('<div class="secnav" data-own="s-div"></div>')
for cf in ["AFC","NFC"]:
    for dv in [f"{cf} East", f"{cf} North", f"{cf} South", f"{cf} West"]:
        blk = _cs[_cs.dvn==dv].copy().sort_values("rating", ascending=False)
        H.append(f'<details><summary>{dv}</summary><div class="scroll"><table><thead><tr>'
                 '<th>#</th><th>Team</th><th>Rec</th><th>Lg</th><th>Off</th>'
                 '<th>Def</th><th>Off scheme</th><th>Pass rate</th>'
                 '<th>Def identity</th><th>TO diff</th></tr></thead><tbody>')
        for i,(_,t) in enumerate(blk.iterrows(),1):
            prv = f"{t.PassRate*100:.0f}%" if pd.notna(t.PassRate) else "—"
            _td = f'{t.to_diff:+.1f}' if pd.notna(t.to_diff) else '—'
            H.append(f'<tr><td class="num">{i}</td>'
                     f'<td class="game"><span class="tm-{t.team}">{TEAMNAME.get(t.team,t.team)}</span></td>'
                     f'<td class="num">{REC.get(t.team,"0-0")}</td><td class="num">#{int(t.pwr_rank)}</td>'
                     f'<td class="num">#{int(t.off_rank)}</td><td class="num">#{int(t.def_rank)}</td>'
                     f'<td>{t.scheme}</td><td class="num">{prv}</td>'
                     f'<td>{t.dsch}</td><td class="num">{_td}</td></tr>')
        H.append('</tbody></table></div></details>')
H.append('</div>')

H.append(f'<div id="v-ledger" class="hide"><h2 id="s-ledger">Ledger — Week {WEEK}</h2>')
H.append('<div class="secnav" data-own="s-ledger"></div>')
_ldays = list(dict.fromkeys(L.day.tolist()))
H.append('<details class="fpanel"><summary>Filters</summary><div class="filters">')
H.append('<div class="fg"><b>Day</b>')
for d in _ldays: H.append(f'<label><input type="checkbox" class="ld" value="{d}" checked onchange="lflt()">{d[:3]}</label>')
H.append('</div><div class="fg"><b>Grade</b>')
for gd in ["A","B","C","D"]: H.append(f'<label><input type="checkbox" class="lt" value="{gd}" checked onchange="lflt()">{gd}</label>')
H.append('</div><div class="fg"><b>Team</b>')
_lteams = sorted({t for gid in L.game.unique() for t in gid.split("@")})
for tmv in _lteams: H.append(f'<label><input type="checkbox" class="ltm" value="{tmv}" checked onchange="lflt()">{tmv}</label>')
H.append('</div><div class="fg"><b>Result</b>')
for rs in ["W","L","P","pending"]: H.append(f'<label><input type="checkbox" class="lr" value="{rs}" checked onchange="lflt()">{rs}</label>')
H.append('</div><div class="fg"><b>&nbsp;</b><button onclick="lAll(1)">Select all</button> '
         '<button onclick="lAll(0)">Clear</button><div id="lcount" class="none" style="margin-top:6px"></div>'
         '</div></div></details>')
import glob as _glob
_all = []
for _f in sorted(_glob.glob(P("ledger_week*.csv"))):
    try:
        _d = pd.read_csv(_f, dtype={"result":str}, keep_default_na=False)
        if {"market","tier","result"}.issubset(_d.columns): _all.append(_d)
    except Exception: pass
if _all:
    S = pd.concat(_all, ignore_index=True)
    S = S[S.result.str.upper().str.strip().isin(["W","L","P"])]
else:
    S = pd.DataFrame(columns=["market","tier","result","week"])
H.append('<h3 class="sd">Season to date</h3>')
if len(S)==0:
    H.append('<div class="card none">No graded picks yet. Records appear here automatically '
             'once games have final scores.</div>')
else:
    H.append('<div class="scroll"><table><thead><tr><th>Market</th><th>Grade</th>'
             '<th>W</th><th>L</th><th>P</th><th>Win %</th><th>Picks</th></tr></thead><tbody>')
    for mk in ["Spread","Moneyline","Total"]:
        blk = S[S.market==mk]
        if len(blk)==0: continue
        for gd in ["A","B","C","D"]:
            g2 = blk[blk.tier==gd]
            if len(g2)==0: continue
            w=int((g2.result=="W").sum()); l=int((g2.result=="L").sum()); p=int((g2.result=="P").sum())
            pct = f"{100*w/(w+l):.1f}%" if (w+l) else "—"
            H.append(f'<tr><td>{mk}</td><td><span class="tag {tc(gd)}">{gd}</span></td>'
                     f'<td class="num">{w}</td><td class="num">{l}</td><td class="num">{p}</td>'
                     f'<td class="num">{pct}</td><td class="num">{w+l+p}</td></tr>')
        w=int((blk.result=="W").sum()); l=int((blk.result=="L").sum()); p=int((blk.result=="P").sum())
        pct = f"{100*w/(w+l):.1f}%" if (w+l) else "—"
        H.append(f'<tr class="tot"><td>{mk}</td><td>all</td><td class="num">{w}</td>'
                 f'<td class="num">{l}</td><td class="num">{p}</td><td class="num">{pct}</td>'
                 f'<td class="num">{w+l+p}</td></tr>')
    H.append('</tbody></table></div>')

_ord = {"A":0,"B":1,"C":2,"D":3}
for mkt in ["Spread","Moneyline","Total"]:
    sub = L[L.market==mkt].copy()
    if len(sub)==0: continue
    sub["_o"] = sub.tier.map(_ord).fillna(9)
    sub = sub.sort_values(["_o","gap"], ascending=[True,False])
    H.append(f'<details class="mkc" open><summary class="mk">{mkt} <span class="mkn">{len(sub)} picks</span></summary>'
             '<div class="scroll"><table><thead><tr>'
             '<th>Date</th><th>Day</th><th>Game</th><th>Pick</th><th>Our #</th>'
             '<th>Market #</th><th>Gap</th><th>Grade</th><th>Result</th></tr></thead><tbody>')
    for _,y in sub.iterrows():
        res = str(y.result).strip() if str(y.result).strip() in ("W","L","P") else ""
        rcls = {"W":"rw","L":"rl","P":"rp"}.get(res,"")
        H.append(f'<tr class="lrow" data-day="{y.day}" data-tier="{y.tier}" '
                 f'data-teams="{y.game.replace("@"," ")}" data-res="{res or "pending"}">'
                 f'<td class="num">{str(y.date)[5:].replace("-","/")}</td>'
                 f'<td class="num">{str(y.day)[:3]}</td><td class="game">{y.game}</td>'
                 f'<td class="pick">{y.pick}</td><td class="num">{y.our_num}</td>'
                 f'<td class="num">{y.market_num}</td><td class="num">{y.gap}</td>'
                 f'<td><span class="tag {tc(y.tier)}">{y.tier}</span></td>'
                 f'<td class="{rcls}">{res or "—"}</td></tr>')
    H.append('</tbody></table></div></details>')
H.append('</div>')

H.append(f"""<details id="s-help"><summary>How to read this card</summary>

<h4 data-tab="all">Quick start</h4>
<p>The dropdown at the top switches between three views: the <b>Weekly Card</b>, the <b>Team Cheat Sheet</b>, and the <b>Ledger</b>. Within the card, use the day and game checkboxes to narrow the slate — leaving a group fully ticked or fully empty means it filters nothing.</p>
<p>Each summary row shows both sides of a market with our pick in bold, and a grade. Below, game notes carry the context numbers cannot: injuries, returning players, trades, coaching changes and line movement.</p>

<h4 data-tab="card">The Grade</h4>
<p>One grade per market, scored out of 100 from two ingredients.</p>
<table><tbody>
<tr><td style="width:235px"><b>Simulated win probability</b><br><span class="wt">70 of 100 points</span></td><td>How often this pick wins across 20,000 simulations of the game. Scaled so 40% earns nothing and 60% or better earns the full 70. This is the dominant term.</td></tr>
<tr><td><b>Value over the price</b><br><span class="wt">30 of 100 points</span></td><td>Simulated probability minus what the price implies, after stripping the book's cut. Scaled so no edge earns nothing and 8 points or more earns the full 30.</td></tr>
</tbody></table>
<p class="sh">Score bands</p>
<table><tbody>
<tr><td style="width:60px"><span class="tag t-a">A</span></td><td>70 or higher</td></tr>
<tr><td><span class="tag t-b">B</span></td><td>55 to 69</td></tr>
<tr><td><span class="tag t-c">C</span></td><td>40 to 54</td></tr>
<tr><td><span class="tag t-d">D</span></td><td>Under 40</td></tr>
</tbody></table>
<p class="sh">Minimum value gates</p>
<p>Because win probability carries most of the weight, a heavy favorite could otherwise grade well while offering almost nothing over the price. Three caps prevent that: under 2 points of edge cannot grade A, under 1 point cannot grade above C, and under half a point is always D, however likely the pick is to win.</p>

<h4 data-tab="card">The simulation</h4>
<p>Every game runs 20,000 times rather than through a single formula. Each run makes two draws.</p>
<table><tbody>
<tr><td style="width:200px"><b>Our own uncertainty</b></td><td>We do not know the true line, only our estimate of it. Each run pulls a true line from a distribution centered on our number with a 3-point spread — an admission the rating could be wrong.</td></tr>
<tr><td><b>The game itself</b></td><td>Given that line, the margin is drawn with a standard deviation of about 13.2 points, adjusted for high totals and high wind. Totals are drawn separately at 10.4 points.</td></tr>
</tbody></table>
<p>Counting across all runs gives the probability each side wins outright, covers, and that the total goes over.</p>
<p class="sh">Why this finds underdogs</p>
<p>Layering our own uncertainty on top widens the distribution. A team the market prices at 36% to win may simulate at 44%. Those surface as strong moneyline grades on teams the book has as underdogs, often beside a weak grade on that same team's spread. Not a contradiction: the team wins more often than the price implies, but still rarely covers a large number.</p>
<p class="sh">Worked example</p>
<table><tbody>
<tr><td style="width:200px">Market</td><td>Seattle −3.5, moneyline −185</td></tr>
<tr><td>Our line</td><td>Ratings, home field, rest and weather give Seattle −5.0</td></tr>
<tr><td>Simulate</td><td>Seattle covers in 54.5% of runs, wins outright in 63%</td></tr>
<tr><td>Spread grade</td><td>Win 54.5% earns 51 of 70. Edge 4.5 points earns 17 of 30. Score 68 → <b>B</b></td></tr>
<tr><td>Moneyline grade</td><td>−185 de-vigs to about 62%. Our 63% is a 1-point edge, so the gate caps it → <b>C</b>. Seattle wins, but you pay full price.</td></tr>
</tbody></table>

<h4 data-tab="card">The "What this tells you" column</h4>
<table><tbody>
<tr><td style="width:245px"><b>"The moneyline prices KC as a 4.6-point favorite, but the spread is only 3.5"</b></td><td>The book's two prices disagree by more than a point, so one is stale. If you back that team, the named market is the better-priced ticket. It says nothing about which side to take.</td></tr>
<tr><td><b>"Markets point different ways"</b></td><td>The best cover and the best value to win outright are different teams. Legitimate: a team can be likely to win without winning by the number. Low-scoring games make this common, because they compress margins toward zero.</td></tr>
<tr><td><b>"Our two models disagree"</b></td><td>The EPA model uses play-by-play efficiency; the Elo model uses game results only. When they diverge past 1.5 standard deviations of that week's spread between them, one is badly wrong with no way to tell which. Low information.</td></tr>
<tr><td><b>"nothing unusual"</b></td><td>No pricing inconsistency, no split between markets, no model disagreement.</td></tr>
</tbody></table>

<h4 data-tab="card">Other columns</h4>
<table><tbody>
<tr><td style="width:130px"><b>Spread</b></td><td>Points a team must win by, or can lose by. "KC −3" means Kansas City must win by 4 or more.</td></tr>
<tr><td><b>Moneyline</b></td><td>Straight bet on who wins, no points. Negative marks the favorite.</td></tr>
<tr><td><b>Total</b></td><td>Combined points by both teams.</td></tr>
<tr><td><b>HFA</b></td><td>Home field advantage in points, shown in game notes. Venue-specific, roughly 0.8 to 2.2. Not a flat 3.</td></tr>
<tr><td><b>Scheme</b></td><td>Early-down pass rate. Top 30% of teams pass-heavy, bottom 30% run-heavy.</td></tr>
</tbody></table>

<h4 data-tab="cheat">The cheat sheet</h4>
<p>All 32 teams, ranked within conference and within division. Offense and defense ranks are league-wide out of 32. Offensive scheme is early-down pass rate: the top 30% are pass-heavy, the bottom 30% run-heavy. Defensive identity compares pass-defense rank against rush-defense rank — a team stronger against the pass by 8 or more places is labeled pass-stopping.</p>
<p>Takeaways and giveaways are per game. TO diff is takeaways minus giveaways; positive is good. Turnovers are among the least stable stats in football and regress hard year to year, so treat a big number as a description of last season rather than a prediction of this one.</p>
<p>The banner at the top of the tab states whether the figures are preseason projections or updated through completed games.</p>

<h4 data-tab="ledger">The ledger</h4>
<p>Every pick logged for the week, split into one table per market and sorted best grade first. Results fill in automatically once a game has a final score: W, L or P for each market. Nothing to enter by hand.</p>
<p>Filters narrow by day, grade and result. Clearing any group shows nothing until you make a selection.</p>

<h4 data-tab="all">Where the numbers come from</h4>
<p>Power ratings are opponent-adjusted EPA and success rate from play-by-play, weighted toward passing, with an eight-game recency half-life. They correlate 0.935 with actual point differentials. Lines, schedules, rest days and injuries come from nflverse; weather from Open-Meteo. All free, no accounts.</p>
<p>Trades, coaching changes, suspensions, line movement and expert picks are not in any dataset. They are researched separately and attached to the card.</p>

</details>

<script>
var NAVS={{card:[['s-issues','Needs attention'],['s-summary','Summary'],['s-notes','Game notes'],['s-help','How to read']],
 cheat:[['s-cheat','Conference ranks'],['s-div','Division ranks'],['s-help','How to read']],
 ledger:[['s-ledger','Ledger'],['s-help','How to read']]}};
function mkLink(p){{var t=document.getElementById(p[0]);if(!t)return null;
 var a=document.createElement('a');a.href='#'+p[0];a.textContent=p[1];
 a.onclick=function(e){{e.preventDefault();
   if(p[0]==='s-help'){{var d=document.getElementById('s-help');if(d)d.open=true;}}
   t.scrollIntoView({{behavior:'smooth',block:'start'}});}};
 return a;}}
function buildNav(v){{
 var n=document.getElementById('nav');
 if(n){{n.innerHTML='';(NAVS[v]||[]).forEach(function(p){{var a=mkLink(p);if(a)n.appendChild(a);}});}}
 // per-section strips: every other section, excluding its own
 [].slice.call(document.querySelectorAll('.secnav')).forEach(function(el){{
   el.innerHTML='';
   var own=el.dataset.own;
   var all=[['s-summary','Summary'],['s-notes','Game notes'],['s-cheat','Conference ranks'],
            ['s-div','Division ranks'],['s-ledger','Ledger'],['s-help','How to read']];
   all.forEach(function(p){{ if(p[0]===own) return; var a=mkLink(p); if(a) el.appendChild(a); }});}});}}
function setView(){{var v=document.getElementById('view').value;
['card','cheat','ledger'].forEach(function(k){{
  var el=document.getElementById('v-'+k); if(el) el.classList.toggle('hide',v!==k);}});
buildNav(v);
 [].slice.call(document.querySelectorAll('#s-help h4')).forEach(function(hd){{
   var t=hd.dataset.tab||'all', on=(t==='all'||t===v);
   hd.style.display=on?'':'none';
   var n=hd.nextElementSibling;
   while(n && n.tagName!=='H4'){{n.style.display=on?'':'none';n=n.nextElementSibling;}}}});
 var any=false;
 [].slice.call(document.querySelectorAll('.iss')).forEach(function(el){{
   var t=el.dataset.tab||'card', on=(t==='all'||t===v);
   el.style.display=on?'':'none'; if(on)any=true;}});
 var hdr=document.getElementById('s-issues');
 if(hdr) hdr.classList.toggle('hide',!any);}}
setView();flt();lflt();
function goWeek(){{var v=document.getElementById('wk').value; if(v) location.href=v;}}
function tog(){{var d=document.body.classList.toggle('dark');
document.getElementById('thm').textContent=d?'Light mode':'Dark mode';
try{{localStorage.setItem('nflthm',d?'1':'0')}}catch(e){{}}}}
try{{if(localStorage.getItem('nflthm')==='1'){{document.body.classList.add('dark');
document.getElementById('thm').textContent='Light mode';}}}}catch(e){{}}
function flt(){{
 var dAll=document.querySelectorAll('.fd').length, gAll=document.querySelectorAll('.fgm').length;
 var ds=[].slice.call(document.querySelectorAll('.fd:checked')).map(function(e){{return e.value}});
 var gs=[].slice.call(document.querySelectorAll('.fgm:checked')).map(function(e){{return e.value}});
 // a group with nothing ticked, or everything ticked, imposes no constraint
 // an empty group hides everything; a fully ticked group imposes no constraint
 [].slice.call(document.querySelectorAll('.row')).forEach(function(rw){{
   var ok;
   if(ds.length===0 || gs.length===0) ok = false;
   else ok = (ds.length===dAll || ds.indexOf(rw.dataset.day)>-1)
          && (gs.length===gAll || gs.indexOf(rw.dataset.gid)>-1);
   rw.style.display = ok ? '' : 'none';}});
 var n=document.querySelectorAll('#v-card tbody .row:not([style*="none"])').length;
 var c=document.getElementById('fcount'); if(c) c.textContent=n+' of '+gAll+' games shown';}}
function expAll(o){{[].slice.call(document.querySelectorAll('details.gc')).forEach(function(d){{d.open=!!o;}});}}
function lflt(){{
 var dA=document.querySelectorAll('.ld').length,tA=document.querySelectorAll('.lt').length,rA=document.querySelectorAll('.lr').length;
 var ds=[].slice.call(document.querySelectorAll('.ld:checked')).map(function(e){{return e.value}});
 var ts=[].slice.call(document.querySelectorAll('.lt:checked')).map(function(e){{return e.value}});
 var mA=document.querySelectorAll('.ltm').length;
 var ms=[].slice.call(document.querySelectorAll('.ltm:checked')).map(function(e){{return e.value}});
 var rs=[].slice.call(document.querySelectorAll('.lr:checked')).map(function(e){{return e.value}});
 var n=0;
 [].slice.call(document.querySelectorAll('.lrow')).forEach(function(rw){{
   var ok;
   if(ds.length===0||ts.length===0||rs.length===0||ms.length===0) ok=false;
   else{{ var tm=rw.dataset.teams.split(' ');
     ok=(ds.length===dA||ds.indexOf(rw.dataset.day)>-1)
      &&(ts.length===tA||ts.indexOf(rw.dataset.tier)>-1)
      &&(rs.length===rA||rs.indexOf(rw.dataset.res)>-1)
      &&(ms.length===mA||ms.indexOf(tm[0])>-1||ms.indexOf(tm[1])>-1);}}
   rw.style.display=ok?'':'none'; if(ok)n++;}});
 var c=document.getElementById('lcount'); if(c)c.textContent=n+' picks shown';}}
function lAll(o){{[].slice.call(document.querySelectorAll('.ld,.lt,.lr,.ltm')).forEach(function(e){{e.checked=!!o}});lflt();}}
function allOn(){{[].slice.call(document.querySelectorAll('.fd,.fgm')).forEach(function(e){{e.checked=true}});flt();}}
function allOff(){{[].slice.call(document.querySelectorAll('.fd,.fgm')).forEach(function(e){{e.checked=false}});flt();}}
</script></div></body></html>""")

_html = "".join(H)

# ---- self-check: catch template and structure faults before writing ----
import re as _re
_ph = set(_re.findall(r'\{[A-Za-z_][A-Za-z0-9_().\[\]]*\}', _html)) - {"{return e.value}"}
if _ph:
    issue("ERROR", f"{len(_ph)} template placeholder(s) did not render",
          f"Raw code is showing on the page instead of values: {', '.join(sorted(_ph)[:6])}. "
          "Usually means an f-string prefix was lost when the template was edited.",
          "Check the f-string segments in nfl_card.py around the <style> block.", tab="all")
if "{{" in _html:
    issue("ERROR", "Unescaped double braces in the output",
          "CSS or script braces are doubled on the page, which breaks styling.",
          "A template segment is missing its f-string prefix.", tab="all")
for _t in ["div","table","tbody","details","summary"]:
    _o = len(_re.findall(r"<%s[ >]" % _t, _html)); _c = _html.count("</%s>" % _t)
    if _o != _c:
        issue("ERROR", f"Unbalanced <{_t}> tags ({_o} open, {_c} closed)",
              "The page may render with sections nested wrongly or cut off.",
              "Report this — it is a bug in the card generator, not your data.", tab="all")
if len(D) and _html.count('class="gc row"') != len(D):
    issue("ERROR", "Game card count does not match the slate",
          f"{_html.count('class=\"gc row\"')} cards rendered for {len(D)} games.",
          "Report this — a game is missing from the notes section.", tab="card")

# consistency: every team's stats must match the ratings table wherever they appear
_incons = []
for _t in set(D.home) | set(D.away):
    _exp = (int(r.loc[_t,"off_rank"]), int(r.loc[_t,"def_rank"]), int(r.loc[_t,"pwr_rank"]),
            str(r.loc[_t,"scheme"]), str(r.loc[_t,"dsch"]))
    _seen = set()
    for _,_g in D.iterrows():
        if _g.home == _t: _seen.add((_g.h_off,_g.h_def,_g.h_pwr,_g.h_sch,_g.h_dsch))
        if _g.away == _t: _seen.add((_g.a_off,_g.a_def,_g.a_pwr,_g.a_sch,_g.a_dsch))
    if len(_seen) > 1 or (_seen and _exp not in _seen):
        _incons.append(_t)
if _incons:
    issue("ERROR", f"Team stats disagree between cards: {', '.join(sorted(_incons)[:8])}",
          "The same team is showing different rankings in different places on this page, "
          "so at least one card is wrong.",
          "Report this — it is a bug in the card generator, not your data.", tab="all")
_expected_cols = {"off_rank","def_rank","pwr_rank","po_rank","ro_rank","pd_rank","rd_rank",
                  "scheme","dsch","Giveaways","Takeaways"}
_missing = _expected_cols - set(r.columns)
if _missing:
    issue("WARN", f"Ratings file is missing {len(_missing)} stat column(s)",
          f"Some stats will show as blank or zero across every card: {', '.join(sorted(_missing))}.",
          "Re-run build_ratings.py to regenerate power_ratings.csv with all columns.", tab="all")

# re-render the issues panel now that self-checks have run
if ISSUES:
    _panel = ['<h2 id="s-issues" class="hide">Needs attention</h2>']
    for i in ISSUES:
        _panel.append(f'<div class="iss" data-tab="{i.get("tab","card")}"><div class="h">{i["lvl"]} — {i["what"]}</div>'
                      f'<div class="w"><b>Why it matters:</b> {i["why"]}</div>'
                      f'<div class="f"><b>Fix:</b> {i["fix"]}</div></div>')
    _new = "".join(_panel)
    if 'id="s-issues"' in _html:
        _s = _html.index('<h2 id="s-issues"'); _e = _html.index('<div id="v-card">')
        _html = _html[:_s] + _new + _html[_e:]
    else:
        _a = _html.index('<div id="v-card">')
        _html = _html[:_a] + _new + _html[_a:]

open(P(f"Week{WEEK}-Card.html"),"w").write(_html)
print(f"\ncard: Week{WEEK}-Card.html | ledger: ledger_week{WEEK}.csv | {len(D)} games, {len(L)} picks"
      + (f" | {pres} results preserved" if pres else ""))
