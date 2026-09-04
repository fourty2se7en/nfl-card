"""
track_gaps.py — the one open empirical question.

Do high-gap leans (where our model disagrees most with the market)
outperform low-gap leans? If they ever do, the confidence model earns
a revision. Right now every backtest says no.

Reads all ledger_week*.csv with the `result` column filled in.
Usage: python3 track_gaps.py
"""
import glob, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
import os
BASE = os.environ.get("NFL_DIR") or os.path.dirname(os.path.abspath(__file__))
P = lambda x: os.path.join(BASE, x)


files = sorted(glob.glob(P("ledger_week*.csv")))
if not files:
    raise SystemExit("no ledger files found")

L = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
L["result"] = L.result.astype(str).str.upper().str.strip()
graded = L[L.result.isin(["W", "L", "P"])].copy()

print(f"ledger rows: {len(L)}   graded: {len(graded)}   ungraded: {len(L)-len(graded)}")
if len(graded) == 0:
    print("\nNothing graded yet. Fill the `result` column (W/L/P) after games.")
    print("This report becomes meaningful around 100+ graded leans.")
    raise SystemExit

g = graded[graded.result != "P"].copy()
g["gap"] = pd.to_numeric(g.gap, errors="coerce").abs()
g["win"] = (g.result == "W").astype(int)

def block(sub, label):
    if len(sub) == 0: return
    print(f"\n{label}  (n={len(sub)})")
    print(f"  overall: {sub.win.sum()}-{len(sub)-sub.win.sum()}  {100*sub.win.mean():.1f}%")
    b = sub.dropna(subset=["gap"])
    if len(b) < 10: return
    print(f"  {'gap':>10} {'record':>11} {'win%':>7}")
    for lo, hi, lbl in [(0,1,"0-1"),(1,2,"1-2"),(2,3,"2-3"),(3,5,"3-5"),(5,99,"5+")]:
        m = b[(b.gap >= lo) & (b.gap < hi)]
        if len(m) == 0: continue
        w = m.win.sum(); l = len(m)-w
        print(f"  {lbl:>10} {f'{w}-{l}':>11} {100*m.win.mean():>6.1f}%")
    # the actual test: does gap size predict?
    if b.gap.nunique() > 3:
        r = np.corrcoef(b.gap, b.win)[0,1]
        lo = b[b.gap < b.gap.median()].win.mean()
        hi = b[b.gap >= b.gap.median()].win.mean()
        print(f"  gap-vs-win correlation: {r:+.3f}")
        print(f"  low-gap {100*lo:.1f}%  vs  high-gap {100*hi:.1f}%  "
              f"({100*(hi-lo):+.1f} pts)")

for mkt in ["Spread", "Total", "Moneyline"]:
    block(g[g.market == mkt], mkt.upper())
block(g, "ALL MARKETS")

print("\n" + "="*58)
print("VERDICT")
print("="*58)
b = g.dropna(subset=["gap"])
if len(b) < 100:
    print(f"  Only {len(b)} graded leans. Need 100+ before this means anything.")
else:
    hi = b[b.gap >= b.gap.median()].win.mean()*100
    lo = b[b.gap < b.gap.median()].win.mean()*100
    if hi > lo + 3 and hi > 52.4:
        print(f"  High-gap leans outperforming ({hi:.1f}% vs {lo:.1f}%).")
        print("  >> Confidence model may deserve revision. Re-run the backtest.")
    else:
        print(f"  High-gap {hi:.1f}% vs low-gap {lo:.1f}% — no edge from gap size.")
        print("  >> Consistent with the backtest. Keep everything marked NO EDGE.")
print(f"\nBreak-even reference: 52.4% at -110")
