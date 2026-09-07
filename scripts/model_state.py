"""
model_state.py — one place every number lives, and one way to read it.

The college card was rebuilt around a file like this after the same constants
were found sitting in four scripts, kept in step by remembering to. Nothing
failed when a copy drifted. The run went green and the published page was
simply wrong.

This card had the same defect, in miniature and already live: build_ratings.py
regressed last season's ratings toward the mean with OFF_REGRESS 0.35 and
DEF_REGRESS 0.45 and wrote the result to a Prior2026 column, and nfl_card.py
then IGNORED that column and recomputed the identical number from its own
copy of the constants written as 0.65 and 0.55. Verified against the committed
power_ratings.csv: the two agree to the fourth decimal, which is the rounding
in the CSV. One number, two files, no way to notice a drift.

Every value is also compiled in below. If model_state.json is missing, broken
or truncated the scripts still run on exactly the values they ran on before
this module existed, so a card still gets built. But FELL_BACK is set and the
card reports it, so a silent fallback cannot look like a normal run.

Nothing here imports anything outside the standard library, so it is safe to
import from any of the scripts.
"""
import json, os, tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, "model_state.json")

# The values as they stood when this module was written, taken from the scripts
# themselves rather than retyped from memory. These are the floor: a value
# present in the JSON wins, a value missing from it falls back to here.
DEFAULTS = {
    "schema": 1,
    # --- build_ratings.py ---
    "ratings": {
        # Which seasons the fit is allowed to see. This is a LIST because the
        # in-season blend needs two: last year carried at a discount and this
        # year at full weight, the way the college model does it. It holds one
        # season today because that is what the card is running on, and today
        # that is correct: the 2026 season has not been played. It stops being
        # correct the moment it has. See "known gap" in the readme section of
        # model_state.json.
        "seasons": [2025],
        "half_life": 8.0,        # weeks, decay within a season
        "off_regress": 0.35,     # toward the mean, for the next-season prior
        "def_regress": 0.45,
        "plays_per_game": 63.0,  # league average offensive plays
        "ridge_alpha": 25.0,
        "early_down_weight": 1.5,
        "epa_weight": 0.6,       # blended against success rate, which takes the rest
        "composite": {"pass_off": 0.40, "pass_def": 0.25,
                      "rush_off": 0.12, "rush_def": 0.08},
    },
    # --- nfl_card.py grading ---
    # Chosen, not measured. There is no backtest behind this card yet, so every
    # one of these is a decision rather than a finding, and they are grouped on
    # their own to keep that obvious.
    "grading": {
        "win_points": 70.0,      # the simulated win probability carries this much
        "value_points": 30.0,    # value over the price carries the rest
        "p_floor": 0.40,         # win probability ramp: this maps to 0 points
        "p_ceil": 0.60,          # and this to the full win_points
        "value_ceil": 8.0,       # edge in probability points that earns full value
        "grade_a": 70.0, "grade_b": 55.0, "grade_c": 40.0,
        # minimum value gates, so a heavy favourite cannot grade well on no edge
        "gate_a": 2.0, "gate_b": 1.0, "gate_d": 0.5,
    },
    # Nothing has been measured on this card. The college card's equivalent
    # section holds a backtested record per strategy and the grades are capped
    # by it. This one is deliberately empty rather than absent, so the shape is
    # there and the emptiness is visible.
    "backtest": {
        "run_utc": "",
        "window": "",
        "record": {},
        "record_notes": {},
        "headline": {},
    },
}

FELL_BACK = ""        # empty when the file was read cleanly; the reason otherwise


def _merge(base, over):
    """Overlay one dict on another, one level of nesting deep."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load():
    global FELL_BACK
    try:
        with open(PATH) as f:
            disk = json.load(f)
        if not isinstance(disk, dict):
            raise ValueError("top level is not an object")
        return _merge(DEFAULTS, disk)
    except FileNotFoundError:
        FELL_BACK = "model_state.json is missing"
    except Exception as e:
        FELL_BACK = f"model_state.json could not be read ({e})"
    print(f"  WARNING: {FELL_BACK}; using the values compiled into model_state.py")
    return _merge(DEFAULTS, {})


STATE = load()
RATINGS = STATE["ratings"]
GRADING = STATE["grading"]
BACKTEST = STATE["backtest"]


def record():
    """Backtested win-loss per strategy, as tuples, ignoring malformed entries.

    Empty until this card has a backtest. Kept so the callers that will read it
    do not have to be written twice.
    """
    out = {}
    for k, v in (BACKTEST.get("record") or {}).items():
        try:
            w, l = int(v[0]), int(v[1])
        except Exception:
            continue
        if w >= 0 and l >= 0:
            out[k] = (w, l)
    return out


def save(section, values):
    """Merge one section back into the file on disk, leaving the rest alone.

    Written to a temporary file in the same directory and moved into place, so
    an interrupted write cannot leave a half-written file that the next run
    then falls back from.
    """
    try:
        with open(PATH) as f:
            disk = json.load(f)
    except Exception:
        disk = {}
    cur = disk.get(section)
    disk[section] = _merge(cur, values) if isinstance(cur, dict) else values
    fd, tmp = tempfile.mkstemp(dir=BASE, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(disk, f, indent=2)
            f.write("\n")
        os.replace(tmp, PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print(f"  model_state.json: updated '{section}'")


if __name__ == "__main__":
    print(json.dumps(STATE, indent=2))
