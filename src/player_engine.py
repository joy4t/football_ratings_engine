"""
PlayerNation — Player Ratings Engine (v1)
=========================================
8 contribution dimensions -> per-90 rate -> minutes-weighted shrinkage toward
role mean -> percentile within position group -> ROLE-AWARE composite.
Plus: a profile-only BEHAVIOUR layer and a CONFIDENCE band per player.

Pipeline (per dimension, per player):
    event -> quality-weighted contribution -> per-90 rate
          -> shrinkage toward role mean -> percentile within role -> score 0-100
    composite = average of the player's EXPECTED dimension scores (role-aware)
    behaviour = style lens on the same events (profile-only, never in composite)
    confidence = data-sufficiency band from minutes (High / Medium / Low)

RUN (Anaconda Prompt, from project root):
    conda activate base
    cd /d D:\\DS_Projects\\Playernation
    python src\\player_engine.py

INPUTS (already on disk):
    data\\world_cup\\*.json
    wyscout-soccer-match-event-dataset\\raw_data\\matches.zip

OUTPUTS (output\\):
    player_ratings.csv         8 dim scores + role-aware composite + confidence
    player_behaviour.csv       4 behaviour traits (profile-only) + confidence
    player_explanations.csv    top action drivers per player per dimension
    player_thirds.csv          share of value by pitch third
    player_thirds_detail.csv   contribution per dimension per third

Every number in CONFIG is a PRODUCT DECISION you own and can defend.
"""

import json
import zipfile
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# =====================================================================
#  CONFIG  —  YOUR DECISIONS. Tune and defend every number here.
# =====================================================================

# NOTE: run from the PROJECT ROOT (paths are relative to it), even though this
# file lives in src\. i.e.  python src\player_engine.py  from D:\DS_Projects\Playernation
DATA_DIR         = os.path.join("data", "world_cup")
RAW_MATCHES_ZIP  = os.path.join("wyscout-soccer-match-event-dataset", "raw_data", "matches.zip")
RAW_MATCHES_NAME = "matches_World_Cup.json"
OUTPUT_DIR       = "output"

K_SHRINKAGE = 300.0    # minutes of evidence before a player is trusted on their own rate

# Confidence bands (replaces the old hard minutes cutoff — everyone is shown).
CONF_HIGH_MIN = 450.0  # >= this -> High confidence
CONF_MED_MIN  = 180.0  # >= this -> Medium; below -> Low (shown but "indicative")

# Role -> EXPECTED dimensions. ONLY these feed a player's composite.
# Non-expected ("supporting") dimensions are still computed and shown in the
# profile — they just don't dilute the headline number.
# >>> This map is a configurable product decision, editable per position. <<<
ROLE_EXPECTED = {
    "Goalkeeper": ["goal_protection", "retention", "transition"],
    "Defender":   ["recovery", "defensive_disruption", "retention", "progression"],
    "Midfielder": ["retention", "progression", "recovery", "chance_creation", "transition"],
    "Forward":    ["finishing", "chance_creation", "progression", "retention"],
}

# Pitch geometry. Wyscout coords are normalized to the ACTING team's attack:
#   x = 0 own goal line, x = 100 opponent goal line; y = 0..100 across.
FINAL_THIRD_X = 66
BOX_X         = 84
BOX_Y_LOW     = 19
BOX_Y_HIGH    = 81
OWN_BOX_X     = 16

# ---- Wyscout tag IDs (verified against the Wyscout API tag dictionary) ----
T_GOAL = 101; T_OWN_GOAL = 102; T_ASSIST = 301; T_KEY_PASS = 302
T_DUEL_LOST = 701; T_DUEL_NEUTRAL = 702; T_DUEL_WON = 703
T_THROUGH = 901; T_INTERCEPTION = 1401; T_SLIDING = 1601
T_RED = 1701; T_YELLOW = 1702; T_SECOND_YELLOW = 1703
T_ACCURATE = 1801; T_INACCURATE = 1802; T_COUNTER = 1901
T_DANGEROUS_LOST = 2001; T_BLOCKED = 2101

# ---- CONTRIBUTION WEIGHT TABLES (per dimension) ----
W = {
    "recovery": {
        "interception":       3.0,
        "def_duel_won":       2.0,
        "def_duel_lost":     -0.5,
        "loose_ball_won":     1.0,
    },
    "retention": {
        "accurate_pass":      0.10,
        "lost_ball":         -1.0,
        "dangerous_lost":    -3.0,
        "att_duel_lost":     -0.5,
    },
    "progression": {   # LOCKED: line-breaking / final-third / box, NOT raw distance
        "final_third_entry":  3.0,
        "pass_into_box":      4.0,
        "line_breaking":      3.0,
        "forward_carry":      1.5,
        "forward_pass_small": 0.15,
        "failed_progressive":-0.4,
    },
    "chance_creation": {
        "assist":             8.0,
        "key_pass":           4.0,
        "accurate_cross":     1.0,
        "pass_into_box":      1.5,
    },
    "finishing": {
        "goal":               8.0,
        "shot_on_target":     2.0,
        "shot_off_target":   -0.3,
        "shot_blocked":       0.0,
    },
    "defensive_disruption": {
        "clearance":          1.5,
        "sliding_tackle_won": 2.5,
    },
    "goal_protection": {
        "save":               4.0,
        "reflex_save":        5.0,
        "keeper_sweep":       2.0,
        "own_box_clearance":  1.0,
    },
    "transition": {
        "counter_action":     2.0,
        "keeper_launch":      1.5,
    },
}

# ---- BEHAVIOUR weights (profile-only; describe STYLE, never ability) ----
# These combine raw tallies into 4 trait signals, then percentile within role.
B = {
    "card_yellow":      1.0,
    "card_red":         3.0,
    "simulation":       2.0,
    "protest":          1.0,
    "dangerous_loss":   2.0,   # for reliability penalty
}

DIMENSIONS = ["recovery", "retention", "progression", "chance_creation",
              "finishing", "defensive_disruption", "goal_protection", "transition"]
BEHAVIOURS = ["discipline", "aggression", "risk_appetite", "reliability"]


# =====================================================================
#  LOAD DATA
# =====================================================================

def load_matches(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(
            f"No match files in {data_dir!r}. Run this from the PROJECT ROOT "
            f"(D:\\DS_Projects\\Playernation), e.g.  python src\\player_engine.py"
        )
    return [json.load(open(f, encoding="utf-8")) for f in files]


def build_player_registry(matches):
    reg = {}
    for m in matches:
        for team_players in m["players"].values():
            for wrapper in team_players:
                if not wrapper:
                    continue
                p = wrapper["player"]
                reg[p["wyId"]] = {"name": p["shortName"], "role": p["role"]["name"]}
    return reg


def load_minutes(zip_path, member, end_default=90.0):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"{zip_path!r} not found — this raw matches.zip holds substitution data "
            f"needed for minutes."
        )
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as f:
            raw = json.load(f)
    minutes = defaultdict(float)
    for m in raw:
        end_min = 120.0 if m.get("duration") == "ExtraTime" else end_default
        for team in m["teamsData"].values():
            formation = team.get("formation") or {}
            subs = formation.get("substitutions")
            if not isinstance(subs, list):
                subs = []
            sub_out, sub_in = {}, {}
            for s in subs:
                try:
                    mn = float(s["minute"])
                except (KeyError, TypeError, ValueError):
                    continue
                if s.get("playerOut"):
                    sub_out[s["playerOut"]] = mn
                if s.get("playerIn"):
                    sub_in[s["playerIn"]] = mn
            for p in (formation.get("lineup") or []):
                pid = p["playerId"]
                minutes[pid] += max(0.0, sub_out.get(pid, end_min) - 0.0)
            for pid, in_m in sub_in.items():
                minutes[pid] += max(0.0, sub_out.get(pid, end_min) - in_m)
    return minutes


# =====================================================================
#  PER-EVENT CONTRIBUTION SCORING
# =====================================================================

def tags_of(ev):
    return {t["id"] for t in ev.get("tags", [])}


def coords(ev):
    pos = ev.get("positions", [])
    if len(pos) >= 2:
        return pos[0]["x"], pos[0]["y"], pos[1]["x"], pos[1]["y"]
    if len(pos) == 1:
        return pos[0]["x"], pos[0]["y"], None, None
    return None, None, None, None


def in_box(x, y):
    return x is not None and y is not None and x >= BOX_X and BOX_Y_LOW <= y <= BOX_Y_HIGH


def third_of(x0):
    if x0 is None:
        return "unknown"
    if x0 < 33:
        return "defensive"
    if x0 < 66:
        return "middle"
    return "attacking"


def score_event(ev, role):
    out = []
    name = ev["eventName"]
    sub  = ev["subEventName"]
    tags = tags_of(ev)
    x0, y0, x1, y1 = coords(ev)
    acc = T_ACCURATE in tags

    def add(dim, cat):
        out.append((dim, cat, W[dim][cat]))

    if name == "Pass":
        forward = (x0 is not None and x1 is not None and x1 > x0)
        if acc:
            add("retention", "accurate_pass")
            if x0 is not None and x1 is not None and x0 < FINAL_THIRD_X <= x1:
                add("progression", "final_third_entry")
            if in_box(x1, y1) and not in_box(x0, y0):
                add("progression", "pass_into_box")
                add("chance_creation", "pass_into_box")
            if sub == "Smart pass" or T_THROUGH in tags:
                add("progression", "line_breaking")
            if forward:
                add("progression", "forward_pass_small")
            if sub == "Cross":
                add("chance_creation", "accurate_cross")
        else:
            if T_DANGEROUS_LOST in tags:
                add("retention", "dangerous_lost")
            else:
                add("retention", "lost_ball")
            ambitious = (sub == "Smart pass" or T_THROUGH in tags or
                         (x0 is not None and x1 is not None and x0 < FINAL_THIRD_X <= x1))
            if ambitious:
                add("progression", "failed_progressive")
        if T_ASSIST in tags:
            add("chance_creation", "assist")
        if T_KEY_PASS in tags:
            add("chance_creation", "key_pass")
        if acc and T_COUNTER in tags:
            add("transition", "counter_action")
        if role == "Goalkeeper" and acc and forward and sub in ("Launch", "High pass"):
            add("transition", "keeper_launch")

    elif name == "Shot":
        if T_GOAL in tags:
            add("finishing", "goal")
        elif T_BLOCKED in tags:
            add("finishing", "shot_blocked")
        elif acc:
            add("finishing", "shot_on_target")
        else:
            add("finishing", "shot_off_target")

    elif name == "Duel":
        won = T_DUEL_WON in tags
        lost = T_DUEL_LOST in tags
        if sub == "Ground defending duel":
            if won:
                add("recovery", "def_duel_won")
            elif lost:
                add("recovery", "def_duel_lost")
        elif sub == "Ground loose ball duel":
            if won:
                add("recovery", "loose_ball_won")
        elif sub == "Ground attacking duel":
            if lost:
                add("retention", "att_duel_lost")
        if T_SLIDING in tags and won:
            add("defensive_disruption", "sliding_tackle_won")
        if won and T_COUNTER in tags:
            add("transition", "counter_action")

    elif name == "Others on the ball":
        if sub == "Clearance":
            add("defensive_disruption", "clearance")
            if x0 is not None and x0 <= OWN_BOX_X:
                add("goal_protection", "own_box_clearance")
        elif sub == "Acceleration":
            if x0 is not None and x1 is not None and x1 > x0 and x1 >= FINAL_THIRD_X:
                add("progression", "forward_carry")
            if T_COUNTER in tags:
                add("transition", "counter_action")

    elif name == "Save attempt":
        add("goal_protection", "reflex_save" if sub == "Reflexes" else "save")

    elif name == "Goalkeeper leaving line":
        if acc:
            add("goal_protection", "keeper_sweep")

    if T_INTERCEPTION in tags:
        add("recovery", "interception")

    return out


# =====================================================================
#  AGGREGATE (contributions + thirds + behaviour tallies)
# =====================================================================

def aggregate(matches, reg):
    dim_sum = defaultdict(float)
    cat_sum = defaultdict(float)
    cat_cnt = defaultdict(int)
    third_dim_sum = defaultdict(float)
    third_net = defaultdict(float)
    beh = defaultdict(lambda: defaultdict(float))  # pid -> behaviour raw tallies

    for m in matches:
        for ev in m["events"]:
            pid = ev["playerId"]
            if pid == 0:
                continue
            role = reg.get(pid, {}).get("role", "Unknown")
            third = third_of(coords(ev)[0])

            for dim, cat, val in score_event(ev, role):
                dim_sum[(pid, dim)] += val
                cat_sum[(pid, dim, cat)] += val
                cat_cnt[(pid, dim, cat)] += 1
                third_dim_sum[(pid, dim, third)] += val
                third_net[(pid, third)] += val

            # ---- behaviour raw tallies (style lens on the same events) ----
            name = ev["eventName"]; sub = ev["subEventName"]; tags = tags_of(ev)
            bp = beh[pid]
            if name == "Foul":
                if sub == "Simulation":
                    bp["simulations"] += 1
                elif sub == "Protest":
                    bp["protests"] += 1
                else:
                    bp["fouls"] += 1
            if T_YELLOW in tags:
                bp["yellows"] += 1
            if T_RED in tags or T_SECOND_YELLOW in tags:
                bp["reds"] += 1
            if name == "Duel":
                bp["duels"] += 1
                if T_SLIDING in tags:
                    bp["slides"] += 1
            if name == "Pass":
                bp["passes"] += 1
                if T_ACCURATE in tags:
                    bp["passes_acc"] += 1
                x0, y0, x1, y1 = coords(ev)
                ambitious = (sub == "Smart pass" or T_THROUGH in tags
                             or (x0 is not None and x1 is not None and x0 < FINAL_THIRD_X <= x1)
                             or (in_box(x1, y1) and not in_box(x0, y0)))
                if ambitious:
                    bp["ambitious"] += 1
            if T_DANGEROUS_LOST in tags:
                bp["dangerous_losses"] += 1

    return dim_sum, cat_sum, cat_cnt, third_dim_sum, third_net, beh


# =====================================================================
#  CONTRIBUTION RATINGS  (per-90 -> shrink -> role percentile -> composite)
# =====================================================================

def confidence_band(mins):
    if mins >= CONF_HIGH_MIN:
        return "High"
    if mins >= CONF_MED_MIN:
        return "Medium"
    return "Low"


def build_ratings(dim_sum, reg, minutes):
    pids = sorted({pid for (pid, _) in dim_sum})
    rows = []
    for pid in pids:
        mins = minutes.get(pid, 0.0)
        if mins <= 0:
            continue                      # can't compute a rate with no minutes
        info = reg.get(pid, {"name": str(pid), "role": "Unknown"})
        row = {"player_id": pid, "name": info["name"], "role": info["role"],
               "minutes": mins, "confidence": confidence_band(mins)}
        for dim in DIMENSIONS:
            row["raw_" + dim] = dim_sum.get((pid, dim), 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No players with minutes — check minutes loading.")

    for dim in DIMENSIONS:
        df["per90_" + dim] = df["raw_" + dim] / df["minutes"] * 90.0

    df["w"] = df["minutes"] / (df["minutes"] + K_SHRINKAGE)
    for dim in DIMENSIONS:
        role_mean = df.groupby("role")["per90_" + dim].transform("mean")
        df["adj_" + dim] = df["w"] * df["per90_" + dim] + (1.0 - df["w"]) * role_mean

    for dim in DIMENSIONS:
        df["score_" + dim] = df.groupby("role")["adj_" + dim].rank(pct=True) * 100.0

    # role-aware composite: average of EXPECTED dimensions only
    def composite(r):
        expected = ROLE_EXPECTED.get(r["role"], DIMENSIONS)
        return float(np.mean([r["score_" + d] for d in expected]))
    df["composite"] = df.apply(composite, axis=1)

    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


# =====================================================================
#  BEHAVIOUR PROFILE  (percentile within role, profile-only)
# =====================================================================

def build_behaviour(beh, df):
    ranked = set(df["player_id"])
    name_of = dict(zip(df["player_id"], df["name"]))
    role_of = dict(zip(df["player_id"], df["role"]))
    conf_of = dict(zip(df["player_id"], df["confidence"]))
    mins_of = dict(zip(df["player_id"], df["minutes"]))

    rows = []
    for pid in ranked:
        b = beh.get(pid, {})
        mins = mins_of[pid]
        passes = max(b.get("passes", 0.0), 1.0)
        # raw trait signals
        indiscipline_p90 = (b.get("fouls", 0) + B["card_yellow"] * b.get("yellows", 0)
                            + B["card_red"] * b.get("reds", 0)
                            + B["simulation"] * b.get("simulations", 0)
                            + B["protest"] * b.get("protests", 0)) / mins * 90.0
        aggression_p90 = (b.get("duels", 0) + b.get("slides", 0) + b.get("fouls", 0)) / mins * 90.0
        risk_ratio = b.get("ambitious", 0) / passes
        completion = b.get("passes_acc", 0) / passes
        danger_rate = b.get("dangerous_losses", 0) / passes
        reliability_raw = completion - B["dangerous_loss"] * danger_rate
        rows.append({
            "player_id": pid, "name": name_of[pid], "role": role_of[pid],
            "confidence": conf_of[pid],
            "_discipline": -indiscipline_p90,   # higher = more disciplined
            "_aggression": aggression_p90,
            "_risk_appetite": risk_ratio,
            "_reliability": reliability_raw,
        })
    bdf = pd.DataFrame(rows)

    for trait in BEHAVIOURS:
        bdf[trait] = bdf.groupby("role")["_" + trait].rank(pct=True) * 100.0

    cols = ["player_id", "name", "role", "confidence"] + BEHAVIOURS
    bdf = bdf[cols].sort_values("name").reset_index(drop=True)
    for t in BEHAVIOURS:
        bdf[t] = bdf[t].round(1)
    return bdf


# =====================================================================
#  EXPLANATIONS + THIRDS
# =====================================================================

def build_explanations(cat_sum, cat_cnt, df, top_n=3):
    ranked = set(df["player_id"])
    per = defaultdict(list)
    for (pid, dim, cat), total in cat_sum.items():
        if pid in ranked:
            per[(pid, dim)].append((cat, total, cat_cnt[(pid, dim, cat)]))
    name_of = dict(zip(df["player_id"], df["name"]))
    role_of = dict(zip(df["player_id"], df["role"]))
    out = []
    for (pid, dim), items in per.items():
        items.sort(key=lambda t: abs(t[1]), reverse=True)
        drivers = "; ".join(f"{cat} x{cnt} ({total:+.1f})" for cat, total, cnt in items[:top_n])
        out.append({"player_id": pid, "name": name_of.get(pid), "role": role_of.get(pid),
                    "dimension": dim, "top_drivers": drivers})
    return pd.DataFrame(out).sort_values(["name", "dimension"]).reset_index(drop=True)


def build_thirds(third_dim_sum, third_net, df):
    ranked = set(df["player_id"])
    name_of = dict(zip(df["player_id"], df["name"]))
    role_of = dict(zip(df["player_id"], df["role"]))
    THIRDS = ["defensive", "middle", "attacking"]
    rows = []
    for pid in ranked:
        vals = {t: third_net.get((pid, t), 0.0) for t in THIRDS}
        pos_total = sum(v for v in vals.values() if v > 0) or 1.0
        row = {"player_id": pid, "name": name_of.get(pid), "role": role_of.get(pid)}
        for t in THIRDS:
            row[t] = round(vals[t], 1)
            row[t + "_pct"] = round(100.0 * max(vals[t], 0.0) / pos_total, 0)
        rows.append(row)
    thirds_df = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)

    det = []
    for (pid, dim, t), total in third_dim_sum.items():
        if pid in ranked:
            det.append({"player_id": pid, "name": name_of.get(pid), "role": role_of.get(pid),
                        "dimension": dim, "third": t, "contribution": round(total, 1)})
    detail_df = pd.DataFrame(det).sort_values(["name", "dimension", "third"]).reset_index(drop=True)
    return thirds_df, detail_df


# =====================================================================
#  MAIN
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading matches ...")
    matches = load_matches(DATA_DIR)
    reg = build_player_registry(matches)
    print(f"  {len(matches)} matches, {len(reg)} players")

    print("Loading minutes (real substitution data) ...")
    minutes = load_minutes(RAW_MATCHES_ZIP, RAW_MATCHES_NAME)

    print("Scoring events ...")
    dim_sum, cat_sum, cat_cnt, third_dim_sum, third_net, beh = aggregate(matches, reg)

    print("Building ratings ...")
    df = build_ratings(dim_sum, reg, minutes)
    bdf = build_behaviour(beh, df)
    exp = build_explanations(cat_sum, cat_cnt, df)
    thirds_df, thirds_detail = build_thirds(third_dim_sum, third_net, df)

    ratings_cols = (["rank", "player_id", "name", "role", "minutes", "confidence", "composite"]
                    + ["score_" + d for d in DIMENSIONS])
    df_out = df[ratings_cols].copy()
    num_cols = ["minutes", "composite"] + ["score_" + d for d in DIMENSIONS]
    df_out[num_cols] = df_out[num_cols].round(1)

    paths = {
        "ratings":  os.path.join(OUTPUT_DIR, "player_ratings.csv"),
        "behaviour":os.path.join(OUTPUT_DIR, "player_behaviour.csv"),
        "exp":      os.path.join(OUTPUT_DIR, "player_explanations.csv"),
        "thirds":   os.path.join(OUTPUT_DIR, "player_thirds.csv"),
        "thirds_d": os.path.join(OUTPUT_DIR, "player_thirds_detail.csv"),
    }
    df_out.to_csv(paths["ratings"], index=False)
    bdf.to_csv(paths["behaviour"], index=False)
    exp.to_csv(paths["exp"], index=False)
    thirds_df.to_csv(paths["thirds"], index=False)
    thirds_detail.to_csv(paths["thirds_d"], index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    print("\n=== TOP 15 BY COMPOSITE (role-relative) ===")
    print(df_out.head(15).to_string(index=False))

    print("\n=== TOP 5 PER ROLE ===")
    for role in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
        sub = df_out[df_out["role"] == role].head(5)
        if not sub.empty:
            print(f"\n-- {role} --")
            print(sub[["rank", "name", "minutes", "confidence", "composite"]].to_string(index=False))

    mod = df[df["name"].str.contains("Modri", case=False, na=False)]
    if not mod.empty:
        r = mod.iloc[0]
        print("\n=== MODRIC SANITY CHECK ===")
        print(f"  composite rank #{int(r['rank'])} of {len(df)}  (composite {r['composite']:.1f}, {r['confidence']})")
        print(f"  progression score : {r['score_progression']:.1f} (percentile among midfielders)")
        b = bdf[bdf["player_id"] == int(r["player_id"])]
        if not b.empty:
            b = b.iloc[0]
            print(f"  behaviour: discipline {b['discipline']:.0f} | aggression {b['aggression']:.0f} "
                  f"| risk {b['risk_appetite']:.0f} | reliability {b['reliability']:.0f}")

    print("\nWrote:")
    for p in paths.values():
        print("  " + p)


if __name__ == "__main__":
    main()