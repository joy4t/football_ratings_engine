"""
PlayerNation - Coach Ratings Engine (v1)
========================================
5 contribution dimensions -> per-match rate -> percentile within the 32-team
field -> EQUAL-WEIGHTED composite.
Plus: a profile-only STYLE layer and a CONFIDENCE band per coach.

Philosophy (locked):
    The engine measures how well a team executed the behaviours most reasonably
    attributed to COACHING - buildup, chance creation, set pieces, defensive
    organisation, pressing - ranked against the tournament field. It does NOT
    claim to measure the coach's causal influence (the data can't isolate coach
    from players). It measures observed, coaching-attributable performance.

Pipeline (per dimension, per team):
    event -> quality-weighted contribution -> per-match rate
          -> percentile within the field -> score 0-100
    composite = mean of the 5 dimension scores (EQUAL weights)
    style     = descriptors on the same events (profile-only, never in composite)
    confidence= matches-played band (High >= 6, Medium 4-5, Low 3)

Attribution principle enforced in code:
    - Open-play GOALS are treated as player finishing -> they do NOT inflate a
      coach's chance_creation. Only the CHANCE (shot location) is credited.
    - Only SET-PIECE goals feed a coach dimension (routines are coach-shaped).
    - Penalties are excluded (player finishing / earned).

RUN (Anaconda Prompt, from PROJECT ROOT):
    conda activate base
    cd /d D:\\DS_Projects\\Playernation
    python src\\coach_engine.py

INPUTS (already on disk):
    data\\world_cup\\*.json
    data\\coach_mapping.json

OUTPUTS (output\\):
    coach_ratings.csv        5 dim scores + equal-weighted composite + confidence
    coach_style.csv          4 style descriptors (profile-only)
    coach_explanations.csv   top action drivers per coach per dimension
    coach_matches.csv        raw per-match contribution per coach per dimension

Every number in CONFIG is a PRODUCT DECISION you own and can defend.
"""

import json
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# =====================================================================
#  CONFIG  -  YOUR DECISIONS. Tune and defend every number here.
# =====================================================================

# Run from PROJECT ROOT (paths relative to it), even though this file is in src\.
DATA_DIR      = os.path.join("data", "world_cup")
COACH_MAPPING = os.path.join("data", "coach_mapping.json")
OUTPUT_DIR    = "output"

# Confidence bands (matches played). SF/final reach 7; R16 exits 4; group exits 3.
CONF_HIGH_MATCHES = 6
CONF_MED_MATCHES  = 4

# The 5 coach dimensions. EQUAL weight in the composite (each = 1/5).
DIMENSIONS = [
    "buildup_quality",
    "chance_creation",
    "set_pieces",
    "defensive_organisation",
    "pressing_structure",
]

# Style descriptors (profile-only - NEVER enter the composite).
STYLE = ["possession_orientation", "block_height", "attack_tempo", "discipline"]

# Set-piece -> shot linkage window (seconds). "Did this corner lead to a shot?"
SET_PIECE_WINDOW_SEC = 10.0

# Pitch geometry. Wyscout coords are normalized to the ACTING team's attack:
#   x = 0 own goal line, x = 100 opponent goal line; y = 0..100 across.
FINAL_THIRD_X = 66
MID_LINE_X    = 50
OWN_THIRD_X   = 33
BOX_X         = 84
BOX_Y_LOW     = 19
BOX_Y_HIGH    = 81
OWN_BOX_X     = 16

# ---- Wyscout tag IDs (same dictionary as the player engine) ----
T_GOAL = 101
T_ASSIST = 301; T_KEY_PASS = 302
T_DUEL_LOST = 701; T_DUEL_WON = 703
T_THROUGH = 901; T_INTERCEPTION = 1401
T_ACCURATE = 1801; T_INACCURATE = 1802

# ---- CONTRIBUTION WEIGHT TABLES (within-dimension; how actions rank vs each
#      other INSIDE one dimension). The COMPOSITE across dimensions is equal. ----
W = {
    "buildup_quality": {
        "final_third_entry":  2.0,   # accurate pass crossing x=66
        "pass_into_box":      3.0,   # accurate pass ending in opp box
        "line_breaking":      2.5,   # smart pass / through-ball
        "progressive_carry":  1.5,   # forward carry into final third
        "turnover_own_third": -2.0,  # lost ball deep in own build-up area
    },
    "chance_creation": {
        "shot_central_box":   3.0,   # shot from central box (best chance)
        "shot_wide_box":      1.5,   # shot from wide box
        "shot_outside_box":   0.5,   # shot from distance
        "assist":             4.0,
        "key_pass":           2.0,
        "cross_into_box":     1.0,   # accurate cross into box
    },
    "set_pieces": {
        "corner_taken":            0.3,  # volume (small)
        "corner_leading_to_shot":  2.0,  # corner -> shot within window
        "free_kick_shot":          1.5,  # direct free-kick shot
        "free_kick_cross_in_box":  1.0,  # accurate FK cross into box
        "set_piece_goal":          5.0,  # goal within a set-piece window
    },
    "defensive_organisation": {
        "clearance_from_box":     1.0,
        "interception_own_half":  1.5,
        "def_duel_won_own_third": 1.0,
        "shot_conceded":          -2.0,  # opponent shot (from open play or box)
        "shot_conceded_from_box": -3.5,  # opponent shot from inside the box
    },
    "pressing_structure": {
        "interception_opp_half":  2.0,
        "interception_opp_third": 3.0,   # won even higher up
        "duel_won_opp_half":      1.0,
        "duel_won_opp_third":     1.5,
    },
}


# =====================================================================
#  LOAD DATA
# =====================================================================

def load_matches(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(
            f"No match files in {data_dir!r}. Run from PROJECT ROOT "
            f"(D:\\DS_Projects\\Playernation), e.g.  python src\\coach_engine.py"
        )
    return [json.load(open(f, encoding="utf-8")) for f in files]


def load_coach_mapping(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Coach mapping not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(tid): info for tid, info in raw.items()}


def build_team_registry(matches):
    reg = {}
    for m in matches:
        for team_id, team in m["teams"].items():
            tid = str(team_id)
            if tid not in reg:
                reg[tid] = {"name": team.get("name")}
    return reg


# =====================================================================
#  PER-EVENT HELPERS
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


def shot_zone(x, y):
    if in_box(x, y):
        return "shot_central_box" if abs(y - 50) <= 15 else "shot_wide_box"
    return "shot_outside_box"


# =====================================================================
#  AGGREGATE  (single ordered walk per match; scores both teams)
# =====================================================================

def new_agg():
    return {
        "dim":       defaultdict(float),   # (team, dim) -> total contribution
        "cat":       defaultdict(float),   # (team, dim, cat) -> total
        "catcnt":    defaultdict(int),     # (team, dim, cat) -> count
        "match_dim": defaultdict(float),   # (match_id, team, dim) -> total
        "matches":   defaultdict(set),     # team -> set of match_ids
        # style tallies
        "passes":    defaultdict(int),     # team -> total passes
        "fouls":     defaultdict(int),     # team -> total fouls
        "def_x_sum": defaultdict(float),   # team -> sum of def-action x (block height)
        "def_x_cnt": defaultdict(int),
        "poss_list": defaultdict(list),    # team -> per-match possession share
        "tempo_list":defaultdict(list),    # team -> per-match passes-per-sequence
    }


def process_match(m, agg):
    match_id = str(m.get("wyId") or m.get("matchId") or id(m))
    team_ids = [str(t) for t in m["teams"].keys()]
    opp = {}
    if len(team_ids) == 2:
        opp[team_ids[0]] = team_ids[1]
        opp[team_ids[1]] = team_ids[0]

    for tid in team_ids:
        agg["matches"][tid].add(match_id)

    # ordered walk (Wyscout files are time-ordered; sort defensively)
    events = sorted(
        m["events"],
        key=lambda e: (e.get("matchPeriod", ""), e.get("eventSec", 0.0)),
    )

    # per-match local state
    m_pass = defaultdict(int)          # possession share
    m_seq  = defaultdict(int)          # sequence count (tempo)
    cur_team = None                    # possession tracker
    last_sp = {}                       # team -> (period, sec, kind)  kind: 'corner'|'fk'

    def add(team, dim, cat):
        val = W[dim][cat]
        agg["dim"][(team, dim)] += val
        agg["cat"][(team, dim, cat)] += val
        agg["catcnt"][(team, dim, cat)] += 1
        agg["match_dim"][(match_id, team, dim)] += val

    for ev in events:
        tid = str(ev["teamId"])
        name = ev["eventName"]
        sub  = ev.get("subEventName", "")
        tags = tags_of(ev)
        x0, y0, x1, y1 = coords(ev)
        acc = T_ACCURATE in tags
        period = ev.get("matchPeriod", "")
        sec = ev.get("eventSec", 0.0)

        # ---- possession sequence tracking (on-ball events only) ----
        if name in ("Pass", "Shot", "Free Kick", "Others on the ball"):
            if tid != cur_team:
                m_seq[tid] += 1
                cur_team = tid

        # =============================================================
        #  PASS
        # =============================================================
        if name == "Pass":
            m_pass[tid] += 1
            agg["passes"][tid] += 1
            if acc:
                if x0 is not None and x1 is not None and x0 < FINAL_THIRD_X <= x1:
                    add(tid, "buildup_quality", "final_third_entry")
                if in_box(x1, y1) and not in_box(x0, y0):
                    add(tid, "buildup_quality", "pass_into_box")
                    if sub == "Cross":
                        add(tid, "chance_creation", "cross_into_box")
                if sub == "Smart pass" or T_THROUGH in tags:
                    add(tid, "buildup_quality", "line_breaking")
            else:
                if x0 is not None and x0 < OWN_THIRD_X:
                    add(tid, "buildup_quality", "turnover_own_third")
            if T_ASSIST in tags:
                add(tid, "chance_creation", "assist")
            if T_KEY_PASS in tags:
                add(tid, "chance_creation", "key_pass")

        # =============================================================
        #  SHOT
        # =============================================================
        elif name == "Shot":
            # chance for the shooting team (location only; goals = finishing, excluded)
            zone = shot_zone(x0, y0)
            add(tid, "chance_creation", zone)
            # shot conceded by the opponent (defensive organisation)
            if tid in opp:
                o = opp[tid]
                cat = "shot_conceded_from_box" if in_box(x0, y0) else "shot_conceded"
                add(o, "defensive_organisation", cat)
            # set-piece goal? (goal within window of this team's last set piece)
            if T_GOAL in tags and tid in last_sp:
                sp_period, sp_sec, _kind = last_sp[tid]
                if period == sp_period and (sec - sp_sec) <= SET_PIECE_WINDOW_SEC:
                    add(tid, "set_pieces", "set_piece_goal")
            # corner -> shot within window
            if tid in last_sp:
                sp_period, sp_sec, kind = last_sp[tid]
                if kind == "corner" and period == sp_period and (sec - sp_sec) <= SET_PIECE_WINDOW_SEC:
                    add(tid, "set_pieces", "corner_leading_to_shot")

        # =============================================================
        #  FREE KICK family (set pieces)
        # =============================================================
        elif name == "Free Kick":
            if sub == "Corner":
                add(tid, "set_pieces", "corner_taken")
                last_sp[tid] = (period, sec, "corner")
            elif sub == "Free kick shot":
                add(tid, "set_pieces", "free_kick_shot")
                last_sp[tid] = (period, sec, "fk")
                if T_GOAL in tags:
                    add(tid, "set_pieces", "set_piece_goal")
            elif sub == "Free kick cross":
                if acc and in_box(x1, y1):
                    add(tid, "set_pieces", "free_kick_cross_in_box")
                last_sp[tid] = (period, sec, "fk")
            # "Throw in", "Goal kick", "Penalty" -> intentionally excluded

        # =============================================================
        #  OTHERS ON THE BALL  (carries, clearances)
        # =============================================================
        elif name == "Others on the ball":
            if sub == "Clearance":
                if x0 is not None and x0 <= OWN_BOX_X:
                    add(tid, "defensive_organisation", "clearance_from_box")
                # clearance counts as a defensive action for block height
                if x0 is not None:
                    agg["def_x_sum"][tid] += x0
                    agg["def_x_cnt"][tid] += 1
            elif sub == "Acceleration":
                if x0 is not None and x1 is not None and x1 > x0 and x1 >= FINAL_THIRD_X:
                    add(tid, "buildup_quality", "progressive_carry")

        # =============================================================
        #  DUEL
        # =============================================================
        elif name == "Duel":
            won = T_DUEL_WON in tags
            # pressing: duels won high up the pitch
            if won and x0 is not None:
                if x0 >= FINAL_THIRD_X:
                    add(tid, "pressing_structure", "duel_won_opp_third")
                elif x0 >= MID_LINE_X:
                    add(tid, "pressing_structure", "duel_won_opp_half")
            # defensive: ground defending duel won deep
            if sub == "Ground defending duel" and won and x0 is not None and x0 < OWN_THIRD_X:
                add(tid, "defensive_organisation", "def_duel_won_own_third")
            # defensive action for block height
            if sub == "Ground defending duel" and x0 is not None:
                agg["def_x_sum"][tid] += x0
                agg["def_x_cnt"][tid] += 1

        # =============================================================
        #  FOUL (style: discipline)
        # =============================================================
        elif name == "Foul":
            agg["fouls"][tid] += 1

        # ---- INTERCEPTION tag (can ride on several event types) ----
        if T_INTERCEPTION in tags and x0 is not None:
            if x0 >= FINAL_THIRD_X:
                add(tid, "pressing_structure", "interception_opp_third")
            elif x0 >= MID_LINE_X:
                add(tid, "pressing_structure", "interception_opp_half")
            else:
                add(tid, "defensive_organisation", "interception_own_half")
            # interception is a defensive action for block height
            agg["def_x_sum"][tid] += x0
            agg["def_x_cnt"][tid] += 1

    # ---- per-match style aggregates ----
    if len(team_ids) == 2:
        a, b = team_ids
        tot = m_pass[a] + m_pass[b]
        if tot > 0:
            agg["poss_list"][a].append(m_pass[a] / tot)
            agg["poss_list"][b].append(m_pass[b] / tot)
    for tid in team_ids:
        if m_seq[tid] > 0:
            agg["tempo_list"][tid].append(m_pass[tid] / m_seq[tid])


# =====================================================================
#  RATINGS  (per-match rate -> percentile within field -> composite)
# =====================================================================

def confidence_band(n):
    if n >= CONF_HIGH_MATCHES:
        return "High"
    if n >= CONF_MED_MATCHES:
        return "Medium"
    return "Low"


def build_ratings(agg, coach_map, team_reg):
    teams = sorted(agg["matches"].keys())
    rows = []
    for tid in teams:
        n = len(agg["matches"][tid])
        if n == 0:
            continue
        info = coach_map.get(tid, {})
        row = {
            "team_id":    tid,
            "coach":      info.get("coach", ""),
            "team":       info.get("team", team_reg.get(tid, {}).get("name", tid)),
            "matches":    n,
            "confidence": confidence_band(n),
        }
        for dim in DIMENSIONS:
            row["raw_" + dim] = agg["dim"].get((tid, dim), 0.0)
            row["per_match_" + dim] = row["raw_" + dim] / n
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No teams aggregated - check data loading.")

    # percentile within the single 32-team field (no roles for coaches)
    for dim in DIMENSIONS:
        df["score_" + dim] = df["per_match_" + dim].rank(pct=True) * 100.0

    # composite = EQUAL-weighted mean of the 5 dimension scores
    df["composite"] = df[["score_" + d for d in DIMENSIONS]].mean(axis=1)

    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


# =====================================================================
#  STYLE  (profile-only; percentile within field)
# =====================================================================

def build_style(agg, df):
    rows = []
    for _, r in df.iterrows():
        tid = r["team_id"]
        poss = np.mean(agg["poss_list"][tid]) if agg["poss_list"][tid] else np.nan
        tempo = np.mean(agg["tempo_list"][tid]) if agg["tempo_list"][tid] else np.nan
        block = (agg["def_x_sum"][tid] / agg["def_x_cnt"][tid]) if agg["def_x_cnt"][tid] else np.nan
        fouls_pm = agg["fouls"][tid] / r["matches"]
        rows.append({
            "team_id": tid, "coach": r["coach"], "team": r["team"],
            "_possession": poss,      # higher = more possession
            "_block": block,          # higher = higher defensive line
            "_tempo": tempo,          # higher = more patient (passes/sequence)
            "_fouls_pm": fouls_pm,    # higher = less disciplined
        })
    sdf = pd.DataFrame(rows)

    # percentile the raw signals within the field (profile-only)
    sdf["possession_orientation"] = sdf["_possession"].rank(pct=True) * 100.0
    sdf["block_height"]           = sdf["_block"].rank(pct=True) * 100.0
    sdf["attack_tempo"]           = sdf["_tempo"].rank(pct=True) * 100.0
    # discipline: fewer fouls = MORE disciplined = higher score (invert)
    sdf["discipline"]             = (1.0 - sdf["_fouls_pm"].rank(pct=True)) * 100.0

    cols = ["team_id", "coach", "team"] + STYLE
    out = sdf[cols].copy()
    for c in STYLE:
        out[c] = out[c].round(0)
    return out.sort_values("team").reset_index(drop=True)


# =====================================================================
#  EXPLANATIONS + PER-MATCH BREAKDOWN
# =====================================================================

def build_explanations(agg, df, top_n=3):
    ranked = set(df["team_id"])
    coach_of = dict(zip(df["team_id"], df["coach"]))
    team_of  = dict(zip(df["team_id"], df["team"]))
    per = defaultdict(list)
    for (tid, dim, cat), total in agg["cat"].items():
        if tid in ranked:
            per[(tid, dim)].append((cat, total, agg["catcnt"][(tid, dim, cat)]))
    out = []
    for (tid, dim), items in per.items():
        items.sort(key=lambda t: abs(t[1]), reverse=True)
        drivers = "; ".join(f"{cat} x{cnt} ({total:+.1f})" for cat, total, cnt in items[:top_n])
        out.append({"coach": coach_of.get(tid), "team": team_of.get(tid),
                    "dimension": dim, "top_drivers": drivers})
    return pd.DataFrame(out).sort_values(["team", "dimension"]).reset_index(drop=True)


def build_match_breakdown(agg, df):
    ranked = set(df["team_id"])
    coach_of = dict(zip(df["team_id"], df["coach"]))
    team_of  = dict(zip(df["team_id"], df["team"]))
    rows = []
    for (match_id, tid, dim), total in agg["match_dim"].items():
        if tid in ranked:
            rows.append({"match_id": match_id, "coach": coach_of.get(tid),
                         "team": team_of.get(tid), "dimension": dim,
                         "contribution": round(total, 1)})
    return pd.DataFrame(rows).sort_values(["team", "match_id", "dimension"]).reset_index(drop=True)


# =====================================================================
#  MAIN
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading matches ...")
    matches = load_matches(DATA_DIR)
    print(f"  {len(matches)} match files")

    print("Loading coach mapping ...")
    coach_map = load_coach_mapping(COACH_MAPPING)
    team_reg = build_team_registry(matches)
    print(f"  {len(coach_map)} coaches mapped, {len(team_reg)} teams in data")

    print("Scoring events (single ordered walk per match) ...")
    agg = new_agg()
    for m in matches:
        process_match(m, agg)

    print("Building ratings ...")
    df = build_ratings(agg, coach_map, team_reg)
    style = build_style(agg, df)
    exp = build_explanations(agg, df)
    mb = build_match_breakdown(agg, df)

    # ---- write outputs ----
    ratings_cols = (["rank", "team_id", "coach", "team", "matches", "confidence", "composite"]
                    + ["score_" + d for d in DIMENSIONS])
    df_out = df[ratings_cols].copy()
    num_cols = ["composite"] + ["score_" + d for d in DIMENSIONS]
    df_out[num_cols] = df_out[num_cols].round(1)

    paths = {
        "ratings": os.path.join(OUTPUT_DIR, "coach_ratings.csv"),
        "style":   os.path.join(OUTPUT_DIR, "coach_style.csv"),
        "exp":     os.path.join(OUTPUT_DIR, "coach_explanations.csv"),
        "matches": os.path.join(OUTPUT_DIR, "coach_matches.csv"),
    }
    df_out.to_csv(paths["ratings"], index=False)
    style.to_csv(paths["style"], index=False)
    exp.to_csv(paths["exp"], index=False)
    mb.to_csv(paths["matches"], index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    print("\n=== COACH RATINGS (by composite, field-relative) ===")
    print(df_out.to_string(index=False))

    print("\n=== CONFIDENCE DISTRIBUTION ===")
    print(df_out["confidence"].value_counts().to_string())

    print("\n=== STYLE PROFILE (profile-only, never in composite) ===")
    print(style.to_string(index=False))

    print("\nWrote:")
    for p in paths.values():
        print("  " + p)


if __name__ == "__main__":
    main()