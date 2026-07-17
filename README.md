# Football Ratings Engine

A methodology and working solution for rating every player and every coach at the FIFA World Cup 2018, built on the Wyscout event dataset.

Two engines. Two philosophies. One shared commitment: explain first, rate later.

- **Player engine** — 8-dimension profile + role-aware composite for 736 players.
- **Coach engine** — 5-dimension profile + equal-weighted composite for 32 head coaches.

The methodology document (`methodology.pdf`) is the primary read. This README covers **how to run** the engines and **how to read** the outputs.

---

## Setup

### 1. Python

Python 3.10+ recommended. The engines use standard scientific Python — nothing exotic.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the data

The engines run on the Wyscout FIFA World Cup 2018 dataset:

- Repo: https://github.com/koenvo/wyscout-soccer-match-event-dataset
- Files needed: the 64 processed WC event JSONs, plus the raw `matches.zip` (used by the player engine for real substitution-based minutes).

Set up the local data folder as shown below.

---

## Folder structure

```
playernation-ratings/
├── README.md
├── methodology.pdf
├── requirements.txt
├── src/
│   ├── player_engine.py
│   └── coach_engine.py
├── data/
│   ├── world_cup/           # 64 processed match JSONs
│   │   ├── match_0.json
│   │   └── ...
│   ├── matches.zip          # raw match metadata (for player minutes)
│   └── coach_mapping.json   # static 32-coach mapping
└── output/                  # engines write here
    ├── player_ratings.csv
    ├── player_behaviour.csv
    ├── player_explanations.csv
    ├── player_thirds.csv
    ├── player_thirds_detail.csv
    ├── coach_ratings.csv
    ├── coach_style.csv
    ├── coach_explanations.csv
    └── coach_matches.csv
```

`coach_mapping.json` maps team IDs to head coach names, Wikipedia-sourced for the 32 WC 2018 teams (Spain = Fernando Hierro, not Lopetegui — sacked two days pre-tournament).

---

## How to run

From the project root:

```bash
# Player engine
python src/player_engine.py

# Coach engine
python src/coach_engine.py
```

Each engine writes its CSVs into `output/`. No arguments required. If your engines expect a different working directory or data path, adjust either the folder layout or the engine's internal paths to match.

---

## Output schema

> The column names below reflect the engines' output. If they don't match your local CSVs exactly, treat the CSV as the source of truth.

### Player engine

#### `player_ratings.csv`
The headline table. One row per player.

| Column | Type | Meaning |
|---|---|---|
| rank | int | Overall rank by composite. |
| player_id | int | Wyscout player ID. |
| name | str | Player name. |
| role | str | Goalkeeper / Defender / Midfielder / Forward. |
| minutes | float | Real minutes played (from lineup + substitutions in `matches.zip`). |
| confidence | str | High (≥450 min) / Medium (180–450) / Low (<180). Low = indicative. |
| composite | float 0–100 | Mean percentile across the dimensions expected for the player's role. |
| score_ball_recovery | float 0–100 | Percentile within role. |
| score_ball_retention | float 0–100 | Percentile within role. |
| score_ball_progression | float 0–100 | Percentile within role. |
| score_chance_creation | float 0–100 | Percentile within role. |
| score_finishing | float 0–100 | Percentile within role. |
| score_defensive_disruption | float 0–100 | Percentile within role. |
| score_goal_protection | float 0–100 | Percentile within role. |
| score_transition | float 0–100 | Percentile within role. |

**Role → composite inputs** (only these dimensions feed the headline composite; all 8 are always shown for profile completeness):

| Role | Composite dimensions |
|---|---|
| Goalkeeper | Goal Protection · Retention · Transition |
| Defender | Recovery · Defensive Disruption · Retention · Progression |
| Midfielder | Retention · Progression · Recovery · Chance Creation · Transition |
| Forward | Finishing · Chance Creation · Progression · Retention |

#### `player_behaviour.csv`
Descriptive style traits. Never enter the composite.

| Column | Type | Meaning |
|---|---|---|
| player_id | int | |
| name | str | |
| role | str | |
| discipline | float 0–100 | Percentile within role. Inverse of fouls + cards + simulations per 90. |
| aggression | float 0–100 | Percentile within role. Duels + tackles + fouls per 90. A descriptor, not a judgment. |
| risk_appetite | float 0–100 | Percentile within role. Ambitious-pass share of all passes. |
| reliability | float 0–100 | Percentile within role. Pass completion penalised for dangerous losses. |

#### `player_explanations.csv`
Top action drivers per player per dimension. Answers *why* a score is what it is.

| Column | Type | Meaning |
|---|---|---|
| player_id | int | |
| name | str | |
| dimension | str | e.g. `ball_recovery`, `progression`. |
| action_category | str | e.g. `interception`, `pass_into_box`. |
| count | int | Number of events in this category. |
| contribution | float | Raw contribution units (weight × count). Explanation only — see the three-scales note below. |

#### `player_thirds.csv`
Where a player operates. Share of positive value across the three pitch thirds.

| Column | Type | Meaning |
|---|---|---|
| player_id | int | |
| name | str | |
| defensive_pct | float 0–100 | Share of positive value in the defensive third. |
| middle_pct | float 0–100 | Share in the middle third. |
| attacking_pct | float 0–100 | Share in the attacking third. |

The three `_pct` columns sum to ~100.

#### `player_thirds_detail.csv`
Same idea, broken down by dimension so you can see *what* they did in each third.

| Column | Type | Meaning |
|---|---|---|
| player_id | int | |
| name | str | |
| dimension | str | |
| defensive | float | Raw contribution units in the defensive third. |
| middle | float | Raw contribution units in the middle third. |
| attacking | float | Raw contribution units in the attacking third. |

### Coach engine

#### `coach_ratings.csv`
The headline coach table. One row per team.

| Column | Type | Meaning |
|---|---|---|
| rank | int | Rank by composite. |
| team_id | int | Wyscout team ID. |
| coach | str | Head coach name (from `coach_mapping.json`). |
| team | str | Team name. |
| matches | int | Matches played at the tournament. |
| confidence | str | High (≥6 — SF/finalists) / Medium (4–5 — R16 exits) / Low (=3 — group exits). |
| composite | float 0–100 | Equal-weighted mean of the 5 dimension scores. |
| score_build_up_quality | float 0–100 | Percentile within the 32-team field. |
| score_chance_creation | float 0–100 | Percentile within the field. |
| score_set_pieces | float 0–100 | Percentile within the field. |
| score_defensive_organisation | float 0–100 | Percentile within the field. |
| score_pressing_structure | float 0–100 | Percentile within the field. |

#### `coach_style.csv`
Descriptive style traits for each team. Never enter the composite.

| Column | Type | Meaning |
|---|---|---|
| team | str | |
| possession_orientation | float 0–100 | Percentile. Team passes ÷ all passes. |
| block_height | float 0–100 | Percentile. Mean x-coordinate of defensive actions — higher = higher line. |
| attack_tempo | float 0–100 | Percentile. Passes per possession sequence — low = direct, high = patient. |
| discipline | float 0–100 | Percentile. Fewer fouls per match = higher score. |

#### `coach_explanations.csv`
Top action drivers per coach per dimension.

| Column | Type | Meaning |
|---|---|---|
| team | str | |
| dimension | str | e.g. `build_up_quality`, `pressing_structure`. |
| action_category | str | e.g. `final_third_entry`, `interception_opp_half`. |
| count | int | Number of events. |
| contribution | float | Raw contribution units — explanation only. |

#### `coach_matches.csv`
Per-match contribution per coach per dimension. The match-level audit trail — lets you verify any tournament rating by tracing it back to individual games.

| Column | Type | Meaning |
|---|---|---|
| team | str | |
| match_id | int/str | Wyscout match ID. |
| opponent | str | |
| dimension | str | |
| contribution | float | Raw contribution units for that match/dimension. |

---

## Reading the outputs — three number scales

The engines produce numbers on three different scales. **Do not conflate them.** This is the single most important reading rule.

### 1. Raw contribution units

The `contribution` columns in `player_explanations.csv`, `player_thirds_detail.csv`, `coach_explanations.csv`, and `coach_matches.csv`.

`weight × count`. Unbounded, internal units. Example: an interception with weight 3.0, seen 9 times → contribution = +27.0.

**Use them to explain, never to rate.** Cite the count ("9 interceptions"), never the raw contribution as a quality claim.

### 2. Share %

The `_pct` columns in `player_thirds.csv`.

Values 0–100, summing to 100 across the three thirds. Answers *where* a player operates, not *how well*.

**Not a quality metric.** A striker with 90% attacking-third share isn't "better" than a defender with 90% defensive-third share — they're just doing different jobs.

### 3. Percentiles

All `score_*` columns, `composite`, and every column in `player_behaviour.csv` and `coach_style.csv`.

Values 0–100. **The only numbers that receive quality/level language, and always phrased role-relative (players) or field-relative (coaches).**

A player's `score_ball_progression = 78` means "top-quintile progression for their role at WC 2018." It does *not* mean 78/100 skill on some absolute scale.

---

## Worked example — reading a row

A Forward, 224 minutes played:

```
composite                     56
score_ball_retention          89
score_ball_recovery           68
score_ball_progression        68
score_defensive_disruption    63
score_chance_creation         20
score_transition              30
score_finishing               38
score_goal_protection         27
confidence                    Medium
```

How to read this:

- **Composite 56** — mean of the Forward-expected dimensions (Finishing, Chance Creation, Progression, Retention). Goal Protection 27 is in the profile but excluded from the composite because it's not a Forward-role expectation.
- **Retention 89** — top-decile for Forwards. Ball-safe under pressure.
- **Chance Creation 20 · Finishing 38** — below-median. Not a goalscorer, doesn't manufacture chances at a Forward's expected level.
- **Recovery 68 · Progression 68 · Defensive Disruption 63** — above median. Works hard off the ball, progresses it when he has it.
- **Confidence Medium** — 224 min. Score is directional but not stable.

Read together: *a hard-working, high-pressing link forward, not a goalscorer.* The composite alone (56) doesn't tell you that. The profile does.

This is the point of the two-layer design: composite for engagement, profile for understanding.

---

## Methodology

The full philosophy — dimension design, tradeoff decisions, evolution path, known limitations — lives in `methodology.pdf`.

Key headlines:

- **Only observable evidence.** No modelling of intent, luck, or hidden tactical thought.
- **Explain first, rate later.** Every score retains its top action drivers.
- **Style ≠ ability.** Behaviour and coach style are profile-only, never in the composite.
- **Quality, not achievement (coach engine).** France (champions) correctly ranks #9 on performance; Germany (group exit) correctly ranks #3. The engine refuses to reward results.
- **Field-relative, honestly.** A 78 means "top-quintile at WC 2018," not "objectively good football." Absolute anchoring needs cross-tournament data — a v2 unlock.

Known limitations are named upfront in the methodology, not hidden. Two worth flagging here:

- **Defensive Organisation** in the coach engine conflates defensive volume with defensive quality — teams pinned back score highly. v1.1 fix: normalise per opponent possession.
- **Transition** in the player engine is thin — sequence-based transition (recover-and-immediately-progress) is a v2 unlock.

---

## Limitations summary

- No age/level fairness — WC is elite-adult only.
- No context modifiers (scoreline, minute, tournament stage don't affect action value in v1).
- Coach engine has no substitution or formation signal (data limit).
- Field-relative percentiles, not absolute scores.

Full list in `methodology.pdf`.
