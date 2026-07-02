"""SkillCorner Open Data loader (SPEC S5: develop pattern engine before CV works).

SkillCorner is *broadcast* tracking (single/multi camera, players extrapolated
by their vision pipeline when out of shot). Each tracking row carries an
`is_detected` flag per player and for the ball. We only keep players/ball
positions where `is_detected` is true, so the resulting Frame stream has the
same "sparse, visibility-limited" character our own broadcast CV pipeline
will eventually produce (SPEC C9 / pattern engine note in the task brief) —
do NOT fall back to the extrapolated positions.

Expected match_dir layout (see scripts/get_skillcorner.sh):
    match_dir/match_data.json      -- match/roster/team metadata
    match_dir/structured_data.jsonl -- one JSON object per frame, 10 Hz:
        {"frame": int, "timestamp": "HH:MM:SS.ff"|null, "period": int|null,
         "ball_data": {"x","y","z","is_detected"},
         "player_data": [{"x","y","player_id","is_detected"}, ...]}

Coordinates: SkillCorner gives meters with origin at the pitch CENTER
(x in [-length/2, length/2], y in [-width/2, width/2], length/width taken
from match_data.json's "pitch_length"/"pitch_width", NOT hardcoded, since
real stadium pitches vary a couple meters from the canonical 105x68). We
shift to our bottom-left-origin convention: x' = x + length/2, y' = y + width/2.
No rescale is applied (a 104m pitch stays 104m) -- values land inside or very
close to the canonical 0..105 / 0..68 box used elsewhere in bto.

attacking direction: match_data.json's "home_team_side" is a list indexed by
period (0-based) with values "left_to_right" | "right_to_left", describing
which way the HOME team is attacking that period. We map that directly:
"left_to_right" -> home attacks toward x=105 (+1), "right_to_left" -> -1.
Away team is always the opposite sign (SPEC: two teams share the same pitch,
one attacks each way). If home_team_side is missing/short, we fall back to a
GK heuristic: at kickoff (first tracked frame of the period), find each
team's deepest (min/max x) player and assume they're the GK; home attacks
away from its own GK's side.

Time: SkillCorner timestamps reset to 00:00:00 at the start of each period,
so we build a per-period offset from match_data.json's
match_periods[].duration_minutes and add the in-period timestamp to it,
giving a monotonically increasing t in seconds from kickoff.
"""

import json
from pathlib import Path

from bto.core import Frame, PlayerPos

FPS = 10.0  # SkillCorner Open Data tracking rate


def _parse_timestamp(ts: str) -> float:
    # "HH:MM:SS.ff" -> seconds
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _period_offsets(match: dict) -> dict[int, float]:
    offsets: dict[int, float] = {}
    running = 0.0
    for p in match.get("match_periods", []):
        offsets[p["period"]] = running
        running += p["duration_minutes"] * 60.0
    return offsets


def _attacking_by_period(match: dict) -> dict[int, dict[str, int]]:
    sides = match.get("home_team_side") or []
    out: dict[int, dict[str, int]] = {}
    for i, side in enumerate(sides, start=1):
        home_dir = 1 if side == "left_to_right" else -1
        out[i] = {"home": home_dir, "away": -home_dir}
    return out


def _gk_heuristic_attacking(rows: list[dict], team_of: dict[int, str], pitch_length: float) -> dict[int, dict[str, int]]:
    """Fallback: infer attacking direction per period from the deepest player
    (proxy for the GK) at kickoff. The team whose deepest player sits near
    x=-length/2 (raw, center-origin) is attacking toward +x (home-relative)."""
    by_period: dict[int, list[dict]] = {}
    for row in rows:
        p = row.get("period")
        if p is None:
            continue
        by_period.setdefault(p, []).append(row)

    out: dict[int, dict[str, int]] = {}
    for period, prows in by_period.items():
        first = next((r for r in prows if r["player_data"]), None)
        if first is None:
            out[period] = {"home": 1, "away": -1}
            continue
        deepest_x: dict[str, float] = {}
        for pd in first["player_data"]:
            team = team_of.get(pd["player_id"])
            if team is None:
                continue
            x = pd["x"]
            if team not in deepest_x or abs(x) > abs(deepest_x[team]):
                deepest_x[team] = x
        home_deep = deepest_x.get("home", -pitch_length / 2)
        home_dir = 1 if home_deep < 0 else -1
        out[period] = {"home": home_dir, "away": -home_dir}
    return out


def load_skillcorner(match_dir: str) -> list[Frame]:
    """Load one SkillCorner Open Data match directory into a list[Frame].

    Only rows with a known period are kept (pre-/post-match filler frames
    where period is null are dropped). Players/ball are included only when
    `is_detected` is true for that frame -- this matches the sparse,
    visibility-limited character of our eventual broadcast CV output.
    """
    match_dir = Path(match_dir)
    match = json.loads((match_dir / "match_data.json").read_text())

    home_id = match["home_team"]["id"]
    away_id = match["away_team"]["id"]
    team_of: dict[int, str] = {}
    for pl in match["players"]:
        if pl["team_id"] == home_id:
            team_of[pl["id"]] = "home"
        elif pl["team_id"] == away_id:
            team_of[pl["id"]] = "away"
        # else: skip (unknown team, e.g. referee) — not expected in player_data

    pitch_length = float(match.get("pitch_length") or 105.0)
    pitch_width = float(match.get("pitch_width") or 68.0)
    half_l, half_w = pitch_length / 2.0, pitch_width / 2.0

    offsets = _period_offsets(match)

    rows: list[dict] = []
    with open(match_dir / "structured_data.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            if row["period"] is not None:
                rows.append(row)

    attacking_by_period = _attacking_by_period(match)
    if not attacking_by_period:
        attacking_by_period = _gk_heuristic_attacking(rows, team_of, pitch_length)

    frames: list[Frame] = []
    for row in rows:
        period = row["period"]
        t = offsets.get(period, 0.0) + _parse_timestamp(row["timestamp"])

        players = []
        for pd in row["player_data"]:
            if not pd["is_detected"]:
                continue
            team = team_of.get(pd["player_id"])
            if team is None:
                continue
            players.append(
                PlayerPos(
                    track_id=str(pd["player_id"]),
                    team=team,
                    x=pd["x"] + half_l,
                    y=pd["y"] + half_w,
                )
            )

        bd = row["ball_data"]
        ball = (bd["x"] + half_l, bd["y"] + half_w) if bd.get("is_detected") else None

        attacking = attacking_by_period.get(period, {"home": 1, "away": -1})

        frames.append(Frame(t=t, players=players, ball=ball, attacking=attacking))

    return frames


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        match_dir = sys.argv[1]
    else:
        match_dir = str(Path(__file__).resolve().parents[2] / "data" / "skillcorner" / "1886347")

    if Path(match_dir).exists():
        frames = load_skillcorner(match_dir)
        assert len(frames) > 0, "no frames loaded"

        n_players = [len(f.players) for f in frames]
        n_with_ball = sum(1 for f in frames if f.ball is not None)
        mean_players = sum(n_players) / len(n_players)

        # bounds: allow a couple meters of slack for stadium-specific pitch size
        for f in frames:
            for p in f.players:
                assert -3.0 <= p.x <= 110.0, f"x out of bounds: {p.x}"
                assert -3.0 <= p.y <= 73.0, f"y out of bounds: {p.y}"
            if f.ball is not None:
                bx, by = f.ball
                assert -5.0 <= bx <= 112.0, f"ball x out of bounds: {bx}"
                assert -5.0 <= by <= 75.0, f"ball y out of bounds: {by}"
            assert set(f.attacking) == {"home", "away"}
            assert f.attacking["home"] == -f.attacking["away"]

        # t should be monotonically non-decreasing
        for a, b in zip(frames, frames[1:]):
            assert b.t >= a.t

        print(f"loaded {len(frames)} frames from {match_dir}")
        print(f"mean players/frame: {mean_players:.2f} (min={min(n_players)}, max={max(n_players)})")
        print(f"frames with detected ball: {n_with_ball} ({100 * n_with_ball / len(frames):.1f}%)")
        print("REAL DATA self-check passed.")
    else:
        print(f"{match_dir} not found; running synthetic-fixture self-check instead")

    # --- synthetic fixture self-check (always runs) -------------------------
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        match_json = {
            "home_team": {"id": 1},
            "away_team": {"id": 2},
            "players": [
                {"id": 100, "team_id": 1},
                {"id": 200, "team_id": 2},
            ],
            "pitch_length": 105.0,
            "pitch_width": 68.0,
            "match_periods": [
                {"period": 1, "duration_minutes": 45.0},
                {"period": 2, "duration_minutes": 45.0},
            ],
            "home_team_side": ["left_to_right", "right_to_left"],
        }
        (tdp / "match_data.json").write_text(json.dumps(match_json))

        rows = [
            {
                "frame": 0,
                "timestamp": "00:00:00.00",
                "period": 1,
                "ball_data": {"x": 0.0, "y": 0.0, "is_detected": True},
                "player_data": [
                    {"x": -10.0, "y": 5.0, "player_id": 100, "is_detected": True},
                    {"x": 10.0, "y": -5.0, "player_id": 200, "is_detected": False},  # dropped
                ],
            },
            {
                "frame": 1,
                "timestamp": "00:00:00.10",
                "period": 1,
                "ball_data": {"x": None, "y": None, "is_detected": False},
                "player_data": [
                    {"x": -9.0, "y": 5.0, "player_id": 100, "is_detected": True},
                    {"x": 9.0, "y": -5.0, "player_id": 200, "is_detected": True},
                ],
            },
            {
                "frame": 2,
                "timestamp": "00:00:01.00",
                "period": 2,
                "ball_data": {"x": 1.0, "y": 1.0, "is_detected": True},
                "player_data": [
                    {"x": 0.0, "y": 0.0, "player_id": 100, "is_detected": True},
                ],
            },
        ]
        with open(tdp / "structured_data.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        frames = load_skillcorner(str(tdp))
        assert len(frames) == 3

        f0 = frames[0]
        assert len(f0.players) == 1  # player 200 dropped (not detected)
        p0 = f0.players[0]
        assert p0.track_id == "100" and p0.team == "home"
        assert p0.x == -10.0 + 52.5 and p0.y == 5.0 + 34.0
        assert f0.ball == (52.5, 34.0)
        assert f0.attacking == {"home": 1, "away": -1}

        f1 = frames[1]
        assert f1.ball is None  # ball not detected -> None
        assert len(f1.players) == 2
        assert f1.t == 0.10  # still period 1, offset 0

        f2 = frames[2]
        assert f2.attacking == {"home": -1, "away": 1}  # period 2 flips
        assert f2.t == 45.0 * 60.0 + 1.0  # period-2 offset + in-period timestamp

        print("synthetic-fixture self-check passed.")
