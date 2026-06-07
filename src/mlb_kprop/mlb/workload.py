from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date, timedelta
from typing import Any

from mlb_kprop.mlb.starters import MlbStatsClient


@dataclass(frozen=True)
class StartSummary:
    game_date: Date
    batters_faced: int
    innings_pitched: float
    strikeouts: int


@dataclass(frozen=True)
class BullpenWorkload:
    game_date: Date | None
    relievers_used: int
    bullpen_ip: float
    bullpen_pitches: int


@dataclass(frozen=True)
class BattersFacedPrediction:
    batters_faced: int
    detail: str
    recent_avg_bf: float | None
    bullpen_adj: float
    recent_bf_std: float | None


def _parse_ip(ip_value: str | float | int) -> float:
    if isinstance(ip_value, (int, float)):
        return float(ip_value)
    text = str(ip_value).strip()
    if not text:
        return 0.0
    if "." in text:
        whole, frac = text.split(".", 1)
        return int(whole) + int(frac[:1]) / 3.0
    return float(text)


def pitcher_recent_starts(
    client: MlbStatsClient,
    pitcher_id: int,
    before_date: Date,
    count: int = 3,
) -> list[StartSummary]:
    """Last N starts strictly before before_date (current season, then prior if needed)."""
    seasons = [before_date.year, before_date.year - 1]
    starts: list[StartSummary] = []

    for season in seasons:
        payload = client._get(
            f"{client.API_BASE}/people/{pitcher_id}/stats",
            {"stats": "gameLog", "group": "pitching", "season": season},
        )
        stats = payload.get("stats") or []
        if not stats:
            continue
        for split in stats[0].get("splits") or []:
            stat = split.get("stat") or {}
            if int(stat.get("gamesStarted") or 0) != 1:
                continue
            game_date = Date.fromisoformat(str(split["date"]))
            if game_date >= before_date:
                continue
            bf = int(stat.get("battersFaced") or 0)
            if bf <= 0:
                continue
            starts.append(
                StartSummary(
                    game_date=game_date,
                    batters_faced=bf,
                    innings_pitched=_parse_ip(stat.get("inningsPitched", 0)),
                    strikeouts=int(stat.get("strikeOuts") or 0),
                )
            )
        if len(starts) >= count:
            break

    starts.sort(key=lambda s: s.game_date, reverse=True)
    return starts[:count]


def team_bullpen_workload(
    client: MlbStatsClient,
    team_id: int,
    game_date: Date,
) -> BullpenWorkload | None:
    """Sum non-starter pitching for a team on a completed game date."""
    games = client.schedule_games_on_date(game_date, team_id=team_id)
    if not games:
        return None

    total_ip = 0.0
    total_pitches = 0
    relievers = 0

    for game in games:
        if game.get("status", {}).get("abstractGameState") != "Final":
            continue
        game_pk = int(game["gamePk"])
        feed = client._get(f"{client.API_BASE}.1/game/{game_pk}/feed/live")
        side_key = "home" if int(game["teams"]["home"]["team"]["id"]) == team_id else "away"
        players = (
            feed.get("liveData", {})
            .get("boxscore", {})
            .get("teams", {})
            .get(side_key, {})
            .get("players", {})
        )
        for player in players.values():
            pitching = player.get("stats", {}).get("pitching", {})
            if not pitching or int(pitching.get("gamesPitched") or 0) == 0:
                continue
            if player.get("position", {}).get("abbreviation") == "SP":
                continue
            ip = _parse_ip(pitching.get("inningsPitched", 0))
            if ip <= 0:
                continue
            relievers += 1
            total_ip += ip
            total_pitches += int(pitching.get("numberOfPitches") or 0)

    if relievers == 0:
        return BullpenWorkload(game_date=game_date, relievers_used=0, bullpen_ip=0.0, bullpen_pitches=0)

    return BullpenWorkload(
        game_date=game_date,
        relievers_used=relievers,
        bullpen_ip=round(total_ip, 2),
        bullpen_pitches=total_pitches,
    )


def team_last_bullpen_workload(
    client: MlbStatsClient,
    team_id: int,
    before_date: Date,
    lookback_days: int = 4,
) -> BullpenWorkload | None:
    """Find the most recent prior game for team_id and return bullpen usage."""
    for offset in range(1, lookback_days + 1):
        check = before_date - timedelta(days=offset)
        workload = team_bullpen_workload(client, team_id, check)
        if workload is not None:
            return workload
    return None


def predict_batters_faced(
    client: MlbStatsClient,
    pitcher_id: int,
    opp_team_id: int,
    run_date: Date,
    config: dict[str, Any],
) -> BattersFacedPrediction:
    """
    Blend default BF, recent-start average, and bullpen-rest adjustment.

    Heavy bullpen usage the prior night → manager may extend the starter (+BF).
    Fresh bullpen → shorter leash (−BF).
    """
    bf_cfg = config.get("batters_faced_model") or {}
    default_bf = float(config.get("default_batters_faced", 24))
    recent_n = int(bf_cfg.get("recent_starts_count", 3))
    recent_weight = float(bf_cfg.get("recent_weight", 0.55))
    default_weight = float(bf_cfg.get("default_weight", 0.25))
    bullpen_weight = float(bf_cfg.get("bullpen_weight", 0.20))
    min_bf = int(bf_cfg.get("min_bf", 18))
    max_bf = int(bf_cfg.get("max_bf", 30))

    high_ip = float(bf_cfg.get("bullpen_high_ip", 4.0))
    low_ip = float(bf_cfg.get("bullpen_low_ip", 2.0))
    adj_high = float(bf_cfg.get("bullpen_bf_adjust_high", 1.5))
    adj_low = float(bf_cfg.get("bullpen_bf_adjust_low", -1.0))
    lookback = int(bf_cfg.get("bullpen_lookback_days", 4))

    starts = pitcher_recent_starts(client, pitcher_id, run_date, count=recent_n)
    recent_avg: float | None = None
    recent_std: float | None = None
    if starts:
        values = [float(s.batters_faced) for s in starts]
        recent_avg = sum(values) / len(values)
        if len(values) > 1:
            mean = recent_avg
            recent_std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    bullpen = team_last_bullpen_workload(client, opp_team_id, run_date, lookback_days=lookback)
    bullpen_adj = 0.0
    pen_detail = "no_prior_game"
    if bullpen is not None and bullpen.game_date is not None:
        if bullpen.bullpen_ip >= high_ip or bullpen.relievers_used >= 5:
            bullpen_adj = adj_high
            pen_detail = f"pen_tired({bullpen.bullpen_ip}ip/{bullpen.relievers_used}r)"
        elif bullpen.bullpen_ip <= low_ip and bullpen.relievers_used <= 3:
            bullpen_adj = adj_low
            pen_detail = f"pen_fresh({bullpen.bullpen_ip}ip/{bullpen.relievers_used}r)"
        else:
            pen_detail = f"pen_neutral({bullpen.bullpen_ip}ip/{bullpen.relievers_used}r)"

    base = default_bf
    if recent_avg is not None:
        base = recent_weight * recent_avg + default_weight * default_bf
        base += bullpen_weight * bullpen_adj
    else:
        base += bullpen_adj

    batters_faced = int(round(max(min_bf, min(max_bf, base))))
    recent_part = f"last{len(starts)}={recent_avg:.1f}" if recent_avg is not None else "no_starts"
    detail = f"{recent_part} | {pen_detail} | adj={bullpen_adj:+.1f} → {batters_faced}bf"
    return BattersFacedPrediction(
        batters_faced=batters_faced,
        detail=detail,
        recent_avg_bf=recent_avg,
        bullpen_adj=bullpen_adj,
        recent_bf_std=recent_std,
    )
