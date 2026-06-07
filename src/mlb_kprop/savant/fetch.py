from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
import yaml

# Savant is happier if we do not hammer the server.
SECONDS_BETWEEN_DOWNLOADS = 2.0

DEFAULT_CONFIG_PATH = Path("config/savant_sources.yaml")


@dataclass(frozen=True)
class SavantSource:
    """One Savant export defined in savant_sources.yaml."""

    id: str
    kind: str
    url: str


@dataclass(frozen=True)
class FetchResult:
    """Summary of one download attempt."""

    source_id: str
    output_path: Path
    row_count: int


def load_sources(config_path: Path) -> list[SavantSource]:
    """Read platoon + custom URLs from the YAML config."""
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    sources: list[SavantSource] = []
    for section_name in ("platoon_pitcher", "platoon_batter", "custom_leaderboard"):
        for entry in config.get(section_name, []):
            sources.append(
                SavantSource(
                    id=str(entry["id"]),
                    kind=str(entry["kind"]),
                    url=str(entry["url"]),
                )
            )
    return sources


def csv_download_url(source: SavantSource) -> str:
    """
    Turn a browser Savant URL into a direct CSV download URL.

    Statcast Search uses /statcast_search/csv?...&all=true
    Custom leaderboards use the same URL with &csv=true
    """
    cleaned = source.url.split("#", maxsplit=1)[0].strip()

    if source.kind == "statcast_search":
        if "/statcast_search/csv" in cleaned:
            base = cleaned
        else:
            base = cleaned.replace("/statcast_search?", "/statcast_search/csv?", 1)
        parts = urlparse(base)
        query = parse_qs(parts.query, keep_blank_values=True)
        query["all"] = ["true"]
        new_query = urlencode(query, doseq=True)
        return urlunparse(parts._replace(query=new_query))

    if source.kind == "custom_leaderboard":
        parts = urlparse(cleaned)
        query = parse_qs(parts.query, keep_blank_values=True)
        query["csv"] = ["true"]
        new_query = urlencode(query, doseq=True)
        return urlunparse(parts._replace(query=new_query))

    raise ValueError(f"Unknown source kind: {source.kind}")


def count_csv_rows(csv_text: str) -> int:
    """Count data rows (header does not count)."""
    lines = [line for line in csv_text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0
    return len(lines) - 1


def download_source(
    source: SavantSource,
    output_path: Path,
    session: requests.Session,
) -> FetchResult:
    """Download one Savant CSV and write it to disk."""
    download_url = csv_download_url(source)
    response = session.get(download_url, timeout=120)
    response.raise_for_status()

    csv_text = response.text
    if len(csv_text.strip()) < 20:
        raise RuntimeError(
            f"Download for '{source.id}' looks empty. "
            "Check the URL in config/savant_sources.yaml."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(csv_text, encoding="utf-8")

    return FetchResult(
        source_id=source.id,
        output_path=output_path,
        row_count=count_csv_rows(csv_text),
    )


def fetch_all_sources(
    run_date: Date,
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_root: Path = Path("data/raw"),
) -> list[FetchResult]:
    """
    Download every source in the config into:
      data/raw/YYYY-MM-DD/<source_id>.csv
    """
    sources = load_sources(config_path)
    if not sources:
        raise RuntimeError(f"No sources found in {config_path}")

    day_dir = data_root / run_date.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "mlb-kprop/0.1 (personal research; baseballsavant fetch)",
        }
    )

    results: list[FetchResult] = []
    for index, source in enumerate(sources):
        if index > 0:
            time.sleep(SECONDS_BETWEEN_DOWNLOADS)

        output_path = day_dir / f"{source.id}.csv"
        print(f"Downloading {source.id} ...")
        result = download_source(source, output_path, session)
        print(f"  saved {result.output_path} ({result.row_count} rows)")
        results.append(result)

    return results


def write_fetch_manifest(results: list[FetchResult], manifest_path: Path) -> None:
    """Write a small text summary so you can see what ran."""
    lines = ["source_id,output_path,row_count"]
    for result in results:
        lines.append(
            f"{result.source_id},{result.output_path},{result.row_count}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
