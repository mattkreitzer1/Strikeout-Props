# MLB Pitcher K Prop Model

Scrape Baseball Savant inputs, build features (platoon splits emphasized), and predict pitcher strikeout **Over/Under** lines with EV vs sportsbooks.

## Project layout

- `config/savant_sources.yaml` — Savant URLs (your 4 platoon splits live here)
- `src/mlb_kprop/` — Python code
- `data/raw/YYYY-MM-DD/` — downloaded CSVs (not committed to git)
- `reports/` — manifests and future daily prop outputs

## Setup (one time)

```bash
cd "/Users/joelkreitzer/Desktop/MLB Cursor Project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --config-settings editable_mode=compat
```

The `editable_mode=compat` flag matters because this project folder has a **space** in the name (`MLB Cursor Project`). Without it, `pip install -e .` can look successful but `python -m mlb_kprop` still fails with `No module named mlb_kprop`.

## Automate Savant downloads (v1 — working now)

Downloads all sources in `config/savant_sources.yaml` into dated folders:

```bash
python -m mlb_kprop fetch-savant
```

## Build cleaned feature tables (v1 — working now)

Turns raw downloads into smaller, model-ready tables:

```bash
python -m mlb_kprop build-features
```

Outputs land in `data/processed/<today>/`.

## Merge pitcher tables (v1 — working now)

Combines platoon pitch-type rows with custom leaderboard context:

```bash
python -m mlb_kprop merge-features
```

Creates:

- `pitcher_merged_long.csv` — platoon rows + `custom_*` columns (season context)
- `pitcher_split_summary.csv` — one row per pitcher per platoon split (pitch-weighted rates)

`run-daily` runs the full stack in one command:

```bash
python -m mlb_kprop run-daily
```

Steps: Savant fetch → features → validate → OddsTrader K lines → MLB probables/lineups → fair K → EV sheet (`reports/value_<date>.csv`) → performance tracker.

## Validate today's data (automated checks)

Runs sanity checks on every file in config (raw downloads, processed tables, merge math):

```bash
python -m mlb_kprop validate-data
```

Writes `reports/validation_<date>.txt`. Exits with an error if any check fails (so you know to investigate before trusting the numbers).

Raw downloads land in `data/raw/<date>/` (four platoon CSVs plus custom boards when configured).

## Phase 1: Fair strikeout projections (pitcher platoon)

Uses `pitcher_split_summary.csv` and `data/starters/<date>.csv` (auto-filled from MLB probables each `run-daily`).

**1. Refresh Savant data (if needed):**

```bash
python -m mlb_kprop run-daily
```

**2. Starters (automatic on `run-daily`, or manual):**

`run-daily` writes probables from the MLB Stats API and sets `opp_lhb_pct` from the posted batting order when available (otherwise `config/mlb_defaults.yaml` default, 0.40).

To edit by hand first:

```bash
python -m mlb_kprop init-starters
```

`data/starters/<today>.csv` columns:

| Column | Meaning |
|--------|---------|
| `player_id` | Savant ID (optional if `player_name` matches) |
| `player_name` | e.g. `Verlander, Justin` |
| `pitcher_throws` | `R` or `L` (leave blank to auto-guess from splits) |
| `opp_lhb_pct` | Share of opposing batters who bat **left** (0.42 = 42% LHB) |
| `batters_faced` | Expected batters (blank = 24 from config) |
| `notes` | Optional |

**3. Score fair K:**

```bash
python -m mlb_kprop score-projections
```

Output: `reports/projections_<date>.csv` with `fair_k` and `fair_k_line` (rounded to nearest 0.5).

Formula: `fair_k = (blended K% / 100) × batters_faced`, where blended K% weights `R_vs_L` / `R_vs_R` (or `L_vs_*` for lefties) by `opp_lhb_pct`.

Tweak defaults in `config/projection_defaults.yaml`.

## Phase 2: Book lines + edge / EV

Paste FanDuel (or other) strikeout props into a lines file and compare to `fair_k`.

**1. Projections (Phase 1) must exist:**

```bash
python -m mlb_kprop score-projections
```

**2. Book lines (automated or manual):**

Fetch strikeout O/U from [OddsTrader](https://www.oddstrader.com/mlb/player-props/?m=766) (free guest API):

```bash
python -m mlb_kprop fetch-odds
```

Writes `data/lines/<today>.csv` using `config/oddstrader.yaml` (default: BetRivers, market `766` = strikeouts). Event ids come from the MLB matchups page; lines use OddsTrader GraphQL `bestLines`.

Or create a template and paste by hand:

```bash
python -m mlb_kprop init-lines
```

| Column | Example |
|--------|---------|
| `player_name` | `Verlander, Justin` |
| `book_line` | `5.5` |
| `over_odds` | `-115` (American) |
| `under_odds` | `-105` |

**3. Value props:**

```bash
python -m mlb_kprop value-props
```

Output: `reports/value_<date>.csv` with model vs no-vig implied probability, **edge**, **EV**, and a **pick** (`OVER` / `UNDER` / `PASS`).

Tweak the normal spread around `fair_k` in `config/value_defaults.yaml` (`k_sigma`, `min_edge`).

### Full daily betting workflow

```bash
python -m mlb_kprop run-daily
```

Output: `reports/value_<date>.csv` with flagged `OVER` / `UNDER` plays.

Partial runs:

- Savant only: `python -m mlb_kprop run-daily --skip-odds --skip-model`
- No odds/EV: `python -m mlb_kprop run-daily --skip-odds`

## Add your custom leaderboard URLs

Open `config/savant_sources.yaml` and paste URLs under `custom_leaderboard:` (same format as platoon entries). Re-run `fetch-savant` to refresh zone/chase/arsenal exports.

## Run in the project folder (Terminal on your Mac)

The **project folder** is:

`/Users/joelkreitzer/Desktop/MLB Cursor Project`

Every command below assumes you are **in that folder** first.

### 1. Open Terminal

Spotlight (⌘ Space) → type **Terminal** → Enter.

### 2. Go to the project folder

Copy and paste this **once per Terminal window** (press Enter after):

```bash
cd "/Users/joelkreitzer/Desktop/MLB Cursor Project"
```

`cd` means “change directory” — you’re telling Terminal to work inside this project.

### 3. Activate Python (first time / new window)

```bash
source .venv/bin/activate
```

Your prompt may show `(.venv)`. Install the package once (or after code changes):

```bash
pip install -e . --config-settings editable_mode=compat
```

### 4. Run the full pipeline locally (optional test)

```bash
python -m mlb_kprop run-daily
```

Results land in `reports/value_<date>.csv` on your Mac. You **don’t** need this every day if GitHub Actions is set up.

---

## Run on GitHub Actions (scheduled, laptop off)

Workflow: [`.github/workflows/daily.yml`](.github/workflows/daily.yml)

- **Schedule:** 15:00 UTC daily (~11 AM Eastern Daylight Time). GitHub may delay scheduled runs by minutes to hours on free accounts.
- **Manual run:** GitHub → Actions → *Daily K props* → *Run workflow*
- **Artifacts:** backup CSVs (14-day retention)

### One-time: push code to GitHub

Still in the project folder (`cd` command from above):

```bash
git config user.email "YOUR_GITHUB_EMAIL"
git config user.name "YOUR_NAME"

git commit -m "Add MLB K prop pipeline and daily GitHub Actions workflow"
```

Create a **new empty repo** on [github.com](https://github.com/new) (private recommended), then:

```bash
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

Enable **Settings → Actions → General → Allow all actions**.

### Email to your phone

After each successful run, GitHub sends a digest to your inbox (top flagged plays + CSV attachment).

**1. Gmail app password** (if you use Gmail): Google Account → Security → 2-Step Verification → App passwords → create one for “Mail”.

**2. GitHub repo secrets** (Settings → Secrets and variables → Actions → New repository secret):

| Secret | Example |
|--------|---------|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | your Gmail address |
| `SMTP_PASSWORD` | 16-character app password |
| `EMAIL_TO` | same Gmail (or phone email if you use SMS gateway) |

Optional: `EMAIL_FROM` (defaults to `SMTP_USER`).

**3. Test email locally** (optional):

```bash
cd "/Users/joelkreitzer/Desktop/MLB Cursor Project"
source .venv/bin/activate
export SMTP_USER="you@gmail.com"
export SMTP_PASSWORD="your-app-password"
export EMAIL_TO="you@gmail.com"
python -m mlb_kprop send-email --date 2026-05-30 --dry-run   # preview
python -m mlb_kprop send-email --date 2026-05-30             # send
```

**4. Test on GitHub:** Actions → *Daily K props* → *Run workflow*. Check your inbox when it finishes.

`data/raw/` and `reports/` stay out of git; CI uploads artifacts and emails the digest.

## Performance tracker (high-EV plays)

Every `run-daily` records flagged plays (`pick` = OVER or UNDER in the value sheet) into a persistent ledger and grades them once MLB box scores are final. Grading uses actual strikeouts from the MLB Stats API vs the book line at bet time (flat 1-unit stakes, profit from the American odds on the pick).

Outputs (committed by GitHub Actions so history survives between runs):

| File | Purpose |
|------|---------|
| `data/tracker/ledger.csv` | One row per pick — line, EV, odds, actual K, W/L, units |
| `data/tracker/daily_rollup.csv` | Per-slate win/loss and ROI |
| `data/tracker/summary.txt` | Rolling totals + last 7 slate days (also appended to the daily email) |

**Backfill from existing value sheets:**

```bash
python -m mlb_kprop track-performance --date 2026-05-28
python -m mlb_kprop track-performance --date 2026-05-29
python -m mlb_kprop track-performance --date 2026-05-30
```

**Grade only** (no new picks for that date):

```bash
python -m mlb_kprop track-performance --date 2026-06-03 --no-record
```

Config: `config/tracker_defaults.yaml`.

## Data strategy (short)

- **Platoon Search CSVs** — primary for matchup K%, BB%, whiff, xwOBA by pitch type.
- **Custom leaderboard** — zone/chase, arsenal; avoid duplicating overall K% when platoon stats are used.
