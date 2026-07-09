# WC7 Dashboard — hosted World Cup 2026 model

Live Dixon-Coles model dashboard, auto-refit + live odds via GitHub Actions,
served by Cloudflare Pages.

## How it works
- `index.html` — the app. Loads all data from `data/data.js` (no rebuild needed).
- `data/match_xg_dataset.json` — source of truth. Append matches here.
- `scripts/refit.py` — refits ratings + baseline, rewrites `data/data.js`.
- `scripts/pull_odds.py` — pulls h2h + totals from The Odds API (Pinnacle,
  DraftKings, FanDuel) into `D.live_odds`, refreshes `D.dk_lines`.
- `.github/workflows/update.yml` — cron (6x/day) + manual trigger + auto-run
  whenever `match_xg_dataset.json` changes. Commits `data.js`, which triggers
  a Cloudflare Pages redeploy automatically.

## Setup checklist
1. Push this repo to GitHub (`wc-7-dashboard`).
2. Secret must be named exactly `ODDS_API_KEY`
   (repo -> Settings -> Secrets and variables -> Actions).
3. Cloudflare Pages: Framework preset = None, Build command = (empty),
   Build output directory = `/` (repo root).
4. Actions tab -> "Refit model + pull odds" -> Run workflow (first manual run).

## Adding a match after it finishes
Append to `data/match_xg_dataset.json`:
```json
{"home": "France", "away": "Morocco", "xg_home": 1.9, "xg_away": 0.8,
 "date": "2026-07-09", "goals_home": 2, "goals_away": 0}
```
Push. The workflow refits and redeploys automatically within ~2 minutes.

## Credit budget (The Odds API free tier: 500/mo)
One pull = h2h+totals x us,eu regions = ~4 credits. 6 runs/day ≈ 24/day.
