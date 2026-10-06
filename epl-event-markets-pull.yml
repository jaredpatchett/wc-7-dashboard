name: EPL per-match markets pull (manual)

# Manual trigger only. One API call per (match, market), ~10 credits each.
# Partial results are ALWAYS saved (commit + downloadable artifact) even if the run stops early;
# re-running resumes and never re-spends credits on pairs already saved.
# Needs data/epl_odds_multi.json (run the multi-market pull for the season first).
on:
  workflow_dispatch:
    inputs:
      date_from:
        description: 'First match date (YYYY-MM-DD)'
        default: '2024-08-16'
      date_to:
        description: 'Last match date (YYYY-MM-DD)'
        default: '2025-05-25'
      markets:
        description: 'Markets, comma-separated (10 credits per match per market)'
        default: 'player_goal_scorer_anytime,alternate_totals,btts'
      max_credits:
        description: 'Stop the run if it spends more than this'
        default: '12000'

permissions:
  contents: write

jobs:
  pull:
    runs-on: ubuntu-latest
    timeout-minutes: 180
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Pull per-match markets
        env:
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
        run: |
          python scripts/epl_event_markets_pull.py \
            --from "${{ inputs.date_from }}" --to "${{ inputs.date_to }}" \
            --markets "${{ inputs.markets }}" --max-credits "${{ inputs.max_credits }}"
      - name: Save data as a downloadable artifact (backup)
        if: ${{ always() }}
        uses: actions/upload-artifact@v4
        with:
          name: epl-event-markets
          path: data/epl_event_markets.json
          if-no-files-found: ignore
      - name: Commit results (runs even if the pull stopped early)
        if: ${{ always() }}
        run: |
          git config user.name "github-actions"
          git config user.email "actions@users.noreply.github.com"
          if [ -f data/epl_event_markets.json ]; then
            git add data/epl_event_markets.json
            git commit -m "data: EPL per-match markets ${{ inputs.date_from }}..${{ inputs.date_to }}" || echo "nothing to commit"
            git pull --rebase origin ${{ github.ref_name }} || true
            git push
          else
            echo "no data file produced"
          fi
