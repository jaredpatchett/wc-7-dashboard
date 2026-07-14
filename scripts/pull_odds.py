#!/usr/bin/env python3
"""
pull_odds.py — pulls World Cup odds from The Odds API and merges them into
data/data.js as D.live_odds (and refreshes D.dk_lines for matched fixtures).

Requires env var ODDS_API_KEY (set as a GitHub Actions secret).

CREDIT BUDGETING (free tier = 500 credits/month):
  Main pull: h2h + totals across "us,eu" regions = 4 credits per run.
  Alternate totals: The Odds API only returns the primary line (e.g. 2.5)
  from the main /odds endpoint -- getting neighboring lines (1.5, 3.5, etc)
  requires a SEPARATE per-event call to /events/{id}/odds. This adds
  roughly 1 extra call per live event (typically 1-2 events this late in
  the tournament), ~2-4 more credits per run. At a 6x/day cron, still
  comfortably inside the free tier.

  Why this matters: the play-classifier's signal-consistency gate (added
  after the Morocco ML mistake) requires REAL market prices at neighboring
  total-goals lines to confirm a totals signal isn't a one-line pricing
  quirk. Without alternate_totals, that gate can never pass for anything,
  which was silently downgrading genuinely good totals plays (Over 2.5 on
  France-Spain, consistent across every threshold in the model, still
  came back "Lean" instead of "Elite" purely from missing this data).
"""
import json, re, os, sys, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timezone

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_JS = os.path.join(ROOT, 'data', 'data.js')

SPORT_KEY = 'soccer_fifa_world_cup'
MARKETS = 'h2h,totals'
REGIONS = 'us,eu'          # eu region includes Pinnacle
PREFERRED_BOOKS = ['pinnacle', 'draftkings', 'fanduel']

# The Odds API team names -> WC7 dataset team names (extend as needed)
NAME_MAP = {
    'United States': 'USA', 'South Korea': 'South Korea',
    'Republic of Korea': 'South Korea', 'Korea Republic': 'South Korea',
    'Cape Verde': 'Cabo Verde', 'Cabo Verde Islands': 'Cabo Verde',
    'Czech Republic': 'Czechia', 'Bosnia and Herzegovina': 'Bosnia',
    'Türkiye': 'Turkey', 'Turkiye': 'Turkey', 'Ireland': 'Ireland',
    'Côte d\u2019Ivoire': 'Ivory Coast', "Cote d'Ivoire": 'Ivory Coast',
    'DR Congo': 'DR Congo', 'Congo DR': 'DR Congo',
}


def canon(name):
    return NAME_MAP.get(name, name)


def decimal_to_american(dec):
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    return int(round(-100 / (dec - 1)))


def load_data_js(path):
    txt = open(path).read()
    m = re.match(r'\s*var\s+D\s*=\s*(\{.*\});?\s*$', txt, re.S)
    if not m:
        raise ValueError('data.js format unexpected')
    return json.loads(m.group(1))


def fetch_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        remaining = resp.headers.get('x-requests-remaining')
        return json.loads(resp.read()), remaining


def fetch_alternate_totals(event_id, key):
    """Per-event call for alternate totals lines (1.5, 2.5, 3.5, etc).
    Not available on the bulk /odds endpoint -- The Odds API only exposes
    additional markets through the per-event endpoint. Returns a dict of
    {bookmaker_key: {over_X: american_odds, under_X: american_odds}} or
    {} on any failure (never blocks the main odds pull)."""
    params = urllib.parse.urlencode({
        'apiKey': key, 'regions': REGIONS, 'markets': 'alternate_totals',
        'oddsFormat': 'decimal',
    })
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events/{event_id}/odds?{params}'
    try:
        data, _ = fetch_json(url)
    except Exception as e:
        print(f'  alternate_totals fetch failed for event {event_id}: {e}')
        return {}

    out = {}
    for bk in data.get('bookmakers', []):
        if bk['key'] not in PREFERRED_BOOKS:
            continue
        b = {}
        for mkt in bk.get('markets', []):
            if mkt['key'] != 'alternate_totals':
                continue
            for o in mkt['outcomes']:
                tag = 'over' if o['name'] == 'Over' else 'under'
                b[f"{tag}_{o.get('point')}"] = decimal_to_american(o['price'])
        if b:
            out[bk['key']] = b
    return out


def main():
    key = os.environ.get('ODDS_API_KEY')
    if not key:
        print('ODDS_API_KEY not set — skipping odds pull (refit-only run).')
        return 0

    params = urllib.parse.urlencode({
        'apiKey': key, 'regions': REGIONS, 'markets': MARKETS,
        'oddsFormat': 'decimal', 'dateFormat': 'iso',
    })
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds?{params}'
    try:
        events, remaining = fetch_json(url)
    except urllib.error.HTTPError as e:
        print(f'Odds API HTTP {e.code}: {e.read()[:300]}')
        return 0 if e.code in (404, 422) else 1  # no events = fine, keep site up

    live = []
    for ev in events:
        home, away = canon(ev['home_team']), canon(ev['away_team'])
        entry = {'home': home, 'away': away,
                 'commence': ev.get('commence_time'), 'books': {}}
        for bk in ev.get('bookmakers', []):
            if bk['key'] not in PREFERRED_BOOKS:
                continue
            b = {}
            for mkt in bk.get('markets', []):
                if mkt['key'] == 'h2h':
                    for o in mkt['outcomes']:
                        nm = canon(o['name'])
                        side = ('ml_h' if nm == home else
                                'ml_a' if nm == away else 'ml_d')
                        b[side] = decimal_to_american(o['price'])
                elif mkt['key'] == 'totals':
                    for o in mkt['outcomes']:
                        tag = 'over' if o['name'] == 'Over' else 'under'
                        b[f"{tag}_{o.get('point')}"] = \
                            decimal_to_american(o['price'])
            if b:
                entry['books'][bk['key']] = b

        # enrich with alternate totals lines -- needed for the classifier's
        # signal-consistency gate. Best-effort: never blocks the main pull.
        alt = fetch_alternate_totals(ev['id'], key)
        for book_key, alt_lines in alt.items():
            entry['books'].setdefault(book_key, {}).update(alt_lines)

        if entry['books']:
            live.append(entry)

    D = load_data_js(DATA_JS)
    D['live_odds'] = live
    D['odds_updated'] = datetime.now(timezone.utc).isoformat(timespec='seconds')

    # keep the app's existing dk_lines panel fresh where fixtures match.
    # merge PER FIELD across preferred books, not per book -- a book that
    # has moneyline but not totals (like DraftKings on this pull) shouldn't
    # block totals data that IS available from Pinnacle/FanDuel for the
    # same fixture.
    dk = D.get('dk_lines', {})
    for ev in live:
        fkey = f"{ev['home']}_vs_{ev['away']}"
        merged = {}
        for book_key in reversed(PREFERRED_BOOKS):  # reversed so first-listed wins on conflicts
            book = ev['books'].get(book_key)
            if book:
                merged.update(book)
        if merged:
            dk[fkey] = merged
    D['dk_lines'] = dk

    with open(DATA_JS, 'w') as f:
        f.write('var D=' + json.dumps(D) + ';')
    print(f'Odds OK: {len(live)} events merged (incl. alternate totals). '
          f'API credits remaining: {remaining}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
