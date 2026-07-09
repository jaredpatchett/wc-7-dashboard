#!/usr/bin/env python3
"""
pull_odds.py — pulls World Cup odds from The Odds API and merges them into
data/data.js as D.live_odds (and refreshes D.dk_lines for matched fixtures).

Requires env var ODDS_API_KEY (set as a GitHub Actions secret).

CREDIT BUDGETING (free tier = 500 credits/month):
  Cost per request = #markets x #regions. This script makes ONE request for
  h2h + totals across the "us,eu" regions = 4 credits per run. At a
  6x/day cron on match days, that's ~24 credits/day — comfortably inside
  the free tier for the remainder of the tournament.
"""
import json, re, os, sys, urllib.request, urllib.parse
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
        with urllib.request.urlopen(url, timeout=30) as resp:
            remaining = resp.headers.get('x-requests-remaining')
            events = json.loads(resp.read())
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
        if entry['books']:
            live.append(entry)

    D = load_data_js(DATA_JS)
    D['live_odds'] = live
    D['odds_updated'] = datetime.now(timezone.utc).isoformat(timespec='seconds')

    # keep the app's existing dk_lines panel fresh where fixtures match
    dk = D.get('dk_lines', {})
    for ev in live:
        fkey = f"{ev['home']}_vs_{ev['away']}"
        book = ev['books'].get('draftkings') or ev['books'].get('pinnacle')
        if book and any(k in book for k in ('ml_h', 'ml_a')):
            dk.setdefault(fkey, {})
            for k in ('ml_h', 'ml_d', 'ml_a'):
                if k in book:
                    dk[fkey][k] = book[k]
    D['dk_lines'] = dk

    with open(DATA_JS, 'w') as f:
        f.write('var D=' + json.dumps(D) + ';')
    print(f'Odds OK: {len(live)} events merged. '
          f'API credits remaining: {remaining}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
