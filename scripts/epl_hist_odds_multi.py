#!/usr/bin/env python3
"""
epl_hist_odds_multi.py -- near-closing odds for completed EPL matches from The
Odds API *historical* endpoints, for h2h (1X2), totals (O/U) and spreads (Asian
handicap), across several bookmakers, so strategies can be tested against both
the sharp price (Pinnacle) and the soft books (DraftKings, FanDuel, ...).

Requires env var ODDS_API_KEY (paid plan). Never put the key on the command
line or in chat.

Cost control (IMPORTANT): the API prices a call at ~10 credits per market per
"region", and every group of up to 10 bookmakers counts as one region when you
use the `bookmakers` parameter. This script passes `bookmakers` (<=10 keys) so a
snapshot should cost ~10 x number_of_markets. Do NOT assume -- run with
--probe first: it takes ONE real snapshot, prints the actual credits charged and
which bookmakers/markets came back, and saves nothing.

Method per run (same as epl_hist_odds.py):
  1. list EPL events each date in [--from, --to] (historical events endpoint)
  2. group by exact kickoff time; ONE sport-level snapshot at kickoff-10min per
     distinct kickoff; keep events kicking off at that time
  3. append to --out (resumable: kickoffs already saved are skipped)
Stops if spend > --max-credits or remaining < --reserve.
"""
import argparse, json, os, sys, time, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timedelta, timezone

BASE = 'https://api.the-odds-api.com/v4'
SPORT = 'soccer_epl'
DEFAULT_BOOKS = 'pinnacle,betfair_ex_eu,draftkings,fanduel,betmgm,williamhill'


class Budget:
    def __init__(self, max_credits, reserve):
        self.max_credits, self.reserve = max_credits, reserve
        self.spent, self.remaining, self.last = 0.0, None, None

    def update(self, headers):
        last, rem = headers.get('x-requests-last'), headers.get('x-requests-remaining')
        try:
            if last is not None:
                self.last = float(last); self.spent += self.last
            if rem is not None:
                self.remaining = float(rem)
        except ValueError:
            pass

    def check(self):
        if self.spent > self.max_credits:
            sys.exit(f'STOP: spent {self.spent} credits > --max-credits {self.max_credits}')
        if self.remaining is not None and self.remaining < self.reserve:
            sys.exit(f'STOP: only {self.remaining} credits remaining (< reserve {self.reserve})')


def fetch_json(path, params, budget):
    url = f'{BASE}{path}?' + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            body = json.loads(r.read().decode())
            budget.update(r.headers)
    except urllib.error.HTTPError as e:
        sys.exit(f'HTTP {e.code} on {path}: {e.read().decode()[:300]}')   # never echo the URL (has the key)
    budget.check()
    return body


iso = lambda dt: dt.strftime('%Y-%m-%dT%H:%M:%SZ')
parse_ts = lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
unwrap = lambda body: body['data'] if isinstance(body, dict) and 'data' in body else body


def extract(ev):
    """Per-bookmaker h2h / totals / spreads for one event -> {book: {market: {...}}}."""
    books = {}
    for b in ev.get('bookmakers', []):
        rec = {}
        for m in b.get('markets', []):
            k, outs = m.get('key'), m.get('outcomes', [])
            if k == 'h2h':
                px = {o['name']: o['price'] for o in outs}
                if ev['home_team'] in px and ev['away_team'] in px and 'Draw' in px:
                    rec['h2h'] = {'home': px[ev['home_team']], 'draw': px['Draw'], 'away': px[ev['away_team']]}
            elif k == 'totals':
                ov = [o for o in outs if o['name'] == 'Over']; un = [o for o in outs if o['name'] == 'Under']
                if ov and un and 'point' in ov[0]:
                    rec['totals'] = {'point': ov[0]['point'], 'over': ov[0]['price'], 'under': un[0]['price']}
            elif k == 'spreads':
                h = [o for o in outs if o['name'] == ev['home_team']]; a = [o for o in outs if o['name'] == ev['away_team']]
                if h and a and 'point' in h[0]:
                    rec['spreads'] = {'home_point': h[0]['point'], 'home': h[0]['price'], 'away_point': a[0]['point'], 'away': a[0]['price']}
        if rec:
            books[b['key']] = rec
    return books


def snapshot_params(a, key, t):
    return {'apiKey': key, 'markets': a.markets, 'bookmakers': a.bookmakers, 'regions': 'eu,us,uk',
            'oddsFormat': 'decimal', 'date': iso(t)}


def discover(a, key, budget, d0, d1, stop_after_first=False):
    kickoffs, d = {}, d0
    while d <= d1:
        body = fetch_json(f'/historical/sports/{SPORT}/events', {'apiKey': key, 'date': iso(d)}, budget)
        for ev in unwrap(body):
            if parse_ts(ev['commence_time']).date() == d.date():
                kickoffs.setdefault(ev['commence_time'], set()).add(ev['id'])
        if stop_after_first and kickoffs:
            break
        d += timedelta(days=1)
        time.sleep(0.25)
    return kickoffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='d_from', required=True)
    ap.add_argument('--to', dest='d_to', required=True)
    ap.add_argument('--markets', default='h2h,totals,spreads')
    ap.add_argument('--bookmakers', default=DEFAULT_BOOKS)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'data', 'epl_odds_multi.json'))
    ap.add_argument('--max-credits', type=float, default=2000)
    ap.add_argument('--reserve', type=float, default=2000)
    ap.add_argument('--probe', action='store_true')
    a = ap.parse_args()

    key = os.environ.get('ODDS_API_KEY')
    if not key:
        sys.exit('ODDS_API_KEY env var not set')
    if len(a.bookmakers.split(',')) > 10:
        sys.exit('Use at most 10 bookmakers (more than 10 is billed as extra regions)')
    budget = Budget(a.max_credits, a.reserve)
    d0 = datetime.strptime(a.d_from, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    d1 = datetime.strptime(a.d_to, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    if a.probe:
        ks = discover(a, key, budget, d0, d1, stop_after_first=True)
        if not ks:
            sys.exit('PROBE: no EPL kickoffs found in range')
        ct = sorted(ks)[0]
        before = budget.spent
        body = fetch_json(f'/historical/sports/{SPORT}/odds', snapshot_params(a, key, parse_ts(ct) - timedelta(minutes=10)), budget)
        evs = [e for e in unwrap(body) if e['commence_time'] == ct]
        print(f'PROBE kickoff {ct}: {len(evs)} match(es) | credits charged for this ONE snapshot: {budget.last} | remaining: {budget.remaining}')
        n_snap_est = 245
        print(f'-> a full season is ~{n_snap_est} snapshots => ~{n_snap_est * (budget.last or 0):,.0f} credits (+ ~300 for discovery)')
        for e in evs[:1]:
            books = extract(e)
            print(f'sample: {e["home_team"]} v {e["away_team"]}; bookmakers returned: {sorted(books)}')
            for bk, rec in books.items():
                print(f'  {bk}: markets {sorted(rec)}  ' + json.dumps({k: rec[k] for k in rec if k != 'h2h'}))
        return

    kickoffs = discover(a, key, budget, d0, d1)
    existing = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {e['commence_time'] for e in existing}
    todo = sorted(k for k in kickoffs if k not in done)
    print(f'{len(todo)} kickoff times to snapshot (skipping {len(done)} already saved). Spent so far: {budget.spent}')
    rows = list(existing)
    for ct in todo:
        snap = parse_ts(ct) - timedelta(minutes=10)
        body = fetch_json(f'/historical/sports/{SPORT}/odds', snapshot_params(a, key, snap), budget)
        n = 0
        for ev in unwrap(body):
            if ev['commence_time'] != ct:
                continue
            books = extract(ev)
            if books:
                rows.append({'event_id': ev['id'], 'commence_time': ct, 'home': ev['home_team'], 'away': ev['away_team'],
                             'snapshot': iso(snap), 'books': books}); n += 1
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rows, open(a.out, 'w'))
        print(f'{ct}: {n} match(es) | spent {budget.spent:.0f} | remaining {budget.remaining}')
        time.sleep(0.25)
    print(f'DONE. {len(rows)} matches in {os.path.relpath(a.out)}. Credits spent this run: {budget.spent:.0f}')


if __name__ == '__main__':
    main()
