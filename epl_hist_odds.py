#!/usr/bin/env python3
"""
epl_hist_odds.py -- pull near-closing 1X2 (h2h) odds for completed EPL matches
from The Odds API *historical* endpoints, for backtesting the model against
the market.

Requires env var ODDS_API_KEY (a PAID plan -- historical endpoints are not on
the free tier). Never pass the key on the command line or paste it in chat.

How it works (kept deliberately cheap):
  1. For each calendar date in [--from, --to], list the EPL events scheduled
     that day (historical events endpoint, snapshot at 00:00 UTC that date).
  2. Group those events by exact kickoff time. For each distinct kickoff time T,
     take ONE sport-level odds snapshot at T minus 10 minutes and keep the
     events kicking off at T. (One call serves every match at that kickoff.)
  3. Save bookmaker h2h prices (decimal) to data/epl_hist_odds.json. Resumable:
     kickoffs already in the file are skipped on re-run.

Credit guard: reads x-requests-remaining / x-requests-last headers after every
call and stops if spend exceeds --max-credits or remaining drops below
--reserve. Use --dry-run first: it makes ONE cheap call and prints the plan.

Cost note: the documented cost of a historical odds call is ~10 credits per
region per market; this uses 1 region ('eu', includes Pinnacle) x 1 market
(h2h) = ~10 per kickoff snapshot. Check the printed headers on the first real
call and adjust --max-credits if your plan differs.
"""
import argparse, json, os, sys, time, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timedelta, timezone

BASE = 'https://api.the-odds-api.com/v4'
SPORT = 'soccer_epl'
OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'epl_hist_odds.json')


class Budget:
    def __init__(self, max_credits, reserve):
        self.max_credits, self.reserve = max_credits, reserve
        self.spent, self.remaining = 0, None

    def update(self, headers):
        last = headers.get('x-requests-last'); rem = headers.get('x-requests-remaining')
        if last is not None:
            try: self.spent += float(last)
            except ValueError: pass
        if rem is not None:
            try: self.remaining = float(rem)
            except ValueError: pass

    def check(self):
        if self.spent > self.max_credits:
            sys.exit(f'STOP: spent {self.spent} credits > --max-credits {self.max_credits}')
        if self.remaining is not None and self.remaining < self.reserve:
            sys.exit(f'STOP: only {self.remaining} credits remaining (< reserve {self.reserve})')


def fetch_json(path, params, budget):
    url = f'{BASE}{path}?' + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = json.loads(r.read().decode())
            budget.update(r.headers)
    except urllib.error.HTTPError as e:
        # never echo the URL (it contains the key)
        sys.exit(f'HTTP {e.code} on {path}: {e.read().decode()[:200]}')
    budget.check()
    return body


def iso(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_ts(s):
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)


def unwrap(body):
    """Historical endpoints wrap results as {timestamp, previous_timestamp, next_timestamp, data}."""
    return body['data'] if isinstance(body, dict) and 'data' in body else body


def h2h_from_event(ev):
    books = {}
    for b in ev.get('bookmakers', []):
        for m in b.get('markets', []):
            if m.get('key') != 'h2h':
                continue
            px = {o['name']: o['price'] for o in m.get('outcomes', [])}
            if ev['home_team'] in px and ev['away_team'] in px and 'Draw' in px:
                books[b['key']] = {'home': px[ev['home_team']], 'draw': px['Draw'], 'away': px[ev['away_team']]}
    return books


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='d_from', required=True)
    ap.add_argument('--to', dest='d_to', required=True)
    ap.add_argument('--regions', default='eu')
    ap.add_argument('--max-credits', type=float, default=700)
    ap.add_argument('--reserve', type=float, default=200)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    key = os.environ.get('ODDS_API_KEY')
    if not key:
        sys.exit('ODDS_API_KEY env var not set')
    budget = Budget(a.max_credits, a.reserve)

    existing = []
    if os.path.exists(OUT):
        existing = json.load(open(OUT))
    done_kickoffs = {e['commence_time'] for e in existing}

    d0 = datetime.strptime(a.d_from, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    d1 = datetime.strptime(a.d_to, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    # Step 1: discover kickoff times (events endpoint, one call per date)
    kickoffs = {}
    d = d0
    while d <= d1:
        body = fetch_json(f'/historical/sports/{SPORT}/events', {'apiKey': key, 'date': iso(d)}, budget)
        for ev in unwrap(body):
            ct = ev['commence_time']
            if parse_ts(ct).date() == d.date():
                kickoffs.setdefault(ct, set()).add(ev['id'])
        if a.dry_run:
            break
        d += timedelta(days=1)
        time.sleep(0.25)

    todo = sorted(k for k in kickoffs if k not in done_kickoffs)
    if a.dry_run:
        print(f'DRY RUN (one events call on {a.d_from}): found {len(kickoffs)} kickoff time(s) that day.')
        print(f'Credits spent so far: {budget.spent}, remaining: {budget.remaining}')
        print(f'Full run would be ~(distinct kickoffs in range) x ~10 credits for odds snapshots, plus ~1 per date for discovery.')
        return

    print(f'{len(todo)} kickoff times to snapshot (skipping {len(done_kickoffs)} already saved). Spent so far: {budget.spent}')

    # Step 2: one odds snapshot per distinct kickoff, taken 10 min before
    rows = list(existing)
    for ct in todo:
        snap = parse_ts(ct) - timedelta(minutes=10)
        body = fetch_json(f'/historical/sports/{SPORT}/odds',
                          {'apiKey': key, 'regions': a.regions, 'markets': 'h2h', 'oddsFormat': 'decimal', 'date': iso(snap)},
                          budget)
        n = 0
        for ev in unwrap(body):
            if ev['commence_time'] != ct:
                continue
            books = h2h_from_event(ev)
            if books:
                rows.append({'event_id': ev['id'], 'commence_time': ct, 'home': ev['home_team'], 'away': ev['away_team'],
                             'snapshot': iso(snap), 'books': books})
                n += 1
        # save after each snapshot so a stopped run loses nothing
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(rows, open(OUT, 'w'), indent=1)
        print(f'{ct}: {n} match(es) saved | spent {budget.spent} | remaining {budget.remaining}')
        time.sleep(0.25)

    print(f'DONE. {len(rows)} matches in {os.path.relpath(OUT)}. Credits spent this run: {budget.spent}')


if __name__ == '__main__':
    main()
