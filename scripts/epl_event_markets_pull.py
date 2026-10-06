#!/usr/bin/env python3
"""
epl_event_markets_pull.py -- per-match historical odds for ADDITIONAL markets
(player_goal_scorer_anytime, alternate_totals, btts, player_shots_on_target, ...)
for matches already listed in data/epl_odds_multi.json (that file supplies the
event IDs and kickoff times, so run the multi-market pull for the season first).

One API call per (match, market) at kickoff-10min; ~10 credits each (verified by
the probe: 10 per call, bookmakers list billed as a single group). Resumable:
(event, market) pairs already in --out are skipped, results are saved after every
match, and the credit cap stops the run cleanly. The workflow commits + uploads
whatever was saved even if the run stops early.

Requires env ODDS_API_KEY. Never put the key on the command line or in chat.
"""
import argparse, json, os, sys, time, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timedelta, timezone

BASE = 'https://api.the-odds-api.com/v4'
SPORT = 'soccer_epl'
HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, '..', 'data', 'epl_odds_multi.json')
DEFAULT_BOOKS = 'pinnacle,betfair_ex_eu,fanduel,betmgm,williamhill,draftkings'


def call(path, params):
    url = f'{BASE}{path}?' + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            return r.status, json.loads(r.read().decode()), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200], e.headers          # never echo the URL (has the key)


def compact(ev, market):
    """-> {bookmaker: [ {name, description, point, price}, ... ]} for the requested market only."""
    out = {}
    for b in ev.get('bookmakers', []):
        for m in b.get('markets', []):
            if m.get('key') == market:
                out[b['key']] = [{'name': o.get('name'), 'description': o.get('description'),
                                  'point': o.get('point'), 'price': o.get('price')} for o in m.get('outcomes', [])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='d_from', required=True)
    ap.add_argument('--to', dest='d_to', required=True)
    ap.add_argument('--markets', default='player_goal_scorer_anytime,alternate_totals,btts')
    ap.add_argument('--bookmakers', default=DEFAULT_BOOKS)
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'data', 'epl_event_markets.json'))
    ap.add_argument('--max-credits', type=float, default=12000)
    ap.add_argument('--reserve', type=float, default=3000)
    a = ap.parse_args()

    key = os.environ.get('ODDS_API_KEY')
    if not key:
        sys.exit('ODDS_API_KEY env var not set')
    if len(a.bookmakers.split(',')) > 10:
        sys.exit('Use at most 10 bookmakers (more is billed as extra regions)')
    markets = [m.strip() for m in a.markets.split(',') if m.strip()]
    d0, d1 = a.d_from, a.d_to

    events = sorted((r for r in json.load(open(SRC)) if d0 <= r['commence_time'][:10] <= d1), key=lambda r: r['commence_time'])
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r['event_id'], r['market']) for r in rows}
    todo = [(e, m) for e in events for m in markets if (e['event_id'], m) not in done]
    print(f'{len(events)} matches in range | {len(todo)} (match, market) calls to make | {len(done)} already saved | est. credits ~{len(todo)*10:,}')

    spent, remaining, last_event = 0.0, None, None
    for n, (e, m) in enumerate(todo, 1):
        ct = datetime.strptime(e['commence_time'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        snap = (ct - timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%SZ')
        status, body, hdr = call(f'/historical/sports/{SPORT}/events/{e["event_id"]}/odds',
                                 {'apiKey': key, 'markets': m, 'bookmakers': a.bookmakers, 'regions': 'eu,us,uk',
                                  'oddsFormat': 'decimal', 'date': snap})
        try: spent += float(hdr.get('x-requests-last') or 0)
        except ValueError: pass
        try: remaining = float(hdr.get('x-requests-remaining'))
        except (TypeError, ValueError): pass
        if status != 200:
            print(f'HTTP {status} for {e["home"]} v {e["away"]} [{m}]: {str(body)[:120]} -- skipping')
            if status in (401, 403, 429):
                break
            continue
        data = body.get('data', body)
        if isinstance(data, list): data = data[0] if data else {}
        rows.append({'event_id': e['event_id'], 'commence_time': e['commence_time'], 'home': e['home'], 'away': e['away'],
                     'snapshot': snap, 'market': m, 'books': compact(data, m)})
        if e['event_id'] != last_event or n == len(todo):          # save after every match
            json.dump(rows, open(a.out, 'w')); last_event = e['event_id']
        if n % 50 == 0:
            print(f'  {n}/{len(todo)} calls | spent {spent:.0f} | remaining {remaining}')
        if spent > a.max_credits:
            json.dump(rows, open(a.out, 'w'))
            sys.exit(f'STOP: spent {spent:.0f} > --max-credits {a.max_credits:.0f} (progress saved; re-run to resume)')
        if remaining is not None and remaining < a.reserve:
            json.dump(rows, open(a.out, 'w'))
            sys.exit(f'STOP: only {remaining:.0f} credits remaining (< reserve {a.reserve:.0f}) (progress saved)')
        time.sleep(0.2)
    json.dump(rows, open(a.out, 'w'))
    print(f'DONE. {len(rows)} (match, market) records in {os.path.relpath(a.out)} | credits spent this run: {spent:.0f} | remaining: {remaining}')


if __name__ == '__main__':
    main()
