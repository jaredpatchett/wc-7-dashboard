#!/usr/bin/env python3
"""
epl_event_probe.py -- for ONE historical EPL match, ask The Odds API's per-event
historical endpoint for each 'additional' market separately and report which
bookmakers returned it, what the outcomes look like, and what each call cost.
Saves nothing. Purpose: find out which spreads / alt totals / props actually
exist in the historical feed for DraftKings, FanDuel etc. BEFORE any big pull.

Picks the middle match of data/epl_odds_multi.json (so run the multi-market pull
first). Requires env ODDS_API_KEY. Expect roughly 10 credits per market probed.
"""
import json, os, sys, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timedelta, timezone

BASE = 'https://api.the-odds-api.com/v4'
SPORT = 'soccer_epl'
DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'epl_odds_multi.json')
BOOKS = 'pinnacle,draftkings,fanduel,betmgm,williamhill,betfair_ex_eu'
MARKETS = ['spreads', 'alternate_spreads', 'alternate_totals', 'btts', 'draw_no_bet', 'double_chance',
           'player_goal_scorer_anytime', 'player_shots_on_target']


def call(path, params):
    url = f'{BASE}{path}?' + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            return r.status, json.loads(r.read().decode()), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200], e.headers          # never echo the URL (has the key)


def main():
    key = os.environ.get('ODDS_API_KEY')
    if not key:
        sys.exit('ODDS_API_KEY env var not set')
    rows = sorted(json.load(open(DATA)), key=lambda r: r['commence_time'])
    ev = rows[len(rows) // 2]
    ct = datetime.strptime(ev['commence_time'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    snap = (ct - timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f'Probe match: {ev["home"]} v {ev["away"]}  kickoff {ev["commence_time"]}  snapshot {snap}  event_id {ev["event_id"]}')
    total, remaining = 0.0, None
    for mk in MARKETS:
        status, body, hdr = call(f'/historical/sports/{SPORT}/events/{ev["event_id"]}/odds',
                                 {'apiKey': key, 'markets': mk, 'bookmakers': BOOKS, 'regions': 'eu,us,uk',
                                  'oddsFormat': 'decimal', 'date': snap})
        last = hdr.get('x-requests-last'); remaining = hdr.get('x-requests-remaining', remaining)
        try: total += float(last)
        except (TypeError, ValueError): pass
        if status != 200:
            print(f'\n[{mk}] HTTP {status}: {str(body)[:160]}  (credits charged: {last})'); continue
        data = body.get('data', body)
        if isinstance(data, list): data = data[0] if data else {}
        books = data.get('bookmakers', [])
        print(f'\n[{mk}] HTTP 200 | credits charged: {last} | bookmakers returning it: {sorted(b["key"] for b in books) or "NONE"}')
        for b in books:
            for m in b.get('markets', []):
                if m.get('key') != mk: continue
                outs = m.get('outcomes', [])
                pts = sorted({o.get('point') for o in outs if o.get('point') is not None})
                sample = [f'{o.get("description") or o["name"]} {o.get("point", "")} @{o["price"]}' for o in outs[:4]]
                print(f'   {b["key"]:14s} {len(outs):3d} outcomes; lines: {pts[:8]}{"..." if len(pts) > 8 else ""}; e.g. {sample}')
    print(f'\nTOTAL credits spent by this probe: {total:.0f} | remaining: {remaining}')


if __name__ == '__main__':
    main()
