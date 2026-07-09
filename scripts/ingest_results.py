#!/usr/bin/env python3
"""
ingest_results.py — martj42 international_results ingester for WC7.

Reads the martj42 results.csv (date,home_team,away_team,home_score,
away_score,tournament,city,country,neutral), filters to recent matches
involving >=1 World Cup team, applies name aliases and a per-match weight,
and emits a historical dataset in the SAME tuple shape the engine expects,
plus a per-match weight as a 6th element:

    (home, away, xg_home, xg_away, date, weight)

Since this source is GOALS ONLY (no xG), goals stand in for xG and every
row carries a 0.85 penalty on top of its tournament weight, because goals
are a noisier signal of true performance than xG (per project convention).

USAGE:
    python ingest_results.py results_full.csv historical_dataset.json \
        --xg match_xg_dataset.json
"""
import csv, json, sys, argparse
from collections import Counter

# Canonical WC7 names <- martj42 spellings. Extend as coverage gaps appear.
ALIAS = {
    'United States': 'USA', 'Cape Verde': 'Cabo Verde',
    'Cape Verde Islands': 'Cabo Verde', 'Czech Republic': 'Czechia',
    'Türkiye': 'Turkey', 'Turkiye': 'Turkey',
    'Bosnia and Herzegovina': 'Bosnia', 'Curaçao': 'Curacao',
    'IR Iran': 'Iran', 'Korea Republic': 'South Korea',
    'Korea DPR': 'North Korea', 'China PR': 'China',
    "Côte d'Ivoire": 'Ivory Coast', "Cote d'Ivoire": 'Ivory Coast',
    'DR Congo': 'DR Congo', 'Congo DR': 'DR Congo',
}

# Tournament importance weights. Goals-only penalty (0.85) is applied ON TOP.
GOALS_PENALTY = 0.85
def tournament_weight(t):
    t = t.lower()
    if 'world cup' in t and 'qualification' not in t:
        return 1.0
    if 'uefa nations league' in t or 'qualification' in t:
        return 0.9
    if 'friendly' in t:
        return 0.6
    # continental finals (Euro, Copa, AFCON, etc.)
    if any(k in t for k in ('euro', 'copa', 'african cup', 'gold cup',
                            'asian cup', 'nations cup', 'confederations')):
        return 1.0
    return 0.85


def canon(n):
    return ALIAS.get(n, n)


def load_wc_teams(xg_path):
    ds = json.load(open(xg_path))
    return sorted(set(t for d in ds for t in (d['home'], d['away'])))


def ingest(csv_path, xg_path, since='2023-01-01'):
    wc = set(load_wc_teams(xg_path))
    out, cov = [], Counter()
    for r in csv.DictReader(open(csv_path)):
        if r['date'] < since:
            continue
        if r['home_score'] in ('', 'NA') or r['away_score'] in ('', 'NA'):
            continue  # unplayed / future fixture
        h, a = canon(r['home_team']), canon(r['away_team'])
        if h not in wc and a not in wc:
            continue  # keep opponents of WC teams for schedule strength,
                      # but require at least one WC side
        try:
            gh, ga = float(r['home_score']), float(r['away_score'])
        except ValueError:
            continue
        w = tournament_weight(r['tournament']) * GOALS_PENALTY
        out.append((h, a, gh, ga, r['date'], round(w, 3)))
        if h in wc:
            cov[h] += 1
        if a in wc:
            cov[a] += 1
    zero = sorted(t for t in wc if cov[t] == 0)
    return out, cov, zero, sorted(wc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('out')
    ap.add_argument('--xg', default='match_xg_dataset.json')
    ap.add_argument('--since', default='2023-01-01')
    args = ap.parse_args()

    rows, cov, zero, wc = ingest(args.csv, args.xg, args.since)
    json.dump([list(r) for r in rows], open(args.out, 'w'))

    counts = sorted(cov.values())
    print(f'Ingested {len(rows)} historical matches (>= {args.since}) '
          f'involving a WC team -> {args.out}')
    print(f'Coverage per WC team: min={counts[0]} '
          f'median={counts[len(counts)//2]} max={counts[-1]}')
    if zero:
        print(f'!! ZERO-COVERAGE WC TEAMS (fix ALIAS): {zero}')
    else:
        print('All WC teams have coverage. No alias gaps.')


if __name__ == '__main__':
    main()
