#!/usr/bin/env python3
"""
refit.py — refits WC7 ratings using tournament xG + expanded historical
results, with ridge-regularized MLE, and rewrites data/data.js.

WHY THIS VERSION EXISTS
------------------------
The original refit.py fit ONLY on the 94 in-tournament xG matches (3-5 per
team). With that little data, MLE can't separate "genuinely strong" from
"beat one weak opponent once" — it produces large, misleading edges that
are really schedule-strength noise, almost always favoring whichever team
happened to have a weak group.

This version merges in ~1,700 real international results since 2023
(sourced from the martj42/international_results dataset, goals-only, at a
reduced weight vs actual tournament xG) and fits with a ridge-regularized
MLE (L2 penalty on log-ratings) instead of the old post-hoc shrinkage
exponent. Thinly-connected teams get pulled toward average DURING the fit,
not patched after — no arbitrary match-count floor needed.

Run from repo root:  python scripts/refit.py
Runs automatically in the GitHub Actions workflow.
"""
import json, re, sys, os
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from dixon_coles_engine import fit_ridge

ROOT = os.path.join(os.path.dirname(__file__), '..')
XG_DATASET = os.path.join(ROOT, 'data', 'match_xg_dataset.json')
HIST_DATASET = os.path.join(ROOT, 'data', 'historical_dataset.json')
DATA_JS = os.path.join(ROOT, 'data', 'data.js')


def load_data_js(path):
    txt = open(path).read()
    m = re.match(r'\s*var\s+D\s*=\s*(\{.*\});?\s*$', txt, re.S)
    if not m:
        raise ValueError('data.js does not match expected "var D={...};" format')
    return json.loads(m.group(1))


def main():
    with open(XG_DATASET) as f:
        xg = json.load(f)
    xg5 = [(d['home'], d['away'], d['xg_home'], d['xg_away'], d['date'])
           for d in xg]

    hist5 = []
    if os.path.exists(HIST_DATASET):
        with open(HIST_DATASET) as f:
            hist = json.load(f)
        seen = {(r[0], r[1], r[4]) for r in xg5}
        hist5 = [(r[0], r[1], r[2], r[3], r[4]) for r in hist
                 if (r[0], r[1], r[4]) not in seen]
    else:
        print('WARNING: no historical_dataset.json found — fitting on '
              'tournament data only (thin-sample regime). Run '
              'scripts/ingest_results.py first for the full fix.')

    merged = xg5 + hist5
    baseline = round(float(np.mean([m[2] for m in merged] +
                                   [m[3] for m in merged])), 4)
    ratings = fit_ridge(merged, ridge=1.0)
    if ratings is None:
        raise RuntimeError('fit_ridge returned None — not enough teams?')

    # only keep WC teams in the exported ratings (historical data drags in
    # non-WC opponents used purely as anchors during the fit)
    wc_teams = sorted(set(t for d in xg for t in (d['home'], d['away'])))
    ratings_wc = {t: ratings[t] for t in wc_teams if t in ratings}

    D = load_data_js(DATA_JS)
    D['mle_shrunk'] = ratings_wc
    D['baseline_xg'] = baseline
    D['fit_metadata'] = {
        'matches_fit': len(merged),
        'tournament_matches': len(xg5),
        'historical_matches': len(hist5),
        'baseline_xg': baseline,
        'method': 'ridge (L2 on log-ratings, ridge=1.0)',
        'refit_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    D['last_updated'] = D['fit_metadata']['refit_at']

    with open(DATA_JS, 'w') as f:
        f.write('var D=' + json.dumps(D) + ';')
    print(f'Refit OK: {len(xg5)} tournament + {len(hist5)} historical = '
          f'{len(merged)} matches, baseline_xg={baseline}, '
          f'{len(ratings_wc)} WC teams -> data/data.js')


if __name__ == '__main__':
    main()
