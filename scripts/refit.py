#!/usr/bin/env python3
"""
refit.py — refits WC7 ratings on the full match_xg_dataset and rewrites
data/data.js (preserving all non-ratings keys already in it).

Run from repo root:  python scripts/refit.py
Runs automatically in the GitHub Actions workflow.
"""
import json, re, sys, os
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from dixon_coles_engine import fit_mle

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATASET = os.path.join(ROOT, 'data', 'match_xg_dataset.json')
DATA_JS = os.path.join(ROOT, 'data', 'data.js')


def load_data_js(path):
    txt = open(path).read()
    m = re.match(r'\s*var\s+D\s*=\s*(\{.*\});?\s*$', txt, re.S)
    if not m:
        raise ValueError('data.js does not match expected "var D={...};" format')
    return json.loads(m.group(1))


def main():
    with open(DATASET) as f:
        dataset = json.load(f)
    matches = [(d['home'], d['away'], d['xg_home'], d['xg_away'], d['date'])
               for d in dataset]
    baseline = round(float(np.mean([m[2] for m in matches] +
                                   [m[3] for m in matches])), 4)
    ratings = fit_mle(matches)
    if ratings is None:
        raise RuntimeError('fit_mle returned None — not enough teams?')

    D = load_data_js(DATA_JS)
    D['mle_shrunk'] = ratings
    D['baseline_xg'] = baseline
    D['fit_metadata'] = {
        'matches_fit': len(matches),
        'baseline_xg': baseline,
        'refit_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    D['last_updated'] = D['fit_metadata']['refit_at']

    with open(DATA_JS, 'w') as f:
        f.write('var D=' + json.dumps(D) + ';')
    print(f'Refit OK: {len(matches)} matches, baseline_xg={baseline}, '
          f'{len(ratings)} teams -> data/data.js')


if __name__ == '__main__':
    main()
