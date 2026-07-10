#!/usr/bin/env python3
"""
refit_v2_recency.py — EXPERIMENTAL comparison model. Same ridge-regularized
Dixon-Coles fit as scripts/refit.py, but with recency weighting: matches
closer to today count more in the fit, older matches count less, on a
365-day half-life (a match from a year ago counts half as much as one from
today).

This is NOT wired into production. It writes to data/data_v2.js, a
completely separate file from data/data.js — the live dashboard
(index.html) never reads it. A separate page, index_v2.html, reads
data_v2.js instead, so both models can be viewed side by side without
either one touching the other.

WHY THIS EXISTS
---------------
A walk-forward backtest (refit-before-each-match-date, true out-of-sample,
scored against real results) showed recency weighting improves calibration:

    No weighting (current production):  Brier 0.4985
    365-day half-life:                  Brier 0.4933  <- best tested
    (90-day half-life was WORSE than no weighting at all — too aggressive,
    discards real signal. 180/545/730-day half-lives were all better than
    no weighting but not as good as 365.)

This is a real, validated improvement on average, across many matches. It
is NOT yet proven safe to trust on any single specific game — recency
weighting can meaningfully redistribute a single match's probabilities
(e.g. it moved Spain's attack rating from 3.124 to 2.604 ahead of their
7/10 quarterfinal, flipping Spain ML from a small positive edge to a
negative one) even when the aggregate metric improves. Run this side by
side against scripts/refit.py's output for a while — several match days,
ideally — before considering promoting it to production.

Everything except the ratings (mle_shrunk), baseline_xg, and calibration
is copied straight from the live data.js — same odds, same squad notes,
same corners/shots data, same everything else. Only the core ratings
methodology differs between the two files.

Run from repo root:  python scripts/refit_v2_recency.py
"""
import json, re, sys, os
import numpy as np
from datetime import datetime, timezone, date
from scipy.optimize import minimize
from scipy.stats import poisson

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from dixon_coles_engine import fit_ridge

ROOT = os.path.join(os.path.dirname(__file__), '..')
XG_DATASET = os.path.join(ROOT, 'data', 'match_xg_dataset.json')
HIST_DATASET = os.path.join(ROOT, 'data', 'historical_dataset.json')
DATA_JS = os.path.join(ROOT, 'data', 'data.js')          # source of truth for odds/squad/etc
DATA_V2_JS = os.path.join(ROOT, 'data', 'data_v2.js')    # this script's own output

HALF_LIFE_DAYS = 365  # best performer in the walk-forward comparison
RHO = -0.06
MAX_G = 9


def load_var_js(path, var_name='D'):
    txt = open(path).read()
    m = re.match(r'\s*var\s+' + var_name + r'\s*=\s*(\{.*\});?\s*$', txt, re.S)
    if not m:
        raise ValueError(f'{path} does not match expected "var {var_name}={{...}};" format')
    return json.loads(m.group(1))


def dixon_coles(i, j, la, lb, rho=RHO):
    if i == 0 and j == 0: return 1 - la * lb * rho
    if i == 1 and j == 0: return 1 + lb * rho
    if i == 0 and j == 1: return 1 + la * rho
    if i == 1 and j == 1: return 1 - rho
    return 1


def outcome_probs(ratings, baseline, h, a):
    if h not in ratings or a not in ratings:
        return None
    rh, ra = ratings[h], ratings[a]
    la = rh['attack'] * ra['defense'] * baseline
    lb = ra['attack'] * rh['defense'] * baseline
    pw = pd = pl = 0.0
    for i in range(MAX_G + 1):
        for j in range(MAX_G + 1):
            v = poisson.pmf(i, la) * poisson.pmf(j, lb) * dixon_coles(i, j, la, lb)
            if i > j: pw += v
            elif i == j: pd += v
            else: pl += v
    tot = pw + pd + pl
    return pw / tot, pd / tot, pl / tot


def fit_calibration(tournament, hist5, today):
    """Same walk-forward Platt-scaling approach as refit.py, but every
    inner fit is also recency-weighted with the same half-life, so the
    calibration is honest for THIS model, not borrowed from the other."""
    tournament_sorted = sorted(tournament, key=lambda d: d['date'])
    records = []
    for m in tournament_sorted:
        train_tourney = [d for d in tournament_sorted if d['date'] < m['date']]
        train_hist = [r for r in hist5 if r[4] < m['date']]
        train5 = ([(d['home'], d['away'], d['xg_home'], d['xg_away'], d['date'])
                   for d in train_tourney] + train_hist)
        if len(train5) < 30:
            continue
        baseline = float(np.mean([t[2] for t in train5] + [t[3] for t in train5]))
        ref_date = date.fromisoformat(m['date'])
        ratings = fit_ridge(train5, ridge=1.0, today=ref_date,
                             half_life_days=HALF_LIFE_DAYS)
        if ratings is None:
            continue
        probs = outcome_probs(ratings, baseline, m['home'], m['away'])
        if probs is None:
            continue
        gh, ga = m.get('goals_home'), m.get('goals_away')
        if gh is None or ga is None:
            continue
        idx = 0 if gh > ga else (1 if gh == ga else 2)
        records.append((*probs, idx))

    if len(records) < 20:
        return 0.0, 1.0, None, None, len(records)

    X, Y = [], []
    for pw, pd, pl, idx in records:
        X += [pw, pd, pl]
        Y += [1 if idx == 0 else 0, 1 if idx == 1 else 0, 1 if idx == 2 else 0]
    X = np.clip(np.array(X), 1e-6, 1 - 1e-6)
    Y = np.array(Y)
    logit_X = np.log(X / (1 - X))

    def nll(params):
        a, b = params
        z = a + b * logit_X
        return np.sum(np.log1p(np.exp(-z * (2 * Y - 1))))

    res = minimize(nll, [0.0, 1.0], method='Nelder-Mead')
    a, b = res.x

    def recal(p):
        z = a + b * np.log(p / (1 - p))
        return 1 / (1 + np.exp(-z))

    briers_before, briers_after = [], []
    for pw, pd, pl, idx in records:
        actual = [1 if idx == 0 else 0, 1 if idx == 1 else 0, 1 if idx == 2 else 0]
        briers_before.append(sum((p - a_) ** 2 for p, a_ in zip([pw, pd, pl], actual)))
        raw = [recal(pw), recal(pd), recal(pl)]
        s = sum(raw)
        pred_after = [r / s for r in raw]
        briers_after.append(sum((p - a_) ** 2 for p, a_ in zip(pred_after, actual)))

    return (round(float(a), 4), round(float(b), 4),
            round(float(np.mean(briers_before)), 4),
            round(float(np.mean(briers_after)), 4), len(records))


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

    merged = xg5 + hist5
    today = date.today()
    baseline = round(float(np.mean([m[2] for m in merged] +
                                   [m[3] for m in merged])), 4)
    ratings = fit_ridge(merged, ridge=1.0, today=today,
                         half_life_days=HALF_LIFE_DAYS)
    if ratings is None:
        raise RuntimeError('fit_ridge returned None — not enough teams?')

    wc_teams = sorted(set(t for d in xg for t in (d['home'], d['away'])))
    ratings_wc = {t: ratings[t] for t in wc_teams if t in ratings}

    cal_a, cal_b, brier_before, brier_after, n_cal = fit_calibration(xg, hist5, today)

    # Start from the live data.js as a template — same odds, squad notes,
    # corners/shots, everything — and only replace the ratings-related keys.
    D2 = load_var_js(DATA_JS, 'D')
    D2['mle_shrunk'] = ratings_wc
    D2['baseline_xg'] = baseline
    D2['calibration'] = {'a': cal_a, 'b': cal_b}
    D2['fit_metadata'] = {
        'model_variant': 'v2_recency_weighted',
        'half_life_days': HALF_LIFE_DAYS,
        'matches_fit': len(merged),
        'tournament_matches': len(xg5),
        'historical_matches': len(hist5),
        'baseline_xg': baseline,
        'method': f'ridge (L2 on log-ratings, ridge=1.0) + {HALF_LIFE_DAYS}-day recency half-life',
        'calibration_method': 'Platt scaling on walk-forward backtest '
                               '(refit-before-each-match-date, true out-of-sample, '
                               'same recency weighting applied inside each inner fit)',
        'calibration_n_matches': n_cal,
        'brier_before_calibration': brier_before,
        'brier_after_calibration': brier_after,
        'validated_vs_baseline': {
            'baseline_brier_no_weighting': 0.4985,
            'this_model_brier': brier_after,
            'note': 'baseline number is from the one-time comparison run on '
                    '2026-07-10; both numbers move as more matches are '
                    'played, compare fresh if reproducing this.'
        },
        'refit_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    D2['last_updated'] = D2['fit_metadata']['refit_at']

    with open(DATA_V2_JS, 'w') as f:
        f.write('var D=' + json.dumps(D2) + ';')
    print(f'[v2/recency] Refit OK: {len(xg5)} tournament + {len(hist5)} historical = '
          f'{len(merged)} matches, half_life={HALF_LIFE_DAYS}d, baseline_xg={baseline}, '
          f'{len(ratings_wc)} WC teams -> data/data_v2.js')
    print(f'[v2/recency] Calibration: a={cal_a}, b={cal_b} on {n_cal} walk-forward matches '
          f'(Brier {brier_before} -> {brier_after})')


if __name__ == '__main__':
    main()
