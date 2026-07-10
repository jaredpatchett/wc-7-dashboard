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
from scipy.optimize import minimize
from scipy.stats import poisson

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from dixon_coles_engine import fit_ridge

ROOT = os.path.join(os.path.dirname(__file__), '..')
XG_DATASET = os.path.join(ROOT, 'data', 'match_xg_dataset.json')
HIST_DATASET = os.path.join(ROOT, 'data', 'historical_dataset.json')
DATA_JS = os.path.join(ROOT, 'data', 'data.js')

RHO = -0.06
MAX_G = 9


def load_data_js(path):
    txt = open(path).read()
    m = re.match(r'\s*var\s+D\s*=\s*(\{.*\});?\s*$', txt, re.S)
    if not m:
        raise ValueError('data.js does not match expected "var D={...};" format')
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


def fit_calibration(tournament, hist5):
    """
    True walk-forward backtest: for each tournament match, refit ratings
    using ONLY data strictly before that match's date, then score the
    prediction against the real result. Pooled across all matches, fits a
    2-parameter Platt (logistic) recalibration — corrects any systematic
    over/underconfidence found in the raw model without needing a
    parametric assumption about WHY it's biased.

    Returns (a, b, brier_before, brier_after, n_matches). Falls back to
    identity calibration (a=0, b=1) if there isn't enough data yet.
    """
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
        ratings = fit_ridge(train5, ridge=1.0)
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

    cal_a, cal_b, brier_before, brier_after, n_cal = fit_calibration(xg, hist5)

    D = load_data_js(DATA_JS)
    D['mle_shrunk'] = ratings_wc
    D['baseline_xg'] = baseline
    D['calibration'] = {'a': cal_a, 'b': cal_b}
    D['fit_metadata'] = {
        'matches_fit': len(merged),
        'tournament_matches': len(xg5),
        'historical_matches': len(hist5),
        'baseline_xg': baseline,
        'method': 'ridge (L2 on log-ratings, ridge=1.0)',
        'calibration_method': 'Platt scaling on walk-forward backtest '
                               '(refit-before-each-match-date, true out-of-sample)',
        'calibration_n_matches': n_cal,
        'brier_before_calibration': brier_before,
        'brier_after_calibration': brier_after,
        'refit_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    D['last_updated'] = D['fit_metadata']['refit_at']

    with open(DATA_JS, 'w') as f:
        f.write('var D=' + json.dumps(D) + ';')
    print(f'Refit OK: {len(xg5)} tournament + {len(hist5)} historical = '
          f'{len(merged)} matches, baseline_xg={baseline}, '
          f'{len(ratings_wc)} WC teams -> data/data.js')
    print(f'Calibration: a={cal_a}, b={cal_b} on {n_cal} walk-forward matches '
          f'(Brier {brier_before} -> {brier_after})')


if __name__ == '__main__':
    main()
