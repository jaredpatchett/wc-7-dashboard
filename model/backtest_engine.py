"""
backtest_engine.py
====================
Walk-forward backtesting: refits the model at each matchday using ONLY
matches that occurred BEFORE that date, then predicts that day's matches.
This is the only honest way to validate a sports model — using the current
(fully-trained) model to "predict" historical matches is look-ahead bias,
since the current ratings already have those results baked in.

USAGE:
    from dixon_coles_engine import fit_mle, score_matrix, match_probabilities
    from backtest_engine import walk_forward_backtest

    with open('data/match_xg_dataset.json') as f:
        dataset = json.load(f)
    matches = [(d['home'], d['away'], d['xg_home'], d['xg_away'], d['date'])
               for d in dataset]
    results = {(d['home'], d['away']): (d['goals_home'], d['goals_away'])
               for d in dataset if d['goals_home'] is not None}

    predictions = walk_forward_backtest(matches, results)

VALIDATION RESULTS AS OF THIS PACKAGE (see CHANGELOG.md for detail):
- N=70 out-of-sample predictions across the tournament through July 6
- Brier score ~0.19 vs ~0.22 for random 3-way guessing (real but modest edge)
- Modal-outcome accuracy ~55-60% vs ~46% for an "always pick home" baseline
- The model's single highest-probability outcome (H/D/A) was NEVER "Draw"
  across all 70 matches — max Draw probability seen was ~30.5%. This is
  expected Dixon-Coles behavior, not a bug: Draw only becomes the single
  most-likely outcome when two teams are almost perfectly even AND the
  rho-correction boost to low scores is enough to tip it over two separate
  win probabilities. Confirmed by rerunning on the expanded 70-match set
  after initially checking on 56 — same result both times.
- IMPORTANT CAVEAT: this backtest only validates PROBABILITY CALIBRATION.
  It does NOT prove the model would have beaten real betting markets,
  because no historical odds archive exists for most of these matches.
  Only 5 of the 70 backtest entries have verified real odds attached
  (see backtest_results/tracker_seed_data.json) — do not treat the other
  65 "leans" as validated bets, only as directional calibration checks.
"""

import numpy as np
from dixon_coles_engine import fit_mle, score_matrix, match_probabilities


def walk_forward_backtest(matches, actual_results, min_prior_matches=12,
                           k_attack=4.0, k_defense=8.0, baseline_xg=None):
    """
    matches: [(home, away, xg_home, xg_away, date_str), ...] sorted or not
    actual_results: {(home, away): (goals_home, goals_away)}
    min_prior_matches: don't predict until at least this many prior matches
                        exist (too few teams/games -> unstable MLE fit)
    baseline_xg: if None, computed fresh at each matchday from all prior
                 matches' average xG (recommended — keeps it honest/in-sample)

    Returns: list of dicts with date, home, away, pw, pd, pl, outcome
    """
    sorted_matches = sorted(matches, key=lambda m: m[4])
    dates = sorted(set(m[4] for m in sorted_matches))
    predictions = []

    for d in dates:
        prior = [m for m in sorted_matches if m[4] < d]
        todays = [m for m in sorted_matches if m[4] == d]
        if len(prior) < min_prior_matches:
            continue

        avg_xg = baseline_xg if baseline_xg else np.mean(
            [m[2] for m in prior] + [m[3] for m in prior]
        )
        ratings = fit_mle(prior, k_attack=k_attack, k_defense=k_defense)
        if ratings is None:
            continue

        for (hk, ak, xgh, xga, dt) in todays:
            if hk not in ratings or ak not in ratings:
                continue
            actual = actual_results.get((hk, ak))
            if actual is None:
                continue

            mat, lam_h, lam_a = score_matrix(ratings, hk, ak, avg_xg)
            pw, pd, pl = match_probabilities(mat)

            gh, ga = actual
            outcome = 'H' if gh > ga else ('D' if gh == ga else 'A')

            predictions.append({
                "date": d, "home": hk, "away": ak,
                "pw": round(pw, 4), "pd": round(pd, 4), "pl": round(pl, 4),
                "outcome": outcome,
            })

    return predictions


def calibration_report(predictions):
    """Print Brier score, log loss, modal accuracy, and calibration buckets."""
    rows = []
    for p in predictions:
        rows.append((p['pw'], 1 if p['outcome'] == 'H' else 0))
        rows.append((p['pd'], 1 if p['outcome'] == 'D' else 0))
        rows.append((p['pl'], 1 if p['outcome'] == 'A' else 0))
    rows = np.array(rows)
    probs, hits = rows[:, 0], rows[:, 1]

    brier = np.mean((probs - hits) ** 2)
    eps = 1e-10
    log_loss = -np.mean(np.log(np.clip(probs[hits == 1], eps, 1)))

    correct = sum(
        1 for p in predictions
        if max({'H': p['pw'], 'D': p['pd'], 'A': p['pl']},
                key={'H': p['pw'], 'D': p['pd'], 'A': p['pl']}.get) == p['outcome']
    )
    home_baseline = sum(1 for p in predictions if p['outcome'] == 'H') / len(predictions)

    print(f"N = {len(predictions)} predictions")
    print(f"Brier score: {brier:.4f} (random 3-way baseline ~0.222)")
    print(f"Log loss: {log_loss:.4f} (random 3-way baseline ~1.099)")
    print(f"Modal-outcome accuracy: {correct}/{len(predictions)} = {correct/len(predictions)*100:.1f}%")
    print(f"'Always pick home' baseline: {home_baseline*100:.1f}%")

    return {"brier": brier, "log_loss": log_loss,
            "modal_accuracy": correct / len(predictions),
            "home_baseline": home_baseline}
