"""
dixon_coles_engine.py
======================
Core probability engine for WC7 — a Dixon-Coles bivariate Poisson model
with maximum-likelihood-fitted attack/defense ratings and empirical-Bayes
shrinkage regularization.

METHODOLOGY SUMMARY
--------------------
1. Each team gets two parameters: attack (alpha) and defense (beta).
   Expected goals for a match: lambda_home = alpha_home * beta_away * baseline
                                lambda_away = alpha_away * beta_home * baseline
2. Parameters are fit by maximum likelihood on observed xG (not raw goals —
   xG is a better signal of true performance, goals are noisier).
3. Dixon-Coles rho correction adjusts the joint probability of low-scoring
   outcomes (0-0, 1-0, 0-1, 1-1), which the naive product-of-Poissons
   over/undercounts. rho is negative for soccer (~-0.06 empirically).
4. SHRINKAGE (the most important design decision in this model):
   Raw MLE fits on 3-5 matches per team are wildly overconfident. We shrink
   each team's parameter toward the league average, weighted by sample size:
       shrunk = raw ** (n / (n + K))
   where n = matches played, K = shrinkage strength.
   K_ATTACK = 4.0, K_DEFENSE = 8.0  <- these are DIFFERENT on purpose.

   WHY THE SPLIT: an early build used K=4 for both and got badly burned on
   Portugal vs Spain — Spain's defense rating (built on just 3 matches) was
   so extreme it suppressed Portugal's expected goals to an unrealistic 0.55,
   giving a combined-goals total (1.58) that a liquid real-money market
   (Kalshi) disagreed with by 30+ percentage points. Diagnosis: an extreme
   DEFENSE rating on a tiny sample compounds directly into the OPPONENT's
   expected goals, giving it outsized leverage on totals markets specifically.
   Attack-side shrinkage at K=4 was independently validated (see
   CHANGELOG.md — it correctly caught Norway as live value vs Brazil).
   So: leave attack shrinkage alone, make defense shrinkage more conservative.
   We tested K_DEFENSE up to 50 and confirmed even that wouldn't fully close
   the gap to market pricing — fully closing it would require discarding
   real signal, which is overfitting, not fixing. K=8 was chosen as a
   genuine, moderate correction, not a market-chasing patch.

KNOWN LIMITATIONS (see README.md "Roadmap" for detail):
- No opponent-strength adjustment beyond what MLE implicitly captures.
  A team that blows out a weak side (e.g. beating Qatar 3-1 on xG) gets an
  inflated attack rating even after shrinkage, because MLE with 3-5 games
  per team can't fully separate "genuinely great attack" from "played a
  weak defense once." This has bitten the model at least twice
  (Brazil/Group C, Switzerland/Qatar) — see CHANGELOG.md.
- No live/in-match updating. Pure pre-match model. Cannot see red cards,
  penalties, or momentum shifts coming (see Mexico vs England miss).
- No player-level modeling. Squad state is a single flat multiplier per
  team (see squad_adjustments in fitted_ratings_current.json), not a
  weighted function of which specific players are in the confirmed XI.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from collections import defaultdict
from datetime import date

RHO_DEFAULT = -0.06
K_ATTACK_DEFAULT = 4.0
K_DEFENSE_DEFAULT = 8.0


def dixon_coles_correction(i, j, lambda_home, lambda_away, rho):
    """Adjustment factor for the joint probability of scoreline (i, j)."""
    if i == 0 and j == 0:
        return 1 - lambda_home * lambda_away * rho
    elif i == 1 and j == 0:
        return 1 + lambda_away * rho
    elif i == 0 and j == 1:
        return 1 + lambda_home * rho
    elif i == 1 and j == 1:
        return 1 - rho
    return 1.0


def fit_mle(matches, k_attack=K_ATTACK_DEFAULT, k_defense=K_DEFENSE_DEFAULT,
            time_weighted=False, today=None, half_life_days=None):
    """
    Fit shrinkage-regularized attack/defense ratings via MLE on a list of
    matches: [(home, away, xg_home, xg_away, date_str_optional), ...]

    Returns: dict {team: {"attack": float, "defense": float,
                           "raw_attack": float, "raw_defense": float,
                           "matches": int}}
    """
    teams = sorted(set(t for m in matches for t in m[:2]))
    N = len(teams)
    if N < 4:
        return None  # not enough teams for a stable joint fit

    t2i = {t: i for i, t in enumerate(teams)}
    home_idx = np.array([t2i[m[0]] for m in matches])
    away_idx = np.array([t2i[m[1]] for m in matches])
    xg_home = np.array([m[2] for m in matches])
    xg_away = np.array([m[3] for m in matches])

    if time_weighted and today is not None:
        xi = np.log(2) / half_life_days if half_life_days else 0.006
        weights = np.array([
            np.exp(-xi * (today - date.fromisoformat(m[4])).days) for m in matches
        ])
    else:
        weights = np.ones(len(matches))

    def neg_log_likelihood(params):
        log_alpha = params[:N]
        log_beta = params[N:]
        lam_h = np.exp(log_alpha[home_idx] + log_beta[away_idx])
        lam_a = np.exp(log_alpha[away_idx] + log_beta[home_idx])
        lam_h = np.maximum(lam_h, 1e-6)
        lam_a = np.maximum(lam_a, 1e-6)
        return np.sum(weights * (lam_h - xg_home * np.log(lam_h))) + \
               np.sum(weights * (lam_a - xg_away * np.log(lam_a)))

    def jacobian(params):
        log_alpha = params[:N]
        log_beta = params[N:]
        lam_h = np.exp(log_alpha[home_idx] + log_beta[away_idx])
        lam_a = np.exp(log_alpha[away_idx] + log_beta[home_idx])
        lam_h = np.maximum(lam_h, 1e-6)
        lam_a = np.maximum(lam_a, 1e-6)
        res_h = weights * (lam_h - xg_home)
        res_a = weights * (lam_a - xg_away)
        grad = np.zeros(2 * N)
        np.add.at(grad[:N], home_idx, res_h)
        np.add.at(grad[:N], away_idx, res_a)
        np.add.at(grad[N:], away_idx, res_h)
        np.add.at(grad[N:], home_idx, res_a)
        return grad

    constraints = [
        {'type': 'eq', 'fun': lambda p: np.sum(p[:N])},
        {'type': 'eq', 'fun': lambda p: np.sum(p[N:])},
    ]
    result = minimize(
        neg_log_likelihood, np.zeros(2 * N), jac=jacobian, method='SLSQP',
        constraints=constraints, options={'maxiter': 3000, 'ftol': 1e-12}
    )
    if not result.success:
        print(f"WARNING: MLE optimization did not converge cleanly: {result.message}")

    raw_alpha = np.exp(result.x[:N])
    raw_beta = np.exp(result.x[N:])

    match_count = defaultdict(int)
    for m in matches:
        match_count[m[0]] += 1
        match_count[m[1]] += 1

    ratings = {}
    for t in teams:
        n = match_count[t]
        shrink_a = n / (n + k_attack)
        shrink_b = n / (n + k_defense)
        ratings[t] = {
            "attack": round(float(raw_alpha[t2i[t]] ** shrink_a), 4),
            "defense": round(float(raw_beta[t2i[t]] ** shrink_b), 4),
            "raw_attack": round(float(raw_alpha[t2i[t]]), 4),
            "raw_defense": round(float(raw_beta[t2i[t]]), 4),
            "matches": n,
        }
    return ratings


def score_matrix(ratings, home_team, away_team, baseline_xg, rho=RHO_DEFAULT,
                  home_adj=1.0, away_adj=1.0, max_goals=9):
    """
    Build the full scoreline probability matrix for a matchup.
    home_adj / away_adj: squad-state multipliers (e.g. 0.95 for a missing
    starter) applied directly to that team's attack output.
    Returns: (matrix, lambda_home, lambda_away)
    """
    h, a = ratings[home_team], ratings[away_team]
    lam_h = h['attack'] * home_adj * a['defense'] * baseline_xg
    lam_a = a['attack'] * away_adj * h['defense'] * baseline_xg

    mat = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            mat[i, j] = (poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a) *
                         dixon_coles_correction(i, j, lam_h, lam_a, rho))
    mat /= mat.sum()
    return mat, lam_h, lam_a


def match_probabilities(matrix):
    """Given a scoreline matrix, return (P_home_win, P_draw, P_away_win)."""
    p_win = np.tril(matrix, -1).sum()
    p_draw = np.trace(matrix)
    p_loss = np.triu(matrix, 1).sum()
    return p_win, p_draw, p_loss


def totals_probability(matrix, threshold):
    """P(total goals > threshold), e.g. threshold=2.5 for Over 2.5 market."""
    n = matrix.shape[0]
    return sum(matrix[i, j] for i in range(n) for j in range(n) if i + j > threshold)


def handicap_probability(matrix, threshold, side='home'):
    """P(home wins by more than `threshold` goals), or away if side='away'."""
    n = matrix.shape[0]
    if side == 'home':
        return sum(matrix[i, j] for i in range(n) for j in range(n) if i - j > threshold)
    return sum(matrix[i, j] for i in range(n) for j in range(n) if j - i > threshold)


def marginal_distributions(matrix):
    """Return (home_goals_marginal, away_goals_marginal) as 1D arrays."""
    return matrix.sum(axis=1), matrix.sum(axis=0)


def fit_ridge(matches, ridge=3.0, rho=RHO_DEFAULT, today=None, half_life_days=None):
    """
    Ridge-regularized MLE fit — the 'built right' successor to fit_mle for
    large, unbalanced match graphs (mixed WC + historical opponents).

    Instead of fitting raw ratings and then shrinking with an exponent
    (which happens AFTER the optimizer has already let unanchored
    opponents take extreme values), this adds an L2 penalty on the
    log-ratings directly inside the objective:

        penalty = ridge * sum(log_alpha**2 + log_beta**2)

    Because a team with few matches contributes little to the likelihood,
    the penalty dominates for them and pulls their rating toward 1.0 (log 0)
    automatically — no match-count floor, no post-hoc re-anchoring. Teams
    with many matches have enough likelihood signal to overcome the penalty
    and keep a meaningful rating. This is a MAP estimate with a Normal(0,
    1/sqrt(2*ridge)) prior on log-ratings — i.e. the Bayesian-prior item
    from the roadmap, in its simplest honest form.

    Optional recency weighting: if half_life_days is given, each match's
    contribution to the likelihood (not the ridge penalty) is scaled by
    0.5**(age_days/half_life_days), where age is measured from `today`
    (defaults to the latest match date in the set if not given). A match
    exactly one half-life old counts half as much as one from today. Off
    by default (half_life_days=None) — existing callers get identical
    behavior to before this parameter existed.

    Returns the same dict shape as fit_mle (attack/defense/raw_*/matches),
    already on a mean-1.0 footing, so it plugs straight into score_matrix.
    """
    teams = sorted(set(t for m in matches for t in m[:2]))
    N = len(teams)
    if N < 4:
        return None
    t2i = {t: i for i, t in enumerate(teams)}
    hi = np.array([t2i[m[0]] for m in matches])
    ai = np.array([t2i[m[1]] for m in matches])
    xh = np.array([m[2] for m in matches])
    xa = np.array([m[3] for m in matches])

    if half_life_days is not None:
        dates = [date.fromisoformat(m[4]) for m in matches]
        ref_date = today if today is not None else max(dates)
        age_days = np.array([(ref_date - d).days for d in dates], dtype=float)
        age_days = np.clip(age_days, 0, None)  # future-dated matches (shouldn't happen) get full weight
        w = 0.5 ** (age_days / half_life_days)
    else:
        w = np.ones(len(matches))

    def nll(p):
        la, lb = p[:N], p[N:]
        lam_h = np.maximum(np.exp(la[hi] + lb[ai]), 1e-6)
        lam_a = np.maximum(np.exp(la[ai] + lb[hi]), 1e-6)
        like = np.sum(w * (lam_h - xh * np.log(lam_h))) + \
               np.sum(w * (lam_a - xa * np.log(lam_a)))
        return like + ridge * (np.sum(la ** 2) + np.sum(lb ** 2))

    def jac(p):
        la, lb = p[:N], p[N:]
        lam_h = np.maximum(np.exp(la[hi] + lb[ai]), 1e-6)
        lam_a = np.maximum(np.exp(la[ai] + lb[hi]), 1e-6)
        rh, ra = w * (lam_h - xh), w * (lam_a - xa)
        g = np.zeros(2 * N)
        np.add.at(g[:N], hi, rh)
        np.add.at(g[:N], ai, ra)
        np.add.at(g[N:], ai, rh)
        np.add.at(g[N:], hi, ra)
        g[:N] += 2 * ridge * la
        g[N:] += 2 * ridge * lb
        return g

    res = minimize(nll, np.zeros(2 * N), jac=jac, method='L-BFGS-B',
                   options={'maxiter': 5000})
    if not res.success:
        print(f"WARNING: ridge fit did not converge cleanly: {res.message}")

    # center to exact mean-1.0 footing (penalty makes this nearly true already)
    la, lb = res.x[:N], res.x[N:]
    la -= la.mean(); lb -= lb.mean()
    alpha, beta = np.exp(la), np.exp(lb)

    mc = defaultdict(int)
    for m in matches:
        mc[m[0]] += 1; mc[m[1]] += 1
    return {t: {"attack": round(float(alpha[t2i[t]]), 4),
                "defense": round(float(beta[t2i[t]]), 4),
                "raw_attack": round(float(alpha[t2i[t]]), 4),
                "raw_defense": round(float(beta[t2i[t]]), 4),
                "matches": mc[t]} for t in teams}
