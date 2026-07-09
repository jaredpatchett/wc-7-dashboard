"""
pk_shootout_model.py
======================
Kick-order Monte Carlo penalty shootout simulator, replacing a naive
flat 50/50 (or flat team-average) shootout probability.

DATA SOURCE: Opta Analyst historical World Cup shootout data — conversion
rate by kick order (kicks 1-3 convert ~71%, kicks 4-5 drop to ~64-67%,
sudden death drops further to ~59.4%), plus each nation's historical
shootout win/loss record used to build a per-team modifier.

WHY THIS MATTERS: a flat 50/50 assumption for "if the match goes to
penalties" hides real, decades-deep signal. Spain has one of the worst
shootout records in World Cup history; some teams (Argentina, Belgium,
Germany, Croatia) are historically excellent. This directly affects
"To Qualify" / "Advance" market pricing on any knockout match that's
otherwise close to even.
"""

import numpy as np

PK_CONVERSION_BY_ORDER = [0.71, 0.71, 0.71, 0.66, 0.64]  # kicks 1 through 5
PK_SUDDEN_DEATH = 0.594

# Historical national shootout tendency, multiplicative modifier vs base
# conversion rate. 1.0 = league average. Source: Opta / Wikipedia shootout
# records for teams remaining as of this package's creation.
NATIONAL_PK_MODIFIER = {
    "Argentina": 1.12, "Belgium": 1.15, "Germany": 1.15, "Croatia": 1.15,
    "Brazil": 1.02, "Mexico": 1.00, "Portugal": 1.05, "France": 1.00,
    "England": 0.85, "Switzerland": 0.75, "Spain": 0.78, "Colombia": 1.00,
    "USA": 1.05, "Egypt": 0.90, "Morocco": 1.02, "Paraguay": 1.02,
    "Australia": 1.10, "Ghana": 0.85, "Norway": 1.00, "Canada": 0.75,
    "Cabo Verde": 0.90,
}


def simulate_shootout(team_a, team_b, n_sims=20000, seed=None):
    """Monte Carlo shootout. Returns P(team_a wins the shootout)."""
    if seed is not None:
        np.random.seed(seed)
    mod_a = NATIONAL_PK_MODIFIER.get(team_a, 1.0)
    mod_b = NATIONAL_PK_MODIFIER.get(team_b, 1.0)
    wins_a = 0
    for _ in range(n_sims):
        score_a, score_b = 0, 0
        for k in range(5):
            pa = min(0.95, PK_CONVERSION_BY_ORDER[k] * mod_a)
            pb = min(0.95, PK_CONVERSION_BY_ORDER[k] * mod_b)
            score_a += int(np.random.random() < pa)
            score_b += int(np.random.random() < pb)
        if score_a != score_b:
            wins_a += int(score_a > score_b)
            continue
        while score_a == score_b:
            pa = min(0.95, PK_SUDDEN_DEATH * mod_a)
            pb = min(0.95, PK_SUDDEN_DEATH * mod_b)
            a_scores = np.random.random() < pa
            b_scores = np.random.random() < pb
            if a_scores != b_scores:
                wins_a += int(a_scores)
                break
    return wins_a / n_sims


def pk_rate_vs_neutral(team, n_sims=10000, seed=None):
    """A team's shootout win rate against a hypothetical average (1.0x) opponent."""
    if seed is not None:
        np.random.seed(seed)
    mod = NATIONAL_PK_MODIFIER.get(team, 1.0)
    wins = 0
    for _ in range(n_sims):
        sa, sb = 0, 0
        for k in range(5):
            pa = min(0.95, PK_CONVERSION_BY_ORDER[k] * mod)
            pb = PK_CONVERSION_BY_ORDER[k]
            sa += int(np.random.random() < pa)
            sb += int(np.random.random() < pb)
        if sa != sb:
            wins += int(sa > sb)
            continue
        while sa == sb:
            pa = min(0.95, PK_SUDDEN_DEATH * mod)
            asc = np.random.random() < pa
            bsc = np.random.random() < PK_SUDDEN_DEATH
            if asc != bsc:
                wins += int(asc)
                break
    return wins / n_sims


def advance_probability(p_win_90, p_draw_90, pk_rate_home, pk_rate_away):
    """
    Full 'To Qualify' probability for a knockout match: win in regulation,
    plus your share of the draw-goes-to-shootout probability weighted by
    relative shootout strength.
    """
    pk_home_norm = pk_rate_home / (pk_rate_home + pk_rate_away)
    pk_away_norm = pk_rate_away / (pk_rate_home + pk_rate_away)
    adv_home = p_win_90 + p_draw_90 * pk_home_norm
    adv_away = (1 - p_win_90 - p_draw_90) + p_draw_90 * pk_away_norm
    return adv_home, adv_away
