"""
corners_shots_model.py
========================
Fits shrinkage-regularized attack/defense ratings for corners and shots,
using the SAME methodology as the goals engine (dixon_coles_engine.py) but
on different raw data, since corners/shots aren't part of the xG dataset.

DATA PROVENANCE (important — read before adding new data):
- Corners: sourced from a fan-provided cumulative distribution table
  ("X+ corners" percentages at 2.5/3.5/.../8.5 thresholds, both FOR and
  AGAINST per team) pulled from a betting-stats site. NOT raw match-by-match
  counts. See `reconstruct_mean_from_cdf()` below for how a mean estimate
  was reconstructed from those percentages, and CHANGELOG.md for how the
  reconstruction was calibrated against a second, independent raw-totals
  source to correct for an unstable tail-extrapolation bug found during
  QA (Spain's corners-for mean initially came out to an absurd 12.4 before
  the fix).
- Shots: sourced from FBref's "Standard" shooting table (Sh/90, SoT/90,
  G/Sh, G/SoT — for AND against, i.e. two separate team-stat tables).
  This table does NOT include xG, only raw shot/shot-on-target counts and
  conversion rates. That means the shots model captures VOLUME but not
  SHOT QUALITY — a shot from 6 yards and a shot from 30 yards count the
  same. This is a known, disclosed limitation (see README.md roadmap,
  item "npxG per shot").

Both models use the SAME simple structure (deliberately, for consistency):
    projected_for  = league_avg_for * team_attack_ratio * opponent_defense_ratio
    team_attack_ratio    = shrunk_team_for_rate / league_avg_for
    opponent_defense_ratio = shrunk_opponent_against_rate / league_avg_against
This is a multiplicative Dixon-Coles-style structure using OBSERVED RATES
directly as ratios — NOT a joint MLE fit like the goals engine. That's a
simplification, made because corners/shots don't have the game-theoretic
scoring structure that justifies the extra complexity of a full MLE fit,
and because the underlying data (percentile bins for corners) doesn't
support it cleanly anyway.
"""

from collections import defaultdict


def reconstruct_mean_from_cdf(cdf_percentages, tail_ratio=0.5):
    """
    Reconstruct E[X] from a cumulative "X+ occurs Y% of the time" table.
    cdf_percentages = [P(X>2.5), P(X>3.5), ..., P(X>8.5)] as 0-100 values.

    E[X] = sum_{k=0}^inf P(X>k). We're given P(X>2) through P(X>8) (7 values
    at the .5 thresholds). We assume P(X>0)=P(X>1)=1.0 (near-certain for
    corners — teams essentially always get 2+ in a match), and use a FIXED
    geometric tail (ratio=0.5, not derived from the data) for the
    unobserved P(X>9), P(X>10), ... terms.

    IMPORTANT: an earlier version derived the tail ratio empirically from
    the last two data points (P(X>8)/P(X>7)). This blew up badly whenever
    consecutive bins were equal or noisy (Spain's 30%/30% at the two
    highest thresholds produced an implied E[X] of 12.4, wildly implausible
    for a sport where ~10-11 corners/game is already high). The fixed
    ratio=0.5 assumption is deliberately conservative and stable. See
    CHANGELOG.md for the full incident writeup.
    """
    p = [x / 100.0 for x in cdf_percentages]
    base = 2.0 + sum(p)
    tail = p[6] * tail_ratio / (1 - tail_ratio)
    return base + tail


def fit_rate_model(for_rates, against_rates, matches_played, k=3.0):
    """
    Build shrinkage-regularized attack/defense ratios from raw per-team
    rates (corners/game or shots/90, FOR and AGAINST).

    for_rates / against_rates: {team: raw_rate}
    matches_played: {team: n}
    k: shrinkage constant (equivalent to k "prior" matches at league avg)

    Returns: dict {team: {"attack": ratio, "defense": ratio,
                           "raw_for": .., "raw_against": .., "matches": n}},
             league_avg_for, league_avg_against
    """
    def weighted_avg(rates):
        tot, n = 0, 0
        for t, rate in rates.items():
            m = matches_played.get(t, 3)
            tot += rate * m
            n += m
        return tot / n if n else 0

    league_avg_for = weighted_avg(for_rates)
    league_avg_against = weighted_avg(against_rates)

    ratings = {}
    for t in for_rates:
        raw_for = for_rates[t]
        raw_against = against_rates.get(t, league_avg_against)
        n = matches_played.get(t, 3)
        shrunk_for = (raw_for * n + league_avg_for * k) / (n + k)
        shrunk_against = (raw_against * n + league_avg_against * k) / (n + k)
        ratings[t] = {
            "attack": round(shrunk_for / league_avg_for, 4),
            "defense": round(shrunk_against / league_avg_against, 4),
            "raw_for": round(raw_for, 2),
            "raw_against": round(raw_against, 2),
            "matches": n,
        }
    return ratings, round(league_avg_for, 3), round(league_avg_against, 3)


def project_match(ratings, league_avg_for, home_team, away_team):
    """Projected (home_count, away_count) for corners or shots."""
    h, a = ratings[home_team], ratings[away_team]
    home_val = league_avg_for * h['attack'] * a['defense']
    away_val = league_avg_for * a['attack'] * h['defense']
    return home_val, away_val
