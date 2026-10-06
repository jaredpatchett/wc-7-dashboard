import re, glob, os, math
import numpy as np
from datetime import date
from scipy.optimize import minimize

ROOT = os.environ.get('OFE_DIR', '/tmp/ofe')  # git clone https://github.com/openfootball/england.git
MONTHS = {m: i + 1 for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}
DATE = re.compile(r'^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$')
MATCH = re.compile(r'^\s*(?:\d{1,2}[:.]\d{2}\s+)?(.+?)\s+v\s+(.+?)\s+(\d+)-(\d+)(?:\s*\([\d\-]+\))?\s*$')
MDAY = re.compile(r'(?:Matchday|Regular Season\s*-)\s*(\d+)')
MATCH_B = re.compile(r'^\s*(?:\d{1,2}[:.]\d{2}\s+)?(.+?)\s{2,}(\d+)-(\d+)(?:\s*\([\d\-]+\))?\s{2,}(\S.*?)\s*$')
ALIAS = {'Brighton & Hove Albion': 'Brighton', 'Wolverhampton Wanderers': 'Wolves', 'Tottenham Hotspur': 'Tottenham',
         'Manchester City': 'Man City', 'Manchester United': 'Man United', 'Newcastle United': 'Newcastle',
         'West Ham United': 'West Ham', 'Nottingham Forest': 'Nottm Forest', 'Leicester City': 'Leicester',
         'Leeds United': 'Leeds', 'Ipswich Town': 'Ipswich', 'Luton Town': 'Luton', 'West Bromwich Albion': 'West Brom',
         'Norwich City': 'Norwich', 'Sheffield United': 'Sheffield Utd', 'Huddersfield Town': 'Huddersfield',
         'Cardiff City': 'Cardiff', 'Swansea City': 'Swansea', 'Stoke City': 'Stoke', 'Hull City': 'Hull',
         'Coventry City': 'Coventry', 'Aston Villa': 'Aston Villa', 'Crystal Palace': 'Crystal Palace'}


def norm(name):
    n = re.sub(r'\b(AFC|FC)\b', '', name).strip()
    n = re.sub(r'\s+', ' ', n)
    return ALIAS.get(n, n)


def parse(path):
    out, year, last_mon, md, cur = [], None, None, None, None
    for raw in open(path, encoding='utf-8', errors='replace'):
        line = raw.rstrip('\r\n')
        h = re.match(r'^=.*?(\d{4})/(\d{2})\s*$', line)
        if h:
            year = int(h.group(1)); continue
        m = MDAY.search(line)
        if m:
            md = int(m.group(1)); continue
        d = DATE.match(line)
        if d:
            mon, day = MONTHS[d.group(1)], int(d.group(2))
            if d.group(3):
                year = int(d.group(3))
            elif last_mon is not None and mon < last_mon:
                year += 1
            last_mon = mon
            cur = date(year, mon, day)
            continue
        mm = MATCH.match(line)
        if mm and cur:
            out.append((norm(mm.group(1)), norm(mm.group(2)), int(mm.group(3)), int(mm.group(4)), cur.isoformat(), md)); continue
        mb = MATCH_B.match(line)
        if mb and cur:
            out.append((norm(mb.group(1)), norm(mb.group(4)), int(mb.group(2)), int(mb.group(3)), cur.isoformat(), md))
    return out


seasons = {}
for p in sorted(glob.glob(f'{ROOT}/20*/1-premierleague.txt')):
    s = os.path.basename(os.path.dirname(p))
    seasons[s] = parse(p)
print('Parsed seasons (matches):', {s: len(v) for s, v in seasons.items() if s >= '2018-19'})
for s_, v_ in seasons.items():
    if '2018-19' <= s_ <= '2025-26':
        assert len(v_) == 380, f'{s_} parsed {len(v_)} != 380'
        assert len(set(m[0] for m in v_)) == 20, f'{s_} has {len(set(m[0] for m in v_))} distinct teams'
print('All 2018-19..2025-26 seasons: 380 matches and 20 distinct teams each -- parse verified')

RHO = -0.06


def dc(i, j, la, lb):
    if i == 0 and j == 0: return 1 - la * lb * RHO
    if i == 1 and j == 0: return 1 + lb * RHO
    if i == 0 and j == 1: return 1 + la * RHO
    if i == 1 and j == 1: return 1 - RHO
    return 1.0


FACT = [math.factorial(k) for k in range(12)]


def probs(la, lb):
    pw = pd = pl = 0.0
    pa = [math.exp(-la) * la ** k / FACT[k] for k in range(11)]
    pb = [math.exp(-lb) * lb ** k / FACT[k] for k in range(11)]
    for i in range(11):
        for j in range(11):
            v = pa[i] * pb[j] * dc(i, j, la, lb)
            if i > j: pw += v
            elif i == j: pd += v
            else: pl += v
    t = pw + pd + pl
    return pw / t, pd / t, pl / t


def fit(matches, ridge=1.0, half_life=None, ref=None):
    teams = sorted(set(t for m in matches for t in m[:2]))
    N = len(teams); idx = {t: i for i, t in enumerate(teams)}
    hi = np.array([idx[m[0]] for m in matches]); ai = np.array([idx[m[1]] for m in matches])
    xh = np.array([m[2] for m in matches], float); xa = np.array([m[3] for m in matches], float)
    if half_life:
        ages = np.array([(ref - date.fromisoformat(m[4])).days for m in matches], float).clip(0)
        w = 0.5 ** (ages / half_life)
    else:
        w = np.ones(len(matches))

    def unpack(p): return p[:N], p[N:2 * N], p[2 * N], p[2 * N + 1]

    def nll(p):
        la, lb, mu, g = unpack(p)
        lh = np.exp(mu + g + la[hi] + lb[ai]); lw = np.exp(mu + la[ai] + lb[hi])
        return np.sum(w * (lh - xh * np.log(lh))) + np.sum(w * (lw - xa * np.log(lw))) + ridge * (np.sum(la ** 2) + np.sum(lb ** 2))

    def jac(p):
        la, lb, mu, g = unpack(p)
        lh = np.exp(mu + g + la[hi] + lb[ai]); lw = np.exp(mu + la[ai] + lb[hi])
        rh, rw = w * (lh - xh), w * (lw - xa)
        gr = np.zeros(2 * N + 2)
        np.add.at(gr[:N], hi, rh); np.add.at(gr[:N], ai, rw)
        np.add.at(gr[N:2 * N], ai, rh); np.add.at(gr[N:2 * N], hi, rw)
        gr[:N] += 2 * ridge * la; gr[N:2 * N] += 2 * ridge * lb
        gr[2 * N] = rh.sum() + rw.sum(); gr[2 * N + 1] = rh.sum()
        return gr

    x0 = np.zeros(2 * N + 2); x0[2 * N] = 0.1
    r = minimize(nll, x0, jac=jac, method='L-BFGS-B', options={'maxiter': 3000})
    la, lb, mu, g = unpack(r.x)
    return {t: (la[idx[t]], lb[idx[t]]) for t in teams}, mu, g


def predict(model, h, a):
    ratings, mu, g = model
    lah, lbh = ratings.get(h, (0.0, 0.0)); laa, lba = ratings.get(a, (0.0, 0.0))
    return probs(math.exp(mu + g + lah + lba), math.exp(mu + laa + lbh))


def walk(season, train_seasons, half_life=None):
    cur = seasons[season]
    mds = sorted(set(m[5] for m in cur if m[5]))
    rows = []
    for k in mds:
        prior = [m for m in cur if m[5] and m[5] < k]
        todays = [m for m in cur if m[5] == k]
        train = [m for s in train_seasons for m in seasons[s]] + prior
        ref = date.fromisoformat(min(m[4] for m in todays)) 
        model = fit(train, half_life=half_life, ref=ref)
        seen = set(model[0].keys())
        for h, a, hs, as_, d, _ in todays:
            p = predict(model, h, a)
            act = (1, 0, 0) if hs > as_ else ((0, 1, 0) if hs == as_ else (0, 0, 1))
            unseen = (h not in seen) or (a not in seen)
            rows.append((k, p, act, unseen))
    return rows


def report(label, rows, flat):
    b = np.mean([sum((pi - ai) ** 2 for pi, ai in zip(p, a)) for _, p, a, _ in rows])
    ll = -np.mean([math.log(max(p[a.index(1)], 1e-9)) for _, p, a, _ in rows])
    acc = np.mean([int(np.argmax(p) == a.index(1)) for _, p, a, _ in rows])
    fb = np.mean([sum((fi - ai) ** 2 for fi, ai in zip(flat, a)) for _, _, a, _ in rows])
    mp = np.mean([p for _, p, _, _ in rows], axis=0); ma = np.mean([a for _, _, a, _ in rows], axis=0)
    print(f'{label:34s} n={len(rows):3d}  Brier={b:.4f} (flat-rate {fb:.4f})  logloss={ll:.4f}  modal acc={acc*100:.1f}%')
    print(f'{"":34s} avg predicted H/D/A = {mp[0]:.3f}/{mp[1]:.3f}/{mp[2]:.3f}   actual = {ma[0]:.3f}/{ma[1]:.3f}/{ma[2]:.3f}')
    return b


for season, trains in [('2026-27', ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']),
                       ('2025-26', ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25'])]:
    trains = [s for s in trains if s in seasons]
    tm = [m for s in trains for m in seasons[s]]
    flat = (np.mean([m[2] > m[3] for m in tm]), np.mean([m[2] == m[3] for m in tm]), np.mean([m[2] < m[3] for m in tm]))
    print(f'\n=== {season} walk-forward by matchday | train seasons {trains[0]}..{trains[-1]} ===')
    for label, hl in [('ridge, no recency', None), ('ridge, 365d half-life', 365)]:
        rows = walk(season, trains, hl)
        report(f'{season} {label}', rows, flat)
        if season == '2026-27' and hl is None:
            unseen_n = sum(1 for r in rows if r[3])
            print(f'{"":34s} matches involving a team with no training history: {unseen_n}')
            per = {}
            for k, p, a, _ in rows:
                per.setdefault(k, []).append(sum((pi - ai) ** 2 for pi, ai in zip(p, a)))
            print(f'{"":34s} Brier by matchday: ' + ', '.join(f'MD{k}={np.mean(v):.3f}' for k, v in sorted(per.items())))
