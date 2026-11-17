"""Point selection for the confirmatory QM referee (spec v1.0 section 4). Library predictions and junction counts only; no QM.
Usage: python -I select_points.py V3_SUPPORT_MAP.tsv.gz QM_POINTS_v1_0.tsv QM_POINTS_RHO_v1_0.tsv"""
import sys
import numpy as np
import pandas as pd

TERMS = ['angle_CaCN', 'angle_O_C_N_next', 'angle_CA_C_O']
PILOT = [(-60, -40), (-120, 130), (-75, 145), (60, 40)]
K_MAX, N_MIN, EXCL = 8, 100, 15


def wrap(a):
    return (a + 180) % 360 - 180


d = pd.read_csv(sys.argv[1], sep='\t')
d = d[(d.next_pro == 0) & (d.term.isin(TERMS))]
own = d[d.level_v3 == 'own']
wide = own.pivot_table(index=['residue', 'phi', 'psi'], columns='term', values=['n_v3', 'v2', 'v3'])
wide = wide.dropna()  # own-supported in all three terms
rows, rho_rows = [], []
for res, g in wide.groupby(level=0):
    g = g.droplevel(0)
    ok = [all(max(abs(wrap(p - a)), abs(wrap(q - b))) >= EXCL for a, b in PILOT) for p, q in g.index]
    g = g[ok]
    nmin = g['n_v3'][TERMS].min(axis=1)
    strong = g[nmin >= N_MIN]
    weak = g[nmin < N_MIN].assign(_n=nmin[nmin < N_MIN]).sort_values('_n', ascending=False)
    V = {c: g['v3'][TERMS].loc[c].to_numpy(float) for c in g.index}
    chosen, pool = [], list(strong.index)
    seed = nmin.loc[pool].idxmax() if pool else nmin.idxmax()
    chosen.append(seed)

    def mirror_blocked(c):
        return res == 'GLY' and any((-c[0], -c[1]) == x or (wrap(-c[0]), wrap(-c[1])) == x for x in chosen)

    while len(chosen) < K_MAX:
        cand = [c for c in pool if c not in chosen and not mirror_blocked(c)]
        if not cand:
            break
        best = max(cand, key=lambda c: min(np.linalg.norm(V[c] - V[x]) for x in chosen))
        chosen.append(best)
    for c in weak.index:  # fill rule
        if len(chosen) >= K_MAX:
            break
        if c not in chosen and not mirror_blocked(c):
            chosen.append(c)
    for order, c in enumerate(chosen, 1):
        r = dict(residue=res, order=order, phi=c[0], psi=c[1], min_junctions=int(nmin.loc[c]),
                 population_rule='>=100' if nmin.loc[c] >= N_MIN else 'fill')
        for t in TERMS:
            r[f'v2_{t}'] = round(float(g['v2'][t].loc[c]), 4)
            r[f'v3_{t}'] = round(float(g['v3'][t].loc[c]), 4)
        rows.append(r)
pts = pd.DataFrame(rows)
for k in range(4, K_MAX + 1):
    sub = pts[pts.order <= k]
    for t in TERMS:
        c3 = sub[f'v3_{t}'] - sub.groupby('residue')[f'v3_{t}'].transform('mean')
        c2 = sub[f'v2_{t}'] - sub.groupby('residue')[f'v2_{t}'].transform('mean')
        rho = float((c2 * c3).sum() / (c3 * c3).sum())
        rho_rows.append(dict(k=k, term=t, rho=round(rho, 3), sd_v3_centred=round(float(c3.std()), 3),
                             residues=int(sub.residue.nunique()), points=len(sub)))
pts.to_csv(sys.argv[2], sep='\t', index=False)
rho = pd.DataFrame(rho_rows)
rho.to_csv(sys.argv[3], sep='\t', index=False)
print(pts.groupby('residue').agg(points=('order', 'size'), fill=('population_rule', lambda x: (x == 'fill').sum())).T.to_string())
print(rho.to_string(index=False))
