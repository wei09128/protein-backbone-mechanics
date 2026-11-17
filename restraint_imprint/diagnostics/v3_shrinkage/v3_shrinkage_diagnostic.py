"""Uncertainty-aware sensitivity analysis of the v3 - v2 correction (exploratory; decides nothing; changes no library).

Input: V3_SUPPORT_MAP.tsv.gz from the frozen v3 build (run pbm_mechlib_v3.s37m8h; 200 sequence-group bootstrap
draws shared by v2 and v3, so se_v3_minus_v2 includes their covariance).
Cells: own-supported in both v2 and v3, se_v3_minus_v2 > 0.
d_k = (v3 - v2)_k centred within residue x next-Pro (pattern part; levels are not what is being scaled).
tau^2 = max(var_k(d) - mean_k(s_k^2), 0)  (method of moments; cells treated as exchangeable).
B_k = tau^2 / (tau^2 + s_k^2): the share of the cell's correction kept by shrinkage toward v2.
Junction weights = n_v3. Cells are not independent observations; this is a sensitivity analysis, not a validated
empirical-Bayes estimate, and shrinkage toward v2 (rather than toward a smooth rescaled pattern) is one choice of target.
Usage: python -I v3_shrinkage_diagnostic.py V3_SUPPORT_MAP.tsv.gz OUT.json
"""
import json
import sys

import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1], sep='\t')
out = {}
for term, g in d.groupby('term'):
    g = g[(g.level_v3 == 'own') & (g.level_v2 == 'own') & (g.se_v3_minus_v2 > 0)].copy()
    g['dc'] = (g.v3 - g.v2) - (g.v3 - g.v2).groupby([g.residue, g.next_pro]).transform('mean')
    s2 = g.se_v3_minus_v2 ** 2
    tau2 = max(g.dc.var() - s2.mean(), 0.0)
    B = tau2 / (tau2 + s2)
    w = g.n_v3
    out[term] = dict(cells=int(len(g)), sd_pattern_diff=round(float(g.dc.std()), 3),
                     median_se_diff=round(float(g.se_v3_minus_v2.median()), 3), tau=round(float(np.sqrt(tau2)), 3),
                     median_B=round(float(B.median()), 2), cells_B_below_0_5_pct=round(float((B < 0.5).mean() * 100), 0),
                     junction_weighted_mean_B=round(float(np.average(B, weights=w)), 2),
                     junctions_in_cells_B_above_0_8_pct=round(float(w[B > 0.8].sum() / w.sum() * 100), 0))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps(out, indent=1))
