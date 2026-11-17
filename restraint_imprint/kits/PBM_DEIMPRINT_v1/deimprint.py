"""De-imprinting deposited backbone geometry by calculation (spec DEIMPRINT_SPEC_20261113.md v1.0).

  python deimprint.py --selftest
  python deimprint.py --root /mnt/c/Wei/PBM --cdl /path/to/cdl_database.py --out results
"""
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parent
SPEC = KIT / 'DEIMPRINT_SPEC_20261113.md'
SPEC_SHA = '9b5a9682dd29d9575b84255ef0eea7a8f08fff8ca152de1b93b57d1e9b537e49'
FEATURE_SHA = 'e905f8b8202903978fc8d284e50511ab7aa6a9f22b29e86caffd5bce426bd52b'
CDL_SHA = '79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68'
HEADERS_SHA = '25f466e255d6742ff9cfd83eecd92d5769f7bce0ce7e254c02025cb8f71f40d4'
GROUPS_SHA = '33299405ec11d22d0366d58c88324a7f2124313a6b6cfe8700c0896a9efc4348'
NONPGIV = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS', 'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR'}
RES20 = sorted(NONPGIV | {'ILE', 'VAL', 'GLY', 'PRO'})
B = 36
TERMS = {'ACN': ('angle_CaCN', 12, (114.0, 119.0)),
         'OCN': ('angle_O_C_N_next', 14, (121.0, 125.0)),
         'ACO': ('angle_CA_C_O', 10, (119.0, 122.5))}
MIN_CELL, MIN_CELL_BAND, MAD_CUT = 20, 10, 8.0
SPLIT_SEED, BOOT_SEED, DRAWS, N_SCRAMBLE = 20261115, 20261116, 1000, 20
CONTROLLED_T = {'le1.0': 0.494, 'gt1.0_le1.5': 0.871, 'gt1.5_le2.0': 0.927, 'gt2.0': 0.927}
BANDS3 = [('le1.5', 0, 1.5), ('gt1.5_le2.0', 1.5, 2.0), ('gt2.0', 2.0, 99)]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def require(p, want, what):
    got = sha(p)
    if got != want:
        sys.exit(f'STOP: {what} hash {got[:8]} differs from {want[:8]}')
    return got


# ------------------------------------------------------------------ CDL (as the home-advantage kit)
def catmull(G, g, x, y, x0=-180.0):
    u, v = (np.asarray(x, float) - x0) / 10.0, (np.asarray(y, float) - x0) / 10.0
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    tx, ty = u - i0, v - j0

    def w(t):
        t2, t3 = t * t, t * t * t
        return [0.5 * (-t3 + 2 * t2 - t), 0.5 * (3 * t3 - 5 * t2 + 2), 0.5 * (-3 * t3 + 4 * t2 + t), 0.5 * (t3 - t2)]
    wx, wy = w(tx), w(ty)
    out = np.zeros(len(u))
    for a in range(4):
        for b in range(4):
            out += wx[a] * wy[b] * G[g, (i0 + a - 1) % B, (j0 + b - 1) % B]
    return out


def cdl_group(res, nxt):
    g = 'NonPGIV' if res in NONPGIV else 'IleVal' if res in ('ILE', 'VAL') else 'Gly' if res == 'GLY' else 'Pro' if res == 'PRO' else None
    return None if g is None else g + ('_xpro' if nxt == 'PRO' else '_nonxpro')


def cdl_arrays(path):
    db = runpy.run_path(str(path))['cdl_database']
    if getattr(db, 'version', None) != 'CDL v1.2':
        sys.exit('STOP: unexpected CDL version')
    groups = sorted(db)
    arrays = {}
    for term, (_, col, (lo, hi)) in TERMS.items():
        grid = np.full((len(groups), B, B), np.nan)
        for i, gname in enumerate(groups):
            for (phi, psi), row in db[gname].items():
                if row[col] > 0:
                    grid[i, (phi + 180) // 10, (psi + 180) // 10] = row[col]
        med = np.nanmedian(grid)
        if not lo <= med <= hi:
            sys.exit(f'STOP: CDL column {col} for {term} median {med:.2f} outside {lo}-{hi}')
        arrays[term] = grid
    return groups, arrays


# ------------------------------------------------------------------ data
def load_features(path):
    cols = ['pdb_id', 'chain', 'res_idx', 'res_name', 'phi_deg', 'psi_deg', 'omega_measured_deg',
            'is_pro_np1', 'bond_C_N_next'] + [t[0] for t in TERMS.values()]
    head = list(pd.read_csv(path, nrows=0).columns)
    miss = [c for c in cols if c not in head]
    if miss:
        sys.exit(f'STOP: feature file lacks {miss}')
    df = pd.read_csv(path, usecols=cols, low_memory=False, dtype={'pdb_id': str, 'chain': str, 'res_name': str})
    df['pdb_id'] = df['pdb_id'].str.strip().str.upper()
    df['chain'] = df['chain'].fillna('')
    df['res_name'] = df['res_name'].str.strip().str.upper()
    for c in cols[2:]:
        if c not in ('res_name', 'chain'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[~df.duplicated(['pdb_id', 'chain', 'res_idx'], keep=False)]
    s = df[['pdb_id', 'chain', 'res_idx', 'res_name']].sort_values(['pdb_id', 'chain', 'res_idx'])
    nxt = s.groupby(['pdb_id', 'chain'])[['res_idx', 'res_name']].shift(-1)
    s['next_name'] = nxt['res_name'].where(nxt['res_idx'].eq(s['res_idx'] + 1))
    df = df.join(s['next_name'])
    keep = (df['omega_measured_deg'].abs() > 150) & df['is_pro_np1'].isin([0, 1])
    keep &= np.isfinite(df['phi_deg']) & np.isfinite(df['psi_deg'])
    keep &= df['res_name'].isin(RES20) & df['next_name'].isin(RES20)
    keep &= df['bond_C_N_next'].between(0, 1.8, inclusive='right')
    keep &= df['is_pro_np1'].eq(df['next_name'].eq('PRO').astype(int))
    df = df[keep].copy()
    for term, (col, _, _) in TERMS.items():
        for flag in (0, 1):
            m = df['is_pro_np1'] == flag
            med = df.loc[m, col].median()
            mad = (df.loc[m, col] - med).abs().median() * 1.4826
            df.loc[m & ((df[col] - med).abs() > MAD_CUT * mad), col] = np.nan
    return df


def label(H):
    build = pd.to_numeric(H['phenix_build'], errors='coerce')
    lab = pd.Series(None, index=H.index, dtype=object)
    lab[H['family'].eq('PHENIX') & build.ge(2155)] = 'PHENIX_post'
    lab[H['family'].eq('PHENIX') & build.lt(2155)] = 'PHENIX_pre'
    lab[H['family'].eq('REFMAC')] = 'REFMAC'
    return lab


def halves(components):
    comps = np.array(sorted(set(components)))
    rng = np.random.default_rng(SPLIT_SEED)
    order = comps[rng.permutation(len(comps))]
    half = {c: 'AB'[i % 2] for i, c in enumerate(order)}
    sub = {}
    for h in 'AB':
        hs = [c for c in order if half[c] == h]
        for i, c in enumerate(hs):
            sub[c] = 1 + i % 2
    return half, sub


# ------------------------------------------------------------------ cell aggregation
def cell_table(df, term, cdl_pred_col):
    """Per (component, stratum, cell): sum of angle, sum of CDL prediction, count."""
    col = TERMS[term][0]
    d = df[np.isfinite(df[col]) & np.isfinite(df[cdl_pred_col])]
    g = d.groupby(['component', 'stratum', 'cell'], sort=False)
    t = g.agg(sy=(col, 'sum'), sc=(cdl_pred_col, 'sum'), n=(col, 'size')).reset_index()
    return t


class Prep:
    """Per key: sparse (component x cell) sums of angle, CDL prediction and count, restricted to `cells`."""
    def __init__(self, kt, cells, comps):
        from scipy import sparse
        self.cells = cells; self.comps = comps
        ci = {c: i for i, c in enumerate(cells)}; pi = {c: i for i, c in enumerate(comps)}
        t = kt[kt['cell'].isin(ci)]
        self.grp = np.array([c[0] for c in cells])
        self.M = {}
        for k, x in t.groupby('k'):
            r = x['component'].map(pi).to_numpy(); c = x['cell'].map(ci).to_numpy()
            shape = (len(comps), len(cells))
            self.M[k] = tuple(sparse.csr_matrix((x[v].to_numpy(float), (r, c)), shape=shape) for v in ('sy', 'sc', 'n'))

    def patterns(self, w=None):
        w = np.ones(len(self.comps)) if w is None else w
        out = {}
        for k, (Sy, Sc, N) in self.M.items():
            n = w @ N
            with np.errstate(invalid='ignore', divide='ignore'):
                y = (w @ Sy) / n; c = (w @ Sc) / n
            for gg in np.unique(self.grp):
                m = self.grp == gg
                y[m] -= np.nanmean(y[m]); c[m] -= np.nanmean(c[m])
            out[k] = (y, c)
        return out


def tsls(y, X, Z):
    ok = np.all(np.isfinite(np.column_stack([y, X, Z])), axis=1)
    y, X, Z = y[ok], X[ok], Z[ok]
    return np.linalg.solve(Z.T @ X, Z.T @ y)


def fit_half(pat, h, with_cdl=True):
    """gamma from half h: P_R(h) on [P_P(h, sub2), C(h)] instrumented by [P_P(h, sub1), C(h)]."""
    yR = pat[f'REFMAC|{h}'][0]
    p2, c = pat[f'PHENIX_post|{h}|2']
    p1, _ = pat[f'PHENIX_post|{h}|1']
    _, cP = pat[f'PHENIX_post|{h}']
    if with_cdl:
        return tsls(yR, np.column_stack([p2, cP]), np.column_stack([p1, cP]))
    return tsls(yR, p2[:, None], p1[:, None])


def predict_error(pat, gamma, h, with_cdl=True):
    """Noise-corrected mean squared error of the predicted REFMAC pattern in half h.
    Noise in the PHENIX_post predictor is estimated from its two sub-halves: var(mean) ~ (P1 - P2)^2 / 4."""
    pP, cP = pat[f'PHENIX_post|{h}']
    p1, p2 = pat[f'PHENIX_post|{h}|1'][0], pat[f'PHENIX_post|{h}|2'][0]
    yR = pat[f'REFMAC|{h}'][0]
    pred = gamma[0] * pP + (gamma[1] * cP if with_cdl else 0.0)
    ok = np.isfinite(pred) & np.isfinite(yR) & np.isfinite(p1) & np.isfinite(p2)
    noise = np.mean((p1[ok] - p2[ok]) ** 2) / 4
    return float(np.mean((pred[ok] - yR[ok]) ** 2) - gamma[0] ** 2 * noise)


def keyed(tab, half, sub):
    """Rows keyed as 'stratum|half' and 'stratum|half|sub' and 'stratum'."""
    t = tab.copy()
    t['h'] = t['component'].map(half); t['s'] = t['component'].map(sub).astype(str)
    rows = [t.assign(k=t['stratum'])]
    rows.append(t.assign(k=t['stratum'] + '|' + t['h']))
    rows.append(t.assign(k=t['stratum'] + '|' + t['h'] + '|' + t['s']))
    return pd.concat(rows, ignore_index=True)


def analyse_term(prep, w=None):
    pat = prep.patterns(w)
    gA, gB = fit_half(pat, 'A'), fit_half(pat, 'B')
    g = (gA + gB) / 2
    TP = -g[1] / g[0]; aR = g[0] * (1 - TP)
    E_model = (predict_error(pat, gA, 'B') + predict_error(pat, gB, 'A')) / 2
    kA, kB = fit_half(pat, 'A', False), fit_half(pat, 'B', False)
    E_scaled = (predict_error(pat, kA, 'B', False) + predict_error(pat, kB, 'A', False)) / 2
    return dict(gamma=g.tolist(), T_P=float(TP), T_R=float(1 - aR), alpha_R=float(aR), alpha_P=float(1 - TP),
                E_model=E_model, E_scaled=E_scaled, gain=E_scaled - E_model), pat


def used_cells(tab, half, sub, min_n, strata=('PHENIX_post', 'REFMAC')):
    t = tab.copy(); t['h'] = t['component'].map(half)
    n = t[t['stratum'].isin(strata)].groupby(['cell', 'stratum', 'h'])['n'].sum().unstack(['stratum', 'h'])
    need = [(s, h) for s in strata for h in 'AB']
    n = n.reindex(columns=pd.MultiIndex.from_tuples(need)).fillna(0)
    return sorted(n.index[(n >= min_n).all(axis=1)])


def run(df, cdl_groups, cdl, out=None, draws=DRAWS, scrambles=N_SCRAMBLE):
    gmap = {g: i for i, g in enumerate(cdl_groups)}
    gi = np.array([gmap.get(cdl_group(a, b), -1) for a, b in zip(df['res_name'], df['next_name'])])
    df = df[gi >= 0].copy(); gi = gi[gi >= 0]
    df['cell'] = list(zip(gi, ((df['phi_deg'].to_numpy() + 180) // 10).astype(int) % B, ((df['psi_deg'].to_numpy() + 180) // 10).astype(int) % B))
    half, sub = halves(df['component'])
    res, cells_out, deimp, bands, pre = {}, [], [], {}, {}
    for term in TERMS:
        df['cdl_' + term] = catmull(cdl[term], gi, df['phi_deg'], df['psi_deg'])
        tab = cell_table(df, term, 'cdl_' + term)
        cells = used_cells(tab, half, sub, MIN_CELL)
        kt = keyed(tab, half, sub)
        comps = np.array(sorted(set(df['component'])))
        prep = Prep(kt, cells, comps)
        est, pat = analyse_term(prep)
        # controlled transmission averaged over PHENIX_post resolution mix in used cells
        dP = df[(df['stratum'] == 'PHENIX_post') & df['cell'].isin(set(cells)) & np.isfinite(df[TERMS[term][0]])]
        tc = dP['band4'].map(CONTROLLED_T).astype(float)
        est['T_controlled_mix'] = float(tc.mean())
        est['n_cells'] = len(cells)
        est['n_junctions'] = {s: int(tab[tab['cell'].isin(cells) & (tab['stratum'] == s)]['n'].sum()) for s in ('PHENIX_post', 'REFMAC')}
        # scrambled-CDL null: permute CDL cells within group, recompute predictions
        rng_errs = []
        for seed in range(1, scrambles + 1):
            rng = np.random.default_rng(seed)
            G = cdl[term].copy()
            for k in range(G.shape[0]):
                flat = G[k].ravel(); okc = np.isfinite(flat); vals = flat[okc]
                flat[okc] = vals[rng.permutation(len(vals))]; G[k] = flat.reshape(B, B)
            df['cdl_scr'] = catmull(G, gi, df['phi_deg'], df['psi_deg'])
            t2 = cell_table(df, term, 'cdl_scr')
            e2, _ = analyse_term(Prep(keyed(t2, half, sub), cells, comps))
            rng_errs.append(e2['E_model'])
        est['E_model_scrambled'] = rng_errs
        est['beats_all_scrambles'] = bool(est['E_model'] < min(rng_errs))
        # bootstrap over components (halves fixed)
        rng = np.random.default_rng(BOOT_SEED)
        bd = {'T_P': [], 'T_R': [], 'gain': []}
        for _ in range(draws):
            pick = rng.integers(0, len(comps), len(comps))
            w = np.bincount(pick, minlength=len(comps)).astype(float)
            try:
                e, _ = analyse_term(prep, w)
            except np.linalg.LinAlgError:
                continue
            for k in bd:
                bd[k].append(e[k])
        for k, v in bd.items():
            v = np.array(v); est[k + '_ci95'] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        g = np.array(bd['gain']); est['p_gain_le0'] = float((np.sum(g <= 0) + 1) / (len(g) + 1))
        est['controlled_in_ci'] = bool(est['T_P_ci95'][0] <= est['T_controlled_mix'] <= est['T_P_ci95'][1])
        est['amplification'] = {'R': float(1 / est['alpha_R']) if est['alpha_R'] > 0 else None,
                                'P': float(1 / est['alpha_P']) if est['alpha_P'] > 0 else None}
        est['plausible'] = bool(0 <= est['T_P'] < 1 and 0 <= est['T_R'] < 1)
        res[term] = est
        # de-imprinted pattern from full data
        yR, _ = pat['REFMAC']; yP, cP = pat['PHENIX_post']
        nR = tab[tab['stratum'] == 'REFMAC'].groupby('cell')['n'].sum(); nP = tab[tab['stratum'] == 'PHENIX_post'].groupby('cell')['n'].sum()
        for i, c in enumerate(cells):
            deimp.append(dict(term=term, group=cdl_groups[c[0]], phi_bin=-180 + 10 * c[1], psi_bin=-180 + 10 * c[2],
                              P_REFMAC=yR[i], P_PHENIXpost=yP[i], C=cP[i],
                              D_from_REFMAC=yR[i] / est['alpha_R'] if est['alpha_R'] > 0 else np.nan,
                              D_from_PHENIXpost=(yP[i] - est['T_P'] * cP[i]) / est['alpha_P'] if est['alpha_P'] > 0 else np.nan,
                              n_REFMAC=int(nR.get(c, 0)), n_PHENIXpost=int(nP.get(c, 0))))
        # bands
        bands[term] = {}
        for bname, lo, hi in BANDS3:
            sel = df['resolution'].gt(lo) & df['resolution'].le(hi)
            tb = cell_table(df[sel], term, 'cdl_' + term)
            cb = used_cells(tb, half, sub, MIN_CELL_BAND)
            try:
                eb, _ = analyse_term(Prep(keyed(tb, half, sub), cb, comps)) if len(cb) >= 30 else (None, None)
            except np.linalg.LinAlgError:
                eb = None
            bands[term][bname] = None if eb is None else dict(T_P=eb['T_P'], T_R=eb['T_R'], n_cells=len(cb), gain=eb['gain'])
        # PHENIX_pre secondary: P_pre = r * P_REFMAC
        cpre = used_cells(tab, half, sub, MIN_CELL, strata=('PHENIX_pre', 'REFMAC'))
        pp = Prep(kt, cpre, comps).patterns()
        if all(k in pp for k in ('PHENIX_pre', 'REFMAC|A', 'REFMAC|B')):
            yPre = pp['PHENIX_pre'][0]
            r = tsls(yPre, pp['REFMAC|B'][0][:, None], pp['REFMAC|A'][0][:, None])[0]
            pre[term] = dict(alpha_pre_over_alpha_R=float(r), n_cells=len(cpre))
        print(f"{term}: T_P {est['T_P']:.3f} {est['T_P_ci95']}  T_R {est['T_R']:.3f} {est['T_R_ci95']}  "
              f"controlled {est['T_controlled_mix']:.3f}  gain {est['gain']:+.4f} {est['gain_ci95']}  "
              f"beats scrambles {est['beats_all_scrambles']}  cells {len(cells)}", flush=True)
    # Holm over terms (one-sided bootstrap p for gain > 0)
    ps = sorted((res[t]['p_gain_le0'], t) for t in res)
    m, run_max = len(ps), 0.0
    for i, (p, t) in enumerate(ps):
        run_max = max(run_max, min(1.0, (m - i) * p)); res[t]['p_holm'] = run_max
        res[t]['primary_success'] = bool(run_max < 0.05 and res[t]['gain_ci95'][0] > 0 and res[t]['beats_all_scrambles'])
    if out is not None:
        out.mkdir(parents=True)
        (out / 'PARAMETERS.json').write_text(json.dumps({t: {k: v for k, v in r.items() if k not in ('E_model_scrambled',)} for t, r in res.items()}, indent=2))
        (out / 'PRIMARY_TEST.json').write_text(json.dumps({t: dict(E_model=r['E_model'], E_scaled=r['E_scaled'], gain=r['gain'], gain_ci95=r['gain_ci95'],
                                                                     p_holm=r['p_holm'], E_model_scrambled=r['E_model_scrambled'],
                                                                     beats_all_scrambles=r['beats_all_scrambles'], primary_success=r['primary_success'])
                                                               for t, r in res.items()}, indent=2))
        (out / 'BANDS.json').write_text(json.dumps(dict(bands=bands, phenix_pre=pre), indent=2))
        pd.DataFrame(deimp).to_csv(out / 'DEIMPRINTED_PATTERN.tsv', sep='\t', index=False)
        figure(res, bands, out)
    return res, bands, pre


def figure(res, bands, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, t in enumerate(res):
        r = res[t]
        for j, (k, col) in enumerate((('T_P', '#2a78d6'), ('T_R', '#d0632f'))):
            lo, hi = r[k + '_ci95']
            ax[0].errorbar(i + (j - 0.5) * 0.25, r[k], yerr=[[r[k] - lo], [hi - r[k]]], fmt='o', color=col, label=(['PHENIX post (CDL)', 'REFMAC (flat)'][j] if i == 0 else None))
        ax[0].plot([i - 0.3, i + 0.3], [r['T_controlled_mix']] * 2, 'k--', lw=1, label='controlled (Fig. 1)' if i == 0 else None)
    ax[0].set_xticks(range(len(res))); ax[0].set_xticklabels(list(res)); ax[0].set_ylabel('effective transmission'); ax[0].legend(fontsize=7)
    for i, t in enumerate(res):
        r = res[t]
        ax[1].bar(i - 0.2, r['E_scaled'], 0.4, color='#bbbbbb', label='scaled copy' if i == 0 else None)
        ax[1].bar(i + 0.2, r['E_model'], 0.4, color='#2a78d6', label='transmission model' if i == 0 else None)
        ax[1].scatter([i + 0.2] * len(r['E_model_scrambled']), r['E_model_scrambled'], s=6, color='k', zorder=3, label='scrambled CDL' if i == 0 else None)
    ax[1].set_xticks(range(len(res))); ax[1].set_xticklabels(list(res)); ax[1].set_ylabel('out-of-sample error, REFMAC pattern (°², noise-corrected)'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'DeImprint.png', dpi=200); fig.savefig(out / 'DeImprint.pdf')


def selftest():
    """Synthetic archive with known T_P = 0.6, T_R = 0.3; CDL pattern correlated with, but not equal to, D."""
    rng = np.random.default_rng(7)
    groups = ['NonPGIV_nonxpro']
    xs = np.arange(-180, 180, 10)
    PH, PS = np.meshgrid(xs, xs, indexing='ij')
    Df = lambda f, y: 1.2 * np.sin(np.radians(f)) + 0.8 * np.cos(np.radians(y + 30))
    Cf = lambda f, y: 0.9 * Df(f, y) + 0.6 * np.cos(np.radians(2 * f)) + 0.5 * np.sin(np.radians(2 * y))
    C = Cf(PH, PS)
    cdl = {t: (115 + (C if t == 'ACN' else -C if t == 'OCN' else 0.8 * C))[None].copy() for t in TERMS}
    rows = []
    for s, (TT, L) in {'PHENIX_post': (0.6, True), 'REFMAC': (0.3, False)}.items():
        for k in range(300):
            comp = f'{s[:3]}{k}'
            n = 400
            phi = rng.uniform(-150, -40, n); psi = rng.uniform(-60, 170, n)
            base = (1 - TT) * Df(phi, psi) + (TT * Cf(phi, psi) if L else 0)
            for t, (col, _, _) in TERMS.items():
                sign = 1 if t == 'ACN' else -1 if t == 'OCN' else 0.8
                rows.append(pd.DataFrame({'component': comp, 'stratum': s, 'res_name': 'ALA', 'next_name': 'ALA', 'phi_deg': phi, 'psi_deg': psi,
                                          'resolution': 1.8, 'band4': 'gt1.5_le2.0', 'term': t, 'val': 115 + sign * base + rng.normal(0, 1.5, n)}))
    d = pd.concat(rows)
    df = d.pivot_table(index=['component', 'stratum', 'res_name', 'next_name', 'phi_deg', 'psi_deg', 'resolution', 'band4'], columns='term', values='val').reset_index()
    df = df.rename(columns={t: TERMS[t][0] for t in TERMS})
    res, _, _ = run(df, groups, cdl, None, draws=30, scrambles=3)
    for t, r in res.items():
        assert abs(r['T_P'] - 0.6) < 0.08 and abs(r['T_R'] - 0.3) < 0.08, (t, r['T_P'], r['T_R'])
        assert r['gain'] > 0, (t, r['gain'])
    print('SELFTEST PASS: known transmissions recovered; model beats scaled copy')


def main(a):
    hashes = {'spec': require(SPEC, SPEC_SHA, 'spec'), 'code': sha(Path(__file__))}
    if a.out.exists():
        sys.exit(f'STOP: {a.out} exists')
    root = Path(a.root)
    features = Path(a.features) if a.features else root / 'pbm_altloc_full_rdd_audit_v1' / 'blank_only_geometry.csv.gz'
    hashes['features'] = require(features, FEATURE_SHA, 'feature file')
    hashes['cdl'] = require(a.cdl, CDL_SHA, 'CDL database')
    hashes['headers'] = require(KIT / 'inputs' / 'headers.tsv', HEADERS_SHA, 'headers')
    hashes['groups'] = require(KIT / 'inputs' / 'structure_sequence_groups.tsv', GROUPS_SHA, 'groups')
    cdl_groups, cdl = cdl_arrays(a.cdl)
    H = pd.read_csv(KIT / 'inputs' / 'headers.tsv', sep='\t', dtype={'pdb_id': str, 'refmac_version': str}).set_index('pdb_id')
    G = pd.read_csv(KIT / 'inputs' / 'structure_sequence_groups.tsv', sep='\t', dtype=str).set_index('pdb_id')
    H['stratum'] = label(H)
    H['component'] = G['sequence_group']
    res_ = pd.to_numeric(H['resolution'], errors='coerce')
    H['resolution'] = res_
    H['band4'] = pd.cut(res_, [0, 1.0, 1.5, 2.0, 99], labels=list(CONTROLLED_T), right=True).astype(str)
    df = load_features(features)
    df = df.join(H[['stratum', 'component', 'resolution', 'band4']], on='pdb_id')
    df = df[df['stratum'].isin(['PHENIX_post', 'REFMAC', 'PHENIX_pre']) & df['component'].notna() & df['resolution'].notna()]
    print('junctions by stratum', df['stratum'].value_counts().to_dict(), flush=True)
    res, bands, pre = run(df, cdl_groups, cdl, a.out)
    (a.out / 'RUN_LOCK.json').write_text(json.dumps(dict(spec_sha256=SPEC_SHA, hashes=hashes, split_seed=SPLIT_SEED, boot_seed=BOOT_SEED,
                                                         draws=DRAWS, junctions=df['stratum'].value_counts().to_dict()), indent=2))
    print(json.dumps(dict(bands=bands, phenix_pre=pre), indent=1))
    print('Primary success by term:', {t: r['primary_success'] for t, r in res.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/c/Wei/PBM')
    ap.add_argument('--features')
    ap.add_argument('--cdl', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not (a.cdl and a.out):
            ap.error('--cdl and --out required')
        main(a)
