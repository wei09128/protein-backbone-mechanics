#!/usr/bin/env python3
"""Home-advantage test of CDL, CCP4 (before/after 2018) and MechLib v2.

Implements HOME_ADVANTAGE_SPEC_20261112.md v1.1 exactly. The spec hash is checked
before any geometry is read. No refinement, MD or QM is run.

Usage (from the kit folder, inside WSL):
  python home_advantage.py --root /mnt/c/Wei/PBM --cdl /path/to/cdl_database.py \
      --benchmark /path/to/pbm_clean_benchmark_results_v1/structure_mae.tsv --out RESULT_DIR
  python home_advantage.py --selftest      # synthetic end-to-end check, no real data
"""
import argparse, hashlib, json, re, runpy, shlex, sys, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parent
SPEC = KIT / 'HOME_ADVANTAGE_SPEC_20261112.md'
SPEC_SHA = '35f61cfac0aa66a006c438ab874192d2521c5a9ff520ace02b79a98ce56e5002'
FEATURE_SHA = 'e905f8b8202903978fc8d284e50511ab7aa6a9f22b29e86caffd5bce426bd52b'
CDL_SHA = '79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68'
HEADERS_SHA = '25f466e255d6742ff9cfd83eecd92d5769f7bce0ce7e254c02025cb8f71f40d4'
GROUPS_SHA = '33299405ec11d22d0366d58c88324a7f2124313a6b6cfe8700c0896a9efc4348'
COMMITS = {'old': 'a683db684eacb84ead595e3cd97c3f055f19b595',
           'new': '2bc9cc946452592783051b781b496a044d0155a8'}
LIST_SHA = {'old': 'f3b2b8ede5654b1711e556c4136c73dc8026bb1ff4db36a6a49e503c30e4bed8',
            'new': '4fcb5cca58954fb3261fb04f22560e0bad01d829b67357f780362866fcac416f'}
KNOWN_MONOMER_SHA = {('old', 'ALA'): '7297fe387fca4a71057782ec2131d7917c22054eaccd123898ff23c590832e02',
                     ('new', 'ALA'): '7297fe387fca4a71057782ec2131d7917c22054eaccd123898ff23c590832e02',
                     ('old', 'PRO'): '2893af921200bb63da4886e7774ae1e19d4a7aa1d8db855cf1233637c66018eb',
                     ('new', 'PRO'): 'b0185d2cbf3243ce0f394db59cda9922ca5f866b346cc9c9166bc8e224273261'}
# Link targets as recovered in LINK_ANGLE_DIFF.tsv; re-checked against the downloaded files.
LINKS = {'old': {'TRANS': {'ACN': 116.200, 'OCN': 123.000}, 'PTRANS': {'ACN': 116.900, 'OCN': 123.000}},
         'new': {'TRANS': {'ACN': 115.917, 'OCN': 123.469}, 'PTRANS': {'ACN': 118.415, 'OCN': 121.016}}}
ACO_OVERRIDE_NEW = 120.614
NONPGIV = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS',
           'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR'}
RES20 = sorted(NONPGIV | {'ILE', 'VAL', 'GLY', 'PRO'})
B = 36
# term -> (feature column, CDL row index in cctbx cdl_database, plausible CDL median range)
TERMS = {'ACN': ('angle_CaCN', 12, (114.0, 119.0)),
         'OCN': ('angle_O_C_N_next', 14, (121.0, 125.0)),
         'ACO': ('angle_CA_C_O', 10, (119.0, 122.5)),
         'TAU': ('tau_deg', 6, (107.0, 114.0))}
PRIMARY = ['ACN', 'OCN']
PREDICTORS = ['CDL', 'CCP4old', 'CCP4new']
STRATA = ['PHENIX_post', 'PHENIX_pre', 'REFMAC_old', 'REFMAC_new']
BANDS = [(0, 1.0, 'le1.0'), (1.0, 1.5, 'gt1.0_le1.5'), (1.5, 2.0, 'gt1.5_le2.0'), (2.0, 99, 'gt2.0')]
SEED, DRAWS, MIN_JUNCTIONS, MIN_GROUPS, MAD_CUT = 20261112, 10000, 20, 30, 8.0
CONTRASTS = {  # name: (P, Q, home stratum, away stratum)
    'C1': ('CDL', 'CCP4new', 'PHENIX_post', 'REFMAC_new'),
    'C2': ('CCP4new', 'CCP4old', 'REFMAC_new', 'REFMAC_old'),
    'C1_awayPHENIXpre': ('CDL', 'CCP4new', 'PHENIX_post', 'PHENIX_pre'),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(path, expected, label, skip=False):
    got = sha(path)
    if not skip and got != expected:
        sys.exit(f'STOP: {label} hash {got} differs from the frozen {expected}. No silent substitution.')
    return got


# ----------------------------------------------------------------- CIF parsing
def cif_loops(text):
    """Return {block: [(tags, rows)]} for loop_ constructs; '#' lines and ';' fields handled."""
    blocks, block, tags, tokens, in_loop, reading_tags = {}, None, [], [], False, False
    textfield = None

    def flush():
        nonlocal tags, tokens
        if block is not None and tags:
            n = len(tags)
            rows = [tokens[i:i + n] for i in range(0, len(tokens) - len(tokens) % n, n)]
            blocks.setdefault(block, []).append((tags, rows))
        tags, tokens = [], []

    for raw in text.splitlines():
        if textfield is not None:
            if raw.startswith(';'):
                tokens.append(' '.join(textfield)); textfield = None
            else:
                textfield.append(raw)
            continue
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if raw.startswith(';'):
            textfield = [raw[1:]]
            continue
        if line.startswith('data_'):
            flush(); in_loop = False; block = line[5:]; continue
        if line == 'loop_':
            flush(); in_loop, reading_tags = True, True; continue
        if line.startswith('_'):
            if in_loop and reading_tags:
                tags.append(line.split()[0]); continue
            flush(); in_loop = False; continue  # single item outside a loop
        if in_loop:
            reading_tags = False
            try:
                tokens.extend(shlex.split(line, posix=True))
            except ValueError:
                tokens.extend(line.split())
    flush()
    return blocks


def loop_rows(blocks, block, prefix):
    for tags, rows in blocks.get(block, []):
        if tags and tags[0].startswith(prefix):
            names = [t.split('.', 1)[1] for t in tags]
            return [dict(zip(names, r)) for r in rows]
    return []


def link_angles(blocks, link):
    out = {}
    for r in loop_rows(blocks, 'link_' + link, '_chem_link_angle.'):
        a = ((r['atom_1_comp_id'], r['atom_id_1']), (r['atom_2_comp_id'], r['atom_id_2']),
             (r['atom_3_comp_id'], r['atom_id_3']))
        for key, want in (('ACN', (('1', 'CA'), ('1', 'C'), ('2', 'N'))), ('OCN', (('1', 'O'), ('1', 'C'), ('2', 'N')))):
            if a == want or a == want[::-1]:
                out[key] = float(r['value_angle'])
    return out


def deloxt_aco(blocks):
    for r in loop_rows(blocks, 'mod_DEL-OXT', '_chem_mod_angle.'):
        atoms = (r['atom_id_1'], r['atom_id_2'], r['atom_id_3'])
        if r['function'] == 'change' and atoms in (('CA', 'C', 'O'), ('O', 'C', 'CA')):
            return float(r['new_value_angle'])
    return None


def monomer_angles(path, res):
    blocks = cif_loops(Path(path).read_text())
    rows = []
    for b in blocks:
        rows += loop_rows(blocks, b, '_chem_comp_angle.')
    out = {}
    for r in rows:
        if r.get('comp_id') != res:
            continue
        a = (r['atom_id_1'], r['atom_id_2'], r['atom_id_3'])
        if a in (('N', 'CA', 'C'), ('C', 'CA', 'N')):
            out['TAU'] = float(r['value_angle'])
        if a in (('CA', 'C', 'O'), ('O', 'C', 'CA')):
            out['ACO'] = float(r['value_angle'])
    if set(out) != {'TAU', 'ACO'}:
        sys.exit(f'STOP: monomer {res} at {path} lacks N-CA-C or CA-C-O ({out}).')
    return out


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print('downloading', url, flush=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        dest.write_bytes(r.read())


def dictionary_targets(dict_dir, offline=False):
    """Per residue and version: link ACN/OCN (TRANS, PTRANS), monomer tau, effective Ca-C-O."""
    records, hashes = [], {}
    for ver, commit in COMMITS.items():
        base = Path(dict_dir) / ver
        lst = base / 'mon_lib_list.cif'
        if not lst.exists():
            if offline:
                sys.exit(f'STOP: {lst} missing and downloads disabled.')
            fetch(f'https://raw.githubusercontent.com/MonomerLibrary/monomers/{commit}/list/mon_lib_list.cif', lst)
        hashes[f'{ver}/mon_lib_list.cif'] = require(lst, LIST_SHA[ver], f'{ver} mon_lib_list.cif')
        blocks = cif_loops(lst.read_text())
        for link in ('TRANS', 'PTRANS'):
            got = link_angles(blocks, link)
            for k, v in LINKS[ver][link].items():
                if k not in got or abs(got[k] - v) > 1e-3:
                    sys.exit(f'STOP: {ver} {link} {k} parsed {got.get(k)} != recovered {v}')
        aco_mod = deloxt_aco(blocks)
        if ver == 'new' and (aco_mod is None or abs(aco_mod - ACO_OVERRIDE_NEW) > 1e-3):
            sys.exit(f'STOP: new DEL-OXT Ca-C-O override parsed as {aco_mod}')
        if ver == 'old' and aco_mod is not None:
            sys.exit(f'STOP: old DEL-OXT unexpectedly overrides Ca-C-O ({aco_mod})')
        for res in RES20:
            path = base / res[0].lower() / f'{res}.cif'
            if not path.exists():
                if offline:
                    sys.exit(f'STOP: {path} missing and downloads disabled.')
                fetch(f'https://raw.githubusercontent.com/MonomerLibrary/monomers/{commit}/{res[0].lower()}/{res}.cif', path)
            if (ver, res) in KNOWN_MONOMER_SHA:
                hashes[f'{ver}/{res}.cif'] = require(path, KNOWN_MONOMER_SHA[(ver, res)], f'{ver} {res}.cif')
            else:
                hashes[f'{ver}/{res}.cif'] = sha(path)
            m = monomer_angles(path, res)
            records.append(dict(version=ver, res=res, TAU=m['TAU'],
                                ACO=ACO_OVERRIDE_NEW if ver == 'new' else m['ACO'], ACO_monomer=m['ACO'],
                                ACN_TRANS=LINKS[ver]['TRANS']['ACN'], OCN_TRANS=LINKS[ver]['TRANS']['OCN'],
                                ACN_PTRANS=LINKS[ver]['PTRANS']['ACN'], OCN_PTRANS=LINKS[ver]['PTRANS']['OCN']))
    return pd.DataFrame(records), hashes


# ----------------------------------------------------------------- CDL
def catmull_rom_2d(G, g, x, y, x0):
    """Periodic Catmull-Rom on grid G[g, i, j] whose node i sits at x0 + 10 i (as score.py)."""
    u = (np.asarray(x, float) - x0) / 10.0
    v = (np.asarray(y, float) - x0) / 10.0
    i0 = np.floor(u).astype(int); j0 = np.floor(v).astype(int)
    tx, ty = u - i0, v - j0

    def w(t):
        t2, t3 = t * t, t * t * t
        return [0.5 * (-t3 + 2 * t2 - t), 0.5 * (3 * t3 - 5 * t2 + 2),
                0.5 * (-3 * t3 + 4 * t2 + t), 0.5 * (t3 - t2)]
    wx, wy = w(tx), w(ty)
    out = np.zeros(len(u))
    for a in range(4):
        ia = (i0 + a - 1) % B
        for b in range(4):
            out += wx[a] * wy[b] * G[g, ia, (j0 + b - 1) % B]
    return out


def cdl_group(res, nxt):
    g = 'NonPGIV' if res in NONPGIV else 'IleVal' if res in ('ILE', 'VAL') else \
        'Gly' if res == 'GLY' else 'Pro' if res == 'PRO' else None
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
            sys.exit(f'STOP: CDL column {col} for {term} has median {med:.2f}, outside {lo}-{hi}; index map wrong')
        arrays[term] = grid
    return groups, arrays


# ----------------------------------------------------------------- data
def vkey(v, strict=False):
    n = [int(x) for x in re.findall(r'\d+', str(v))][:3]
    if not n or (strict and len(n) < 3):
        return ()
    return tuple(n + [0] * (3 - len(n)))


def strata_labels(H, strict=False):
    H = H.copy()
    H['dep_date'] = pd.to_datetime(H['dep_date'], errors='coerce')
    v = H['refmac_version'].map(lambda x: vkey(x, strict))
    build = pd.to_numeric(H['phenix_build'], errors='coerce')
    ref = H['family'].eq('REFMAC')
    lab = pd.Series(None, index=H.index, dtype=object)
    lab[H['family'].eq('PHENIX') & build.ge(2155)] = 'PHENIX_post'
    lab[H['family'].eq('PHENIX') & build.lt(2155)] = 'PHENIX_pre'
    lab[ref & v.map(lambda k: bool(k) and k <= (5, 8, 189)) & H['dep_date'].ge('2012-01-01')] = 'REFMAC_old'
    lab[ref & v.map(lambda k: bool(k) and k >= (5, 8, 222))] = 'REFMAC_new'
    H['stratum'] = lab
    res = pd.to_numeric(H['resolution'], errors='coerce')
    H['band'] = pd.cut(res, [b[0] for b in BANDS] + [BANDS[-1][1]], labels=[b[2] for b in BANDS], right=True)
    return H


def load_features(path):
    cols = ['pdb_id', 'chain', 'res_idx', 'res_name', 'phi_deg', 'psi_deg', 'omega_measured_deg',
            'is_pro_np1', 'bond_C_N_next'] + [t[0] for t in TERMS.values()]
    head = list(pd.read_csv(path, nrows=0).columns)
    miss = [c for c in cols if c not in head]
    if miss:
        sys.exit(f'STOP: feature file lacks {miss}. Available: {head}')
    df = pd.read_csv(path, usecols=cols, low_memory=False, dtype={'pdb_id': str, 'chain': str, 'res_name': str})
    df['pdb_id'] = df['pdb_id'].str.strip().str.upper()
    df['chain'] = df['chain'].fillna('')
    df['res_name'] = df['res_name'].str.strip().str.upper()
    for c in cols[2:]:
        if c != 'res_name' and c != 'chain':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    n0 = len(df)
    df = df[~df.duplicated(['pdb_id', 'chain', 'res_idx'], keep=False)]
    # next-residue identity: exact next exported index in the same chain (as the closing checks)
    s = df[['pdb_id', 'chain', 'res_idx', 'res_name']].sort_values(['pdb_id', 'chain', 'res_idx'])
    nxt = s.groupby(['pdb_id', 'chain'])[['res_idx', 'res_name']].shift(-1)
    s['next_name'] = nxt['res_name'].where(nxt['res_idx'].eq(s['res_idx'] + 1))
    df = df.join(s['next_name'])
    audit = {'rows_read': n0, 'rows_unique': len(df)}
    keep = (df['omega_measured_deg'].abs() > 150) & df['is_pro_np1'].isin([0, 1])
    keep &= np.isfinite(df['phi_deg']) & np.isfinite(df['psi_deg'])
    keep &= df['res_name'].isin(RES20) & df['next_name'].isin(RES20)
    keep &= df['bond_C_N_next'].between(0, 1.8, inclusive='right')
    agree = df['is_pro_np1'].eq(df['next_name'].eq('PRO').astype(int))
    audit['pro_flag_disagreements_excluded'] = int((keep & ~agree).sum())
    df = df[keep & agree].copy()
    for term, (col, _, _) in TERMS.items():
        for flag in (0, 1):
            m = df['is_pro_np1'] == flag
            med = df.loc[m, col].median()
            mad = (df.loc[m, col] - med).abs().median() * 1.4826
            df.loc[m & ((df[col] - med).abs() > MAD_CUT * mad), col] = np.nan
    audit['rows_scored_context'] = len(df)
    return df, audit


def predict(df, dict_tab, cdl_groups, cdl):
    out = {}
    gmap = {g: i for i, g in enumerate(cdl_groups)}
    gi = np.array([gmap.get(cdl_group(a, b), -1) for a, b in zip(df['res_name'], df['next_name'])])
    ok = gi >= 0
    pro = df['next_name'].eq('PRO').to_numpy()
    for term in TERMS:
        p = np.full(len(df), np.nan)
        p[ok] = catmull_rom_2d(cdl[term], gi[ok], df['phi_deg'].to_numpy()[ok], df['psi_deg'].to_numpy()[ok], -180)
        out[('CDL', term)] = p
    for ver, name in (('old', 'CCP4old'), ('new', 'CCP4new')):
        t = dict_tab[dict_tab['version'] == ver].set_index('res')
        out[(name, 'ACN')] = np.where(pro, t['ACN_PTRANS'].iloc[0], t['ACN_TRANS'].iloc[0])
        out[(name, 'OCN')] = np.where(pro, t['OCN_PTRANS'].iloc[0], t['OCN_TRANS'].iloc[0])
        out[(name, 'TAU')] = df['res_name'].map(t['TAU']).to_numpy(float)
        out[(name, 'ACO')] = df['res_name'].map(t['ACO']).to_numpy(float)
    return out


def structure_mae(df, preds):
    rows = []
    for term, (col, _, _) in TERMS.items():
        obs = df[col].to_numpy(float)
        valid = np.isfinite(obs)
        for p in PREDICTORS:
            valid &= np.isfinite(preds[(p, term)])
        d = df.loc[valid, ['pdb_id']].copy()
        for p in PREDICTORS:
            r = obs[valid] - preds[(p, term)][valid]
            d[p + '|r'] = r
        g = d.groupby('pdb_id')
        n = g.size()
        for p in PREDICTORS:
            r = d[p + '|r']
            mae = r.abs().groupby(d['pdb_id']).mean()
            centred = (r - r.groupby(d['pdb_id']).transform('mean')).abs().groupby(d['pdb_id']).mean()
            rows.append(pd.DataFrame({'pdb_id': mae.index, 'term': term, 'predictor': p, 'mae': mae.values,
                                      'mae_centred': centred.values, 'junctions': n.loc[mae.index].values}))
    s = pd.concat(rows, ignore_index=True)
    return s[s['junctions'] >= MIN_JUNCTIONS]


# ----------------------------------------------------------------- statistics
def did_bootstrap(tab, P, Q, h, a, value='mae', reweight=None, seed=SEED, draws=DRAWS):
    """tab: one row per structure with columns group, stratum, band, P/Q value columns.
    Returns estimate, 95% interval, two-sided bootstrap p, counts. reweight: dict band->weight."""
    w = tab.pivot_table(index=['pdb_id', 'group', 'stratum', 'band'], columns='predictor', values=value,
                        observed=True).reset_index()
    w = w[w['stratum'].isin([h, a]) & w[P].notna() & w[Q].notna()]
    w['D'] = w[P] - w[Q]
    bands = [None] if reweight is None else [b for b, x in reweight.items() if x > 0]
    groups = pd.Index(sorted(w['group'].unique()))
    cells = {}
    for b in bands:
        sub = w if b is None else w[w['band'] == b]
        for s in (h, a):
            m = sub[sub['stratum'] == s].groupby('group')['D'].mean().reindex(groups)
            cells[(b, s)] = (m.notna().to_numpy(float), m.fillna(0).to_numpy())
    ng = {s: w.loc[w['stratum'] == s, 'group'].nunique() for s in (h, a)}
    ns = {s: int((w['stratum'] == s).sum()) for s in (h, a)}
    if min(ng.values()) < MIN_GROUPS:
        return dict(status='not_estimable', groups=ng, structures=ns)

    def estimate(weights):
        total, wsum = 0.0, 0.0
        for b in bands:
            parts = []
            for s in (h, a):
                present, val = cells[(b, s)]
                den = weights @ present
                parts.append(np.where(den > 0, (weights @ val) / np.where(den > 0, den, 1), np.nan))
            est = parts[0] - parts[1]
            share = 1.0 if b is None else reweight[b]
            total = total + share * est
            wsum += share
        return total / wsum

    point = float(estimate(np.ones((1, len(groups))))[0])
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, 250):
        n = min(250, draws - start)
        W = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=n).astype(float)
        samples.append(estimate(W))
    samples = np.concatenate(samples)
    samples = samples[np.isfinite(samples)]
    p = min(1.0, 2 * min((samples <= 0).mean(), (samples >= 0).mean()))
    return dict(status='ok', estimate=point, ci95=np.quantile(samples, [.025, .975]).tolist(),
                ci975=np.quantile(samples, [.0125, .9875]).tolist(), p_boot=float(p),
                valid_draws=int(len(samples)), groups=ng, structures=ns)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [None] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def verdict(r):
    if r.get('status') != 'ok':
        return 'not estimable'
    if r['p_holm'] is not None and r['p_holm'] < 0.05:
        return 'home advantage' if r['estimate'] < 0 else 'reversed'
    return 'not resolved'


def run_contrasts(tab, pooled_bands):
    results = {}
    for name, (P, Q, h, a) in CONTRASTS.items():
        for variant, kw in (('primary', {}), ('centred', {'value': 'mae_centred'}),
                            ('resolution_reweighted', {'reweight': pooled_bands})):
            if name == 'C1_awayPHENIXpre' and variant != 'primary':
                continue
            res = {}
            for term in TERMS:
                print(f'  {name} {variant} {term}', flush=True)
                res[term] = did_bootstrap(tab[tab['term'] == term], P, Q, h, a, **kw)
            prim = [t for t in PRIMARY if res[t].get('status') == 'ok']
            adj = holm([res[t]['p_boot'] for t in prim]) if prim else []
            for t in TERMS:
                if res[t].get('status') == 'ok':
                    res[t]['p_holm'] = adj[prim.index(t)] if t in prim else None
                    res[t]['role'] = 'primary' if t in PRIMARY else 'secondary (no Holm)'
                res[t]['verdict'] = verdict(res[t]) if t in PRIMARY else (
                    'descriptive' if res[t].get('status') == 'ok' else 'not estimable')
            results[f'{name}|{variant}'] = dict(P=P, Q=Q, home=h, away=a, terms=res)
    return results


def run_c3(path):
    s = pd.read_csv(path, sep='\t', dtype={'pdb_id': str, 'component': str})
    s = s[s['subset'].eq('B30_primary')]
    out = {}
    for term, label in (('angle_CaCN', 'ACN'), ('angle_O_C_N_next', 'OCN')):
        d = s[s['term'].eq(term) & s['refinement_stratum'].isin(['PHENIX_post_default', 'REFMAC'])]
        long = pd.concat([pd.DataFrame({'pdb_id': d['pdb_id'] + '|' + d['refinement_stratum'],
                                        'group': d['component'], 'stratum': d['refinement_stratum'],
                                        'band': 'all', 'predictor': p, 'mae': d[c]})
                          for p, c in (('CDL', 'cdl'), ('MechLibV2', 'v2'))])
        out[label] = did_bootstrap(long, 'CDL', 'MechLibV2', 'PHENIX_post_default', 'REFMAC')
    expected = {'ACN': -0.134, 'OCN': -0.192}
    for k, v in expected.items():
        if out[k].get('status') != 'ok' or abs(out[k]['estimate'] - v) > 0.001:
            sys.exit(f'STOP: C3 gate failed for {k}: {out[k]}; expected {v}')
    adj = holm([out[t]['p_boot'] for t in PRIMARY])
    for t, pa in zip(PRIMARY, adj):
        out[t]['p_holm'] = pa
        out[t]['verdict'] = verdict(out[t])
    return {'C3|primary': dict(P='CDL', Q='MechLibV2', home='PHENIX_post (held-out)', away='REFMAC (held-out)',
                               terms=out, gate='reproduced existing program-difference estimates within 0.001')}


# ----------------------------------------------------------------- output
def make_figure(results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [(k, t) for k in results if k.endswith('|primary') for t in TERMS
            if results[k]['terms'].get(t, {}).get('status') == 'ok']
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(rows) + 1.2))
    for i, (k, t) in enumerate(rows):
        r = results[k]['terms'][t]
        col = '#2b6cb0' if t in PRIMARY else '#9aa5b1'
        ax.errorbar(r['estimate'], i, xerr=[[r['estimate'] - r['ci95'][0]], [r['ci95'][1] - r['estimate']]],
                    fmt='o', color=col, ms=4, lw=1)
    ax.axvline(0, color='#555', lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{k.split('|')[0]}  {results[k]['P']} vs {results[k]['Q']}  {t}" for k, t in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('DiD in MAE (deg): negative = better at home')
    ax.set_title('Home advantage of restraint libraries', fontsize=9, loc='left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'HomeAdvantage.{ext}', dpi=200)
    plt.close(fig)


def analyse(df, H, dict_tab, groups_cdl, cdl, out, extra=None):
    preds = predict(df, dict_tab, groups_cdl, cdl)
    smae = structure_mae(df, preds)
    smae = smae.merge(H[['group', 'stratum', 'band', 'resolution', 'family']], left_on='pdb_id', right_index=True)
    smae = smae[smae['stratum'].notna()]
    smae['band'] = smae['band'].astype(str)
    smae.to_csv(out / 'STRUCTURE_MAE.tsv', sep='\t', index=False)
    one = smae[smae['predictor'] == 'CDL'].drop_duplicates(['pdb_id', 'term'])
    pooled = one[one['term'] == 'ACN']['band'].value_counts(normalize=True).to_dict()
    pooled_bands = {b[2]: float(pooled.get(b[2], 0.0)) for b in BANDS}
    cov = one.groupby(['term', 'stratum']).agg(structures=('pdb_id', 'size'), groups=('group', 'nunique'),
                                               junctions=('junctions', 'sum')).reset_index()
    cov.to_csv(out / 'COVERAGE.tsv', sep='\t', index=False)
    tab = (smae.groupby(['predictor', 'stratum', 'term'])
           .apply(lambda g: pd.Series({'mae_group_weighted': g.groupby('group')['mae'].mean().mean(),
                                       'mae_centred_group_weighted': g.groupby('group')['mae_centred'].mean().mean(),
                                       'structures': len(g), 'groups': g['group'].nunique()}))
           .reset_index())
    tab.to_csv(out / 'STRATUM_MAE.tsv', sep='\t', index=False)
    print('Contrasts (10,000 draws each; this takes a while)', flush=True)
    results = run_contrasts(smae, pooled_bands)
    return results, pooled_bands


def main(a):
    spec_hash = require(SPEC, SPEC_SHA, 'spec')
    if a.out.exists():
        sys.exit(f'STOP: {a.out} exists; results directory must be new.')
    root = Path(a.root)
    features = Path(a.features) if a.features else root / 'pbm_altloc_full_rdd_audit_v1' / 'blank_only_geometry.csv.gz'
    hashes = {'spec': spec_hash, 'code': sha(Path(__file__)),
              'features': require(features, FEATURE_SHA, 'feature file'),
              'cdl': require(a.cdl, CDL_SHA, 'CDL database'),
              'headers': require(KIT / 'inputs' / 'headers.tsv', HEADERS_SHA, 'headers'),
              'groups': require(KIT / 'inputs' / 'structure_sequence_groups.tsv', GROUPS_SHA, 'sequence groups'),
              'benchmark_structure_mae': sha(a.benchmark)}
    dict_tab, dhash = dictionary_targets(KIT / 'dictionaries', offline=a.offline)
    hashes['dictionaries'] = dhash
    groups_cdl, cdl = cdl_arrays(a.cdl)
    a.out.mkdir(parents=True)
    dict_tab.to_csv(a.out / 'DICTIONARY_TARGETS.tsv', sep='\t', index=False)
    H = pd.read_csv(KIT / 'inputs' / 'headers.tsv', sep='\t', dtype={'pdb_id': str, 'refmac_version': str}).set_index('pdb_id')
    G = pd.read_csv(KIT / 'inputs' / 'structure_sequence_groups.tsv', sep='\t', dtype=str).set_index('pdb_id')
    H = H.join(G['sequence_group'].rename('group'), how='left')
    if H['group'].isna().any():
        sys.exit('STOP: structures without sequence group')
    df, audit = load_features(features)
    results, pooled = analyse(df, strata_labels(H), dict_tab, groups_cdl, cdl, a.out)
    # strict version parsing sensitivity (primary C2 only, where versions matter)
    Hs = strata_labels(H, strict=True)
    sm = pd.read_csv(a.out / 'STRUCTURE_MAE.tsv', sep='\t', dtype={'pdb_id': str})
    sm = sm.drop(columns=['stratum']).merge(Hs[['stratum']], left_on='pdb_id', right_index=True)
    sm = sm[sm['stratum'].notna()]
    strict = {}
    for name in ('C1', 'C2'):
        P, Q, h, aw = CONTRASTS[name]
        strict[name] = {t: did_bootstrap(sm[sm['term'] == t], P, Q, h, aw) for t in PRIMARY}
    results['strict_version_parsing'] = strict
    results.update(run_c3(a.benchmark))
    lock = dict(spec='HOME_ADVANTAGE_SPEC_20261112.md v1.1', hashes=hashes, seed=SEED, draws=DRAWS,
                min_junctions=MIN_JUNCTIONS, min_groups=MIN_GROUPS, mad_cut=MAD_CUT, feature_audit=audit,
                pooled_band_weights=pooled)
    (a.out / 'RUN_LOCK.json').write_text(json.dumps(lock, indent=2) + '\n')
    (a.out / 'CONTRASTS.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
    make_figure({k: v for k, v in results.items() if 'terms' in v}, a.out)
    print('\nSUMMARY (primary terms)')
    for k, v in results.items():
        if 'terms' in v and k.endswith('|primary'):
            for t in PRIMARY:
                r = v['terms'][t]
                if r.get('status') == 'ok':
                    print(f"{k:28s} {t}: {r['estimate']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] "
                          f"p_holm={r['p_holm']:.3g} -> {r['verdict']}")
                else:
                    print(f'{k:28s} {t}: not estimable {r.get("groups")}')
    print('Return the whole folder:', a.out)


# ----------------------------------------------------------------- self-test
def selftest():
    """Synthetic end-to-end run: a built-in home advantage must be recovered; parser checked on real files."""
    import tempfile
    d = KIT / 'dictionaries'
    blocks = {v: cif_loops((d / v / 'mon_lib_list.cif').read_text()) for v in ('old', 'new')}
    for v in ('old', 'new'):
        for link in ('TRANS', 'PTRANS'):
            got = link_angles(blocks[v], link)
            assert all(abs(got[k] - x) < 1e-3 for k, x in LINKS[v][link].items()), (v, link, got)
    assert abs(deloxt_aco(blocks['new']) - 120.614) < 1e-6 and deloxt_aco(blocks['old']) is None
    ala = monomer_angles(d / 'old' / 'a' / 'ALA.cif', 'ALA')
    assert abs(ala['ACO'] - 117.159) < 1e-6 and abs(ala['TAU'] - 109.627) < 1e-6, ala
    print('PASS parser: links, DEL-OXT override and ALA monomer match the recovered values')
    rng = np.random.default_rng(1)
    # fake dictionaries for all residues: copy ALA values; fake CDL with a phi/psi pattern
    dict_tab = pd.DataFrame([dict(version=v, res=r, TAU=110.0 if v == 'old' else 110.0,
                                  ACO=117.2 if v == 'old' else 120.6, ACO_monomer=117.2,
                                  ACN_TRANS=LINKS[v]['TRANS']['ACN'], OCN_TRANS=LINKS[v]['TRANS']['OCN'],
                                  ACN_PTRANS=LINKS[v]['PTRANS']['ACN'], OCN_PTRANS=LINKS[v]['PTRANS']['OCN'])
                             for v in ('old', 'new') for r in RES20])
    groups_cdl = sorted({cdl_group(r, n) for r in RES20 for n in ('ALA', 'PRO')})
    phi = np.arange(-180, 180, 10)
    pat = np.cos(np.radians(phi))[:, None] * np.ones(B)[None, :]
    base = {'ACN': 116.6, 'OCN': 122.7, 'ACO': 120.5, 'TAU': 111.0}
    cdl = {t: np.repeat((base[t] + 1.5 * pat)[None], len(groups_cdl), 0) for t in TERMS}
    n_struct = 480
    H = pd.DataFrame({'pdb_id': [f'S{i:04d}' for i in range(n_struct)]})
    H['family'] = np.where(np.arange(n_struct) % 4 < 2, 'PHENIX', 'REFMAC')
    H['phenix_build'] = np.where(np.arange(n_struct) % 4 == 0, 2500, np.where(np.arange(n_struct) % 4 == 1, 1800, np.nan))
    H['refmac_version'] = np.where(np.arange(n_struct) % 4 == 2, '5.8.0352', np.where(np.arange(n_struct) % 4 == 3, '5.8.0135', ''))
    H['dep_date'] = '2016-06-01'
    H['resolution'] = rng.uniform(1.0, 2.0, n_struct)
    H['group'] = [f'G{i}' for i in range(n_struct)]
    H = strata_labels(H.set_index('pdb_id'))
    def synth(imprint):
      rows = []
      for sid, h in H.iterrows():
          n = 60
          ph, ps = rng.uniform(-180, 180, n), rng.uniform(-180, 180, n)
          nxt = np.where(rng.random(n) < 0.1, 'PRO', 'ALA')
          true = {t: base[t] + 0.5 * np.cos(np.radians(ph)) for t in TERMS}
          lib = {'PHENIX_post': 'CDL', 'REFMAC_new': 'new', 'REFMAC_old': 'old', 'PHENIX_pre': 'flat'}[h['stratum']]
          geo = {}
          for t in TERMS:
              if lib == 'CDL':
                  target = base[t] + 1.5 * np.cos(np.radians(ph))
              elif lib in ('old', 'new'):
                  r = dict_tab[(dict_tab.version == lib) & (dict_tab.res == 'ALA')].iloc[0]
                  target = {'ACN': np.where(nxt == 'PRO', r.ACN_PTRANS, r.ACN_TRANS),
                            'OCN': np.where(nxt == 'PRO', r.OCN_PTRANS, r.OCN_TRANS),
                            'ACO': np.full(n, r.ACO), 'TAU': np.full(n, r.TAU)}[t]
              else:
                  target = np.full(n, base[t])
              geo[t] = (1 - imprint) * true[t] + imprint * target + rng.normal(0, 0.8, n)
          rows.append(pd.DataFrame({'pdb_id': sid, 'chain': 'A', 'res_idx': np.arange(n), 'res_name': 'ALA',
                                    'next_name': nxt, 'phi_deg': ph, 'psi_deg': ps, 'is_pro_np1': (nxt == 'PRO').astype(int),
                                    **{TERMS[t][0]: geo[t] for t in TERMS}}))
      return pd.concat(rows, ignore_index=True)
    global DRAWS
    DRAWS = 500
    for imprint in (0.7, 0.0):
        with tempfile.TemporaryDirectory() as tmp:
            results, _ = analyse(synth(imprint), H, dict_tab, groups_cdl, cdl, Path(tmp))
        for key in ('C1|primary', 'C2|primary'):
            for t in PRIMARY:
                r = results[key]['terms'][t]
                print(f'imprint {imprint} {key} {t}: {r["estimate"]:+.3f} {r["ci95"]} {r["verdict"]}')
                if imprint:
                    assert r['estimate'] < 0 and r['verdict'] == 'home advantage', (key, t, r)
                else:
                    assert abs(r['estimate']) < 0.05, (key, t, r)
    print('PASS synthetic: home advantage recovered when built in; near zero when absent')
    with tempfile.TemporaryDirectory() as tmp:
        g = np.random.default_rng(2)
        recs = []
        for term, shift in (('angle_CaCN', -0.134), ('angle_O_C_N_next', -0.192)):
            for i in range(200):
                stratum = 'PHENIX_post_default' if i % 2 else 'REFMAC'
                v2 = 1.0 + g.normal(0, 0.05)
                recs.append(dict(pdb_id=f'X{i}', component=f'K{i}', refinement_stratum=stratum, v1=v2, v2=v2,
                                 cdl=v2 + (shift if stratum == 'PHENIX_post_default' else 0.0),
                                 subset='B30_primary', term=term))
        f = Path(tmp) / 'structure_mae.tsv'
        pd.DataFrame(recs).to_csv(f, sep='\t', index=False)
        c3 = run_c3(f)['C3|primary']['terms']
        print('C3 synthetic:', {t: round(c3[t]['estimate'], 4) for t in PRIMARY})
    print('PASS C3 gate logic')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='/mnt/c/Wei/PBM')
    ap.add_argument('--features', help='override path to blank_only_geometry.csv.gz')
    ap.add_argument('--cdl', type=Path, help='cctbx cdl_database.py (sha 79c197e5...)')
    ap.add_argument('--benchmark', type=Path, help='PBM_CLEAN_BENCHMARK_RESULTS_v1 structure_mae.tsv')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--offline', action='store_true', help='do not download monomer files')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not (a.cdl and a.benchmark and a.out):
            ap.error('--cdl, --benchmark and --out are required (or use --selftest)')
        main(a)
