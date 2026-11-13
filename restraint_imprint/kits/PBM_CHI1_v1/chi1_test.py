#!/usr/bin/env python3
"""chi1-conditioned MechLib v2: build and provenance-neutral test (spec v1.1).

Steps: (A) rebuild v2 tau / N-CA-CB / C-CA-CB grids and require exact equality with the
frozen arrays; (B) extract chi1 for training residues from pdb_cache with a same-atom tau
gate; (C) extract held-out tau, N-CA-CB, C-CA-CB and chi1 from the frozen coordinates with
a same-atom Ca-C-N / O-C-N gate; (D) fit chi1 surfaces; (E) score on held-out deposits.
No refinement, MD or QM. Usage:
  python chi1_test.py --root /mnt/c/Wei/PBM --cdl /path/cdl_database.py --out RESULT_DIR
  python chi1_test.py --selftest
"""
import argparse, gzip, hashlib, json, runpy, shlex, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parent
SPEC = KIT / 'CHI1_FAIR_COMPARISON_SPEC_20261111.md'
SPEC_SHA = 'bd10bab8000f9c790083ee2b233488003cd627a5381c5f2793fceb7135fbc607'
H = {  # frozen input hashes
    'train_geometry': 'a2929836e8f23fc6d5109b1423baea24952ae9cb98e492bc64a8c25c3765300f',
    'targets': 'ef7b86b359c84492a0be791de52a1e64112a0153142b9f9f5a0417448d2731b1',
    'training_chains': '35f69bd270b9f6ca274af6c1dfedcc83adc97aa38f82e6104140d99ee535e494',
    'junctions': '8b3ccb9ee7481288c336bae9aafff8411f207a87c6cb3ecba45bcad30f110cf6',
    'metadata': '2e1a820d7e6165d4c838e77d5a999b7205bf71814115065a04e61df055cbfa92',
    'strata': '34525a769eb6b273f3bb1a0efcf964b9ceaaba61fa372dee29d08fc6b1c6bd18',
    'cdl': '79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68',
}
DEFAULTS = {
    'train_geometry': 'pbm_altloc_training_audit_results_v1/blank_only_geometry.csv.gz',
    'targets': 'PBM_CLEAN_REBUILD_v1/results/training_targets.npz',
    'junctions': 'pbm_v2_geometry_qc_v1/junctions.tsv',
    'metadata': 'pbm_v2_geometry_metadata_v1/junction_metadata.tsv',
    'strata': 'PBM_V2_HELDOUT_SCORE/refinement_strata.tsv',
    'train_coords': 'pdb_cache', 'heldout_coords': 'pbm_v2_frozen_coordinates',
}
NONPGIV = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS',
           'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR'}
RES20 = sorted(NONPGIV | {'ILE', 'VAL', 'GLY', 'PRO'})
AA = set(RES20)
GAMMA = {'CYS': 'SG', 'SER': 'OG', 'THR': 'OG1', 'ILE': 'CG1', 'VAL': 'CG1'}
NO_CHI1 = {'GLY', 'ALA', 'PRO'}
WELLS = np.array([-60.0, 180.0, 60.0])  # g-, t, g+
CLASSES = ['g-', 't', 'g+']
ECLIPSE_DEG = 40.0
B, MIN_COUNT, TOL, GATE_COVERAGE, MIN_STRUCT_JUNCTIONS = 36, 10, 0.01, 0.75, 5
SEED, DRAWS = 20261111, 10000
PRIMARY = {'TAU': 'tau_deg', 'NCB': 'angle_N_CA_CB', 'CCB': 'angle_C_CA_CB'}
CONTROLS = {'ACN': 'angle_CaCN', 'OCN': 'angle_O_C_N_next'}
CDL_COL = {'TAU': 6, 'NCB': 4, 'CCB': 8, 'ACN': 12, 'OCN': 14}
CDL_NAME = {'TAU': 'mNAC', 'NCB': 'mNAB', 'CCB': 'mBAC', 'ACN': 'mACN', 'OCN': 'mOCN'}
CDL_RANGE = {'TAU': (107, 114), 'NCB': (107, 114), 'CCB': (107, 114), 'ACN': (114, 119), 'OCN': (121, 125)}
STRATA = {'REFMAC': ['REFMAC'], 'REFMAC+SHELXL': ['REFMAC', 'SHELXL'], 'PHENIX_post': ['PHENIX_post_default']}
KEY = ['pdb_id', 'component', 'chain', 'seq_i', 'seq_next']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(path, name, skip=False):
    got = sha(path)
    if not skip and got != H[name]:
        sys.exit(f'STOP: {name} ({path}) hash {got} differs from frozen {H[name]}')
    return got


# ------------------------------------------------------------------ geometry maths
def angle(a, b, c):
    u, v = a - b, c - b
    cosv = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def chi1_class(chi):
    if not np.isfinite(chi):
        return -1
    d = np.abs(((chi - WELLS) + 180.0) % 360.0 - 180.0)
    k = int(np.argmin(d))
    return k if d[k] <= ECLIPSE_DEG else -1


# ------------------------------------------------------------------ coordinate parsing
def residues_from_pdb(text):
    """{chain: [(resname, atoms{name: xyz})]} from PDB text; first model; blank altloc only."""
    chains, order, seen = {}, {}, set()
    for line in text.splitlines():
        rec = line[:6]
        if rec.startswith('ENDMDL'):
            break
        if rec not in ('ATOM  ', 'HETATM') or len(line) < 54:
            continue
        if line[16] != ' ':
            continue
        name, resn, ch = line[12:16].strip(), line[17:20].strip(), line[21]
        key = (ch, line[22:26], line[26], resn)
        if key not in seen:
            seen.add(key); chains.setdefault(ch, []).append((resn, {})); order[key] = len(chains[ch]) - 1
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        chains[ch][order[key]][1][name] = xyz
    return {'auth': chains}


def residues_from_cif(text):
    """{'auth': {...}, 'label': {...}} from mmCIF atom_site; first model; blank altloc only."""
    lines = text.splitlines()
    i, tags = 0, []
    while i < len(lines):
        if lines[i].strip() == 'loop_' and i + 1 < len(lines) and lines[i + 1].startswith('_atom_site.'):
            i += 1
            while i < len(lines) and lines[i].startswith('_atom_site.'):
                tags.append(lines[i].strip().split('.', 1)[1]); i += 1
            break
        i += 1
    if not tags:
        return {'auth': {}, 'label': {}}
    col = {t: k for k, t in enumerate(tags)}
    need = ['group_PDB', 'label_atom_id', 'label_alt_id', 'label_comp_id', 'Cartn_x', 'Cartn_y', 'Cartn_z']
    if any(n not in col for n in need):
        return {'auth': {}, 'label': {}}
    out = {'auth': {}, 'label': {}}
    order = {'auth': {}, 'label': {}}
    model0 = None
    for line in lines[i:]:
        s = line.strip()
        if not s or s.startswith('#') or s.startswith('loop_') or s.startswith('_'):
            if s.startswith('loop_') or s.startswith('_') or s.startswith('#'):
                break
            continue
        tok = shlex.split(s) if ('"' in s or "'" in s) else s.split()
        if len(tok) != len(tags):
            continue
        if 'pdbx_PDB_model_num' in col:
            m = tok[col['pdbx_PDB_model_num']]
            model0 = m if model0 is None else model0
            if m != model0:
                break
        if tok[col['group_PDB']] not in ('ATOM', 'HETATM') or tok[col['label_alt_id']] not in ('.', '?'):
            continue
        name, resn = tok[col['label_atom_id']].strip('"'), tok[col['label_comp_id']]
        try:
            xyz = np.array([float(tok[col['Cartn_x']]), float(tok[col['Cartn_y']]), float(tok[col['Cartn_z']])])
        except ValueError:
            continue
        for kind, chk, seqk in (('auth', 'auth_asym_id', 'auth_seq_id'), ('label', 'label_asym_id', 'label_seq_id')):
            if chk not in col:
                continue
            ch = tok[col[chk]]
            ins = tok[col['pdbx_PDB_ins_code']] if 'pdbx_PDB_ins_code' in col else '?'
            key = (ch, tok[col[seqk]] if seqk in col else '?', ins, resn)
            od = order[kind]
            if key not in od:
                out[kind].setdefault(ch, []).append((resn, {})); od[key] = len(out[kind][ch]) - 1
            out[kind][ch][od[key]][1][name] = xyz
    return out


def read_text(path):
    p = Path(path)
    if p.suffix == '.gz':
        with gzip.open(p, 'rt', errors='replace') as f:
            return f.read()
    return p.read_text(errors='replace')


def residue_values(resn, at):
    """tau, N-CA-CB, C-CA-CB and chi1 from blank atoms (nan if atoms missing)."""
    tau = angle(at['N'], at['CA'], at['C']) if {'N', 'CA', 'C'} <= at.keys() else np.nan
    ncb = angle(at['N'], at['CA'], at['CB']) if {'N', 'CA', 'CB'} <= at.keys() else np.nan
    ccb = angle(at['C'], at['CA'], at['CB']) if {'C', 'CA', 'CB'} <= at.keys() else np.nan
    g = GAMMA.get(resn, 'CG')
    chi = np.nan
    if resn not in NO_CHI1 and {'N', 'CA', 'CB', g} <= at.keys():
        chi = dihedral(at['N'], at['CA'], at['CB'], at[g])
    return tau, ncb, ccb, chi


# ------------------------------------------------------------------ workers
def train_worker(args):
    """Match training rows of one structure to coordinates by chain, residue name and tau."""
    path, rows = args  # rows: list of (row_index, chain, res_name, tau)
    out = []
    try:
        chains = residues_from_pdb(read_text(path))['auth']
    except Exception:
        return out
    cache = {}
    for idx, ch, resn, tau in rows:
        if ch not in cache:
            vals = [(r, residue_values(r, at)) for r, at in chains.get(ch, [])]
            cache[ch] = vals
        cands = [v for r, v in cache[ch] if r == resn and np.isfinite(v[0]) and abs(v[0] - tau) < TOL]
        if len(cands) == 1:
            out.append((idx, cands[0][3]))
    return out


def heldout_worker(args):
    """Match junction rows of one structure by chain, residue names, Ca-C-N and O-C-N."""
    path, rows = args  # rows: (row_index, chain, res_i, res_next, cacn, ocn)
    out = []
    try:
        kinds = residues_from_cif(read_text(path))
    except Exception:
        return out
    cache = {}
    for idx, ch, ri, rn, cacn, ocn in rows:
        kind = 'auth' if ch in kinds['auth'] else 'label' if ch in kinds['label'] else None
        if kind is None:
            continue
        if (kind, ch) not in cache:
            res = kinds[kind][ch]
            junc = []
            for k in range(len(res) - 1):
                (r1, a1), (r2, a2) = res[k], res[k + 1]
                if not ({'CA', 'C', 'O'} <= a1.keys() and 'N' in a2):
                    continue
                if np.linalg.norm(a1['C'] - a2['N']) > 1.8:
                    continue
                junc.append((r1, r2, angle(a1['CA'], a1['C'], a2['N']), angle(a1['O'], a1['C'], a2['N']),
                             residue_values(r1, a1)))
            cache[(kind, ch)] = junc
        cands = [j for j in cache[(kind, ch)] if j[0] == ri and j[1] == rn
                 and abs(j[2] - cacn) < TOL and abs(j[3] - ocn) < TOL]
        if len(cands) == 1:
            out.append((idx,) + tuple(cands[0][4]))
    return out


def run_pool(fn, jobs, workers, label):
    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(fn, jobs, chunksize=8)):
            results.extend(r)
            if (k + 1) % 500 == 0:
                print(f'  {label}: {k + 1}/{len(jobs)} structures, {time.time() - t0:.0f} s', flush=True)
    return results


# ------------------------------------------------------------------ grids
def cell_index(x, origin=-180.0):
    return (np.floor((np.asarray(x, float) - origin) / 10.0).astype(int)) % B


def accumulate(rows, term, extra_index=None, n_extra=1):
    ri = rows['res_name'].map({r: k for k, r in enumerate(RES20)}).to_numpy(float)
    v = rows[term].to_numpy(float)
    ok = ~np.isnan(ri) & np.isfinite(v)
    pb, qb = cell_index(rows['phi_deg'])[ok], cell_index(rows['psi_deg'])[ok]
    shape = (n_extra, len(RES20), B, B)
    S, N = np.zeros(shape), np.zeros(shape)
    e = np.zeros(ok.sum(), int) if extra_index is None else np.asarray(extra_index)[ok]
    np.add.at(S, (e, ri[ok].astype(int), pb, qb), v[ok]); np.add.at(N, (e, ri[ok].astype(int), pb, qb), 1)
    return (S[0], N[0]) if extra_index is None else (S, N)


def grids(term, rows, pro, pooled_rule):
    s, n = accumulate(rows, term)
    ps, pn = s.sum(0), n.sum(0)
    gmean = float(s.sum() / n.sum())
    if pooled_rule == 'clean':
        pooled = np.where(pn >= MIN_COUNT, ps / np.maximum(pn, 1), gmean)
    else:
        pooled = np.where(pn > 0, ps / np.maximum(pn, 1), gmean)
    unsplit = np.where(n >= MIN_COUNT, s / np.maximum(n, 1), pooled)
    split, counts = [], []
    for flag in (False, True):
        ss, nn = accumulate(rows.loc[pro == flag], term)
        split.append(np.where(nn >= MIN_COUNT, ss / np.maximum(nn, 1), unsplit)); counts.append(nn.astype(np.int64))
    return unsplit, np.stack(split), n.astype(np.int64), np.stack(counts)


def compare(actual, expected, count):
    if actual.shape != expected.shape:
        return False, float('inf')
    d = np.abs(actual.astype(float) - expected.astype(float))
    return (bool(np.array_equal(actual, expected)) if count else bool(np.all(d <= 1e-12))), float(d.max())


def catmull_rom_2d(G, g, x, y, x0):
    u = (np.asarray(x, float) - x0) / 10.0
    v = (np.asarray(y, float) - x0) / 10.0
    i0 = np.floor(u).astype(int); j0 = np.floor(v).astype(int)
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
    g = 'NonPGIV' if res in NONPGIV else 'IleVal' if res in ('ILE', 'VAL') else \
        'Gly' if res == 'GLY' else 'Pro' if res == 'PRO' else None
    return None if g is None else g + ('_xpro' if nxt == 'PRO' else '_nonxpro')


def cdl_arrays(path):
    text = Path(path).read_text(errors='replace')
    db = runpy.run_path(str(path))['cdl_database']
    if getattr(db, 'version', None) != 'CDL v1.2':
        sys.exit('STOP: unexpected CDL version')
    # if the file documents its column order, it must agree with the assumed indices
    for line in text.splitlines():
        if 'mNAB' in line and 'mACN' in line:
            toks = [t.strip(" ',[]#:") for t in line.replace(',', ' ').split()]
            toks = [t for t in toks if t]
            for term, name in CDL_NAME.items():
                if name in toks and 'mCNA' in toks:
                    pos = toks.index(name) - toks.index('mCNA') + 2
                    if pos != CDL_COL[term]:
                        sys.exit(f'STOP: CDL header puts {name} at {pos}, expected {CDL_COL[term]}')
            break
    groups = sorted(db)
    arrays = {}
    for term, col in CDL_COL.items():
        G = np.full((len(groups), B, B), np.nan)
        for i, g in enumerate(groups):
            for (phi, psi), row in db[g].items():
                if row[col] > 0:
                    G[i, (phi + 180) // 10, (psi + 180) // 10] = row[col]
        lo, hi = CDL_RANGE[term]
        if not lo <= np.nanmedian(G) <= hi:
            sys.exit(f'STOP: CDL column {col} ({term}) median {np.nanmedian(G):.2f} outside {lo}-{hi}')
        arrays[term] = G
    return groups, arrays


# ------------------------------------------------------------------ statistics
def bootstrap_mean(series, seed=SEED, draws=DRAWS):
    v = series.dropna()
    rng = np.random.default_rng(seed)
    samples = []
    vals = v.to_numpy()
    for start in range(0, draws, 500):
        n = min(500, draws - start)
        W = rng.multinomial(len(vals), np.full(len(vals), 1 / len(vals)), size=n)
        samples.append((W @ vals) / len(vals))
    s = np.concatenate(samples)
    p = min(1.0, 2 * min((s <= 0).mean(), (s >= 0).mean()))
    return dict(estimate=float(vals.mean()), ci95=np.quantile(s, [.025, .975]).tolist(), p_boot=float(p),
                groups=int(len(vals)))


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [None] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i])); adj[i] = running
    return adj


# ------------------------------------------------------------------ main steps
def step_a(train, targets):
    """Rebuild v2 grids for the primary terms; return eligible rows and Pro flags."""
    t = train.sort_values(['pdb_id', 'chain', 'res_idx']).reset_index(drop=True)
    nxt = t.groupby(['pdb_id', 'chain'], sort=False)[['res_idx', 'res_name']].shift(-1)
    cn, om = t['bond_C_N_next'], t['omega_measured_deg'].abs()
    pro = nxt['res_name'].eq('PRO')
    elig = (nxt['res_idx'].eq(t['res_idx'] + 1) & nxt['res_name'].isin(AA) & np.isfinite(cn) & cn.gt(0) & cn.le(1.8)
            & t['res_name'].isin(AA) & np.isfinite(t['phi_deg']) & np.isfinite(t['psi_deg'])
            & om.gt(150) & om.le(180.001) & t['is_pro_np1'].isin([0, 1]) & t['is_pro_np1'].eq(pro.astype(int)))
    report, oks = {'eligible_junctions': int(elig.sum())}, {}
    with np.load(targets, allow_pickle=False) as z:
        frozen = {k: z[k] for k in z.files}
    for key, term in PRIMARY.items():
        v = t[term]
        ok = elig & np.isfinite(v) & v.gt(0) & v.lt(180)
        if key in ('NCB', 'CCB'):
            ok &= t['res_name'].ne('GLY')
        oks[key] = ok
        matched = None
        for rule in ('clean', 'legacy'):
            arr = grids(term, t.loc[ok], pro[ok], rule)
            res = [compare(a, frozen[f'{term}__{s}'], s.startswith('n_'))
                   for a, s in zip(arr, ('unsplit', 'v2', 'n_unsplit', 'n_split'))]
            if all(r[0] for r in res):
                matched = rule; break
        report[term] = dict(rows=int(ok.sum()), pooled_rule_reproducing=matched,
                            max_abs_difference=max(r[1] for r in res))
        if matched is None:
            sys.exit(f'STOP: rebuilt {term} grids differ from the frozen v2 arrays: {report[term]}')
    print('Rebuild gate PASS', json.dumps(report), flush=True)
    return t, pro, oks, frozen, report


def step_d(t, pro, oks, frozen, chi):
    """Fit chi1 surfaces nested in v2's next-Pro split; return arrays [3, 40, 36, 36] per term."""
    cls = np.array([chi1_class(c) for c in chi])
    out, fallback = {}, []
    for key, term in PRIMARY.items():
        v2 = frozen[f'{term}__v2'].reshape(40, B, B)
        G = np.repeat(v2[None], 3, 0).copy()
        m = oks[key].to_numpy() & (cls >= 0) & ~t['res_name'].isin(NO_CHI1).to_numpy()
        rows = t.loc[m]
        extra = cls[m]
        S, N = accumulate(rows, term, extra_index=extra, n_extra=3)  # [3, 20, 36, 36]
        Sx, Nx = np.zeros((3, 40, B, B)), np.zeros((3, 40, B, B))
        p = pro[m].to_numpy()
        for flag in (False, True):
            sel = p == flag
            s2, n2 = accumulate(rows.loc[sel], term, extra_index=extra[sel], n_extra=3)
            Sx[:, 20 * flag:20 * flag + 20], Nx[:, 20 * flag:20 * flag + 20] = s2, n2
        use = Nx >= MIN_COUNT
        G[use] = Sx[use] / Nx[use]
        out[term] = G
        out[term + '__n'] = Nx.astype(np.int64)
        # training-row fallback: share of chi1-class rows whose own cell has < 10 observations
        ri = rows['res_name'].map({r: k for k, r in enumerate(RES20)}).to_numpy(int) + 20 * p
        own = Nx[extra, ri, cell_index(rows['phi_deg']), cell_index(rows['psi_deg'])]
        fallback.append(dict(term=term, chi1_rows=int(m.sum()), rows_in_supported_cells=int((own >= MIN_COUNT).sum()),
                             fallback_row_fraction=float((own < MIN_COUNT).mean()) if len(own) else None,
                             supported_cells=int(use.sum())))
    return out, pd.DataFrame(fallback)


def predict_scores(d, frozen, chimodel, cdl_groups, cdl):
    """Per-junction absolute errors for M_v2, M_v2chi and CDL, per term."""
    ri = d['res_i'].map({r: k for k, r in enumerate(RES20)}).to_numpy(int)
    pro = d['res_next'].eq('PRO').to_numpy()
    si = ri + 20 * pro
    phi, psi = d['phi_i'].to_numpy(float), d['psi_i'].to_numpy(float)
    gmap = {g: i for i, g in enumerate(cdl_groups)}
    gi = np.array([gmap.get(cdl_group(a, b), -1) for a, b in zip(d['res_i'], d['res_next'])])
    cls = np.array([chi1_class(c) for c in d['chi1']])
    has = (cls >= 0) & ~d['res_i'].isin(NO_CHI1).to_numpy()
    rows = []
    for key, term in {**PRIMARY, **CONTROLS}.items():
        obs = d[{'TAU': 'tau_obs', 'NCB': 'ncb_obs', 'CCB': 'ccb_obs', 'ACN': 'cacn_deg', 'OCN': 'ocn_deg'}[key]].to_numpy(float)
        ok = np.isfinite(obs) & (obs > 0) & (obs < 180) & (gi >= 0)
        if key in ('NCB', 'CCB'):
            ok &= d['res_i'].ne('GLY').to_numpy()
        v2grid = frozen[f'{term}__v2'].reshape(40, B, B)
        pv2 = np.full(len(d), np.nan); pchi = np.full(len(d), np.nan); pcdl = np.full(len(d), np.nan)
        pv2[ok] = catmull_rom_2d(v2grid, si[ok], phi[ok], psi[ok], -175)
        pchi[:] = pv2
        if key in PRIMARY:
            G = chimodel[term]
            m = ok & has
            for c in range(3):
                mc = m & (cls == c)
                if mc.any():
                    pchi[mc] = catmull_rom_2d(G[c], si[mc], phi[mc], psi[mc], -175)
        pcdl[ok] = catmull_rom_2d(cdl[key], gi[ok], phi[ok], psi[ok], -180)
        keep = ok & np.isfinite(pv2) & np.isfinite(pchi) & np.isfinite(pcdl)
        rows.append(pd.DataFrame({'pdb_id': d['pdb_id'].to_numpy()[keep], 'component': d['component'].to_numpy()[keep],
                                  'stratum': d['refinement_stratum'].to_numpy()[keep], 'term': key,
                                  'chi1_conditioned': has[keep] if key in PRIMARY else False,
                                  'v2': np.abs(pv2 - obs)[keep], 'v2chi': np.abs(pchi - obs)[keep],
                                  'cdl': np.abs(pcdl - obs)[keep]}))
    return pd.concat(rows, ignore_index=True)


def contrasts(junc_err):
    s = (junc_err.groupby(['term', 'stratum', 'pdb_id', 'component'])
         .agg(v2=('v2', 'mean'), v2chi=('v2chi', 'mean'), cdl=('cdl', 'mean'), junctions=('v2', 'size'),
              chi1_junctions=('chi1_conditioned', 'sum')).reset_index())
    s = s[s['junctions'] >= MIN_STRUCT_JUNCTIONS]
    out = {}
    for sname, members in STRATA.items():
        for fam, (a, b) in (('Delta_chi', ('v2chi', 'v2')), ('Delta_CDL', ('v2chi', 'cdl'))):
            res = {}
            for key in list(PRIMARY) + list(CONTROLS):
                x = s[(s['term'] == key) & s['stratum'].isin(members)]
                if x['component'].nunique() < 10:
                    res[key] = dict(status='not_estimable', groups=int(x['component'].nunique())); continue
                diff = (x[a] - x[b]).groupby(x['component']).mean()
                r = bootstrap_mean(diff)
                r.update(status='ok', structures=int(len(x)),
                         mae={a: float(x.groupby('component')[a].mean().mean()), b: float(x.groupby('component')[b].mean().mean())})
                res[key] = r
            prim = [k for k in PRIMARY if res[k].get('status') == 'ok']
            for k, pa in zip(prim, holm([res[k]['p_boot'] for k in prim])):
                res[k]['p_holm'] = pa
                if sname == 'REFMAC':
                    res[k]['reading'] = ('chi1 adds information' if fam == 'Delta_chi' else 'lower error than CDL') \
                        if (res[k]['estimate'] < 0 and pa < 0.05) else ('no added information' if fam == 'Delta_chi' else 'not lower')
            out[f'{fam}|{sname}'] = res
    return out, s


def make_figure(res, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    terms = list(PRIMARY) + list(CONTROLS)
    colors = {'REFMAC': '#2b6cb0', 'REFMAC+SHELXL': '#7f9cc0', 'PHENIX_post': '#c05621'}
    for ax, fam in zip(axs, ('Delta_chi', 'Delta_CDL')):
        for j, sname in enumerate(STRATA):
            r = res[f'{fam}|{sname}']
            for i, k in enumerate(terms):
                if r[k].get('status') != 'ok':
                    continue
                y = i + (j - 1) * 0.22
                e = r[k]
                ax.errorbar(e['estimate'], y, xerr=[[e['estimate'] - e['ci95'][0]], [e['ci95'][1] - e['estimate']]],
                            fmt='o', ms=3.5, lw=1, color=colors[sname], label=sname if i == 0 else None)
        ax.axvline(0, color='#555', lw=0.8)
        ax.set_yticks(range(len(terms))); ax.set_yticklabels(['tau', 'N-CA-CB', 'C-CA-CB', 'Ca-C-N (control)', 'O-C-N (control)'])
        ax.invert_yaxis()
        ax.set_xlabel('MAE difference (deg)')
        ax.set_title('With chi1 minus without' if fam == 'Delta_chi' else 'MechLib (v2 + chi1) minus CDL', fontsize=9, loc='left')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axs[0].legend(fontsize=7, frameon=False, loc='lower left')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'Chi1.{ext}', dpi=200)
    plt.close(fig)


def main(a):
    t0 = time.time()
    root = Path(a.root)
    P = {k: root / v for k, v in DEFAULTS.items()}
    if a.out.exists():
        sys.exit(f'STOP: {a.out} exists')
    hashes = {'spec': require(SPEC, None, skip=True)}
    if hashes['spec'] != SPEC_SHA:
        sys.exit('STOP: spec hash differs from the frozen v1.1')
    for k in ('train_geometry', 'targets', 'junctions', 'metadata', 'strata'):
        hashes[k] = require(P[k], k)
    hashes['training_chains'] = require(KIT / 'inputs' / 'training_chains.tsv', 'training_chains')
    hashes['cdl'] = require(a.cdl, 'cdl')
    hashes['code'] = sha(Path(__file__))
    cdl_groups, cdl = cdl_arrays(a.cdl)
    # ---- A: training geometry and rebuild gate
    cols = ['pdb_id', 'chain', 'res_idx', 'res_name', 'phi_deg', 'psi_deg', 'omega_measured_deg', 'is_pro_np1',
            'bond_C_N_next'] + list(PRIMARY.values())
    train = pd.read_csv(P['train_geometry'], usecols=cols, keep_default_na=False, low_memory=False,
                        dtype={'pdb_id': str, 'chain': str})
    train['pdb_id'] = train['pdb_id'].str.strip().str.upper()
    for c in set(cols) - {'pdb_id', 'chain', 'res_name'}:
        train[c] = pd.to_numeric(train[c], errors='coerce')
    keys = pd.read_csv(KIT / 'inputs' / 'training_chains.tsv', sep='\t', dtype=str, keep_default_na=False)
    keys['pdb_id'] = keys['pdb_id'].str.upper()
    train = train.merge(keys[['pdb_id', 'chain']], on=['pdb_id', 'chain'], how='inner')
    if len(train) != 1224328:
        sys.exit(f'STOP: training rows after chain restriction {len(train)} != frozen 1,224,328')
    t, pro, oks, frozen, gate = step_a(train, P['targets'])
    a.out.mkdir(parents=True)
    (a.out / 'REBUILD_GATE.json').write_text(json.dumps(gate, indent=2) + '\n')
    # ---- B: training chi1
    need = oks['TAU'] & ~t['res_name'].isin(NO_CHI1)
    jobs = []
    for pdb, g in t.loc[need].groupby('pdb_id'):
        path = P['train_coords'] / f'{pdb.lower()}.pdb'
        jobs.append((str(path), list(zip(g.index, g['chain'], g['res_name'], g['tau_deg']))))
    print(f'Training chi1 extraction: {len(jobs)} structures', flush=True)
    got = run_pool(train_worker, jobs, a.workers, 'training')
    chi = np.full(len(t), np.nan)
    for idx, c in got:
        chi[idx] = c
    gated = np.zeros(len(t), bool)
    gated[[i for i, _ in got]] = True
    cov_train = float(gated[need.to_numpy()].mean())
    # ---- C: held-out
    dt = {n: str for n in KEY[:3]}
    rows = pd.read_csv(P['junctions'], sep='\t', dtype=dt)
    meta = pd.read_csv(P['metadata'], sep='\t', dtype=dt, usecols=KEY + ['B30_window_pass'])
    strata = pd.read_csv(P['strata'], sep='\t', dtype=str, usecols=['pdb_id', 'component', 'refinement_stratum'])
    d = rows.merge(meta, on=KEY, validate='one_to_one').merge(strata, on=KEY[:2], validate='many_to_one')
    if len(d) != 618881 or d['pdb_id'].nunique() != 1295:
        sys.exit('STOP: unexpected held-out join counts')
    d = d[d['B30_window_pass'].eq('pass')].reset_index(drop=True)
    for c in ('cacn_deg', 'ocn_deg', 'phi_i', 'psi_i'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    jobs = []
    for pdb, g in d.groupby('pdb_id'):
        path = P['heldout_coords'] / f'{pdb.upper()}.cif.gz'
        jobs.append((str(path), list(zip(g.index, g['chain'], g['res_i'], g['res_next'], g['cacn_deg'], g['ocn_deg']))))
    print(f'Held-out extraction: {len(jobs)} structures', flush=True)
    got = run_pool(heldout_worker, jobs, a.workers, 'held-out')
    for c in ('tau_obs', 'ncb_obs', 'ccb_obs', 'chi1'):
        d[c] = np.nan
    if got:
        arr = np.array(got, float)
        idx = arr[:, 0].astype(int)
        d.loc[idx, ['tau_obs', 'ncb_obs', 'ccb_obs', 'chi1']] = arr[:, 1:]
    gate_h = d['tau_obs'].notna()
    elig_h = d['res_i'].isin(AA) & d['res_next'].isin(AA) & d['cacn_deg'].notna()
    cov_held = float(gate_h[elig_h].mean())
    cov = dict(training_chi1_bearing_rows=int(need.sum()), training_gated=int(gated[need.to_numpy()].sum()),
               training_coverage=cov_train, heldout_B30_rows=int(elig_h.sum()), heldout_gated=int(gate_h[elig_h].sum()),
               heldout_coverage=cov_held,
               heldout_chi1_class_counts={CLASSES[k] if k >= 0 else 'none/eclipsed': int(v) for k, v in
                                          pd.Series([chi1_class(x) for x in d.loc[gate_h, 'chi1']]).value_counts().items()},
               threshold=GATE_COVERAGE)
    (a.out / 'CHI1_COVERAGE.json').write_text(json.dumps(cov, indent=2) + '\n')
    print('Coverage', json.dumps(cov), flush=True)
    if cov_train < GATE_COVERAGE or cov_held < GATE_COVERAGE:
        sys.exit('STOP: same-atom gate coverage below 75%; no scores computed (see CHI1_COVERAGE.json)')
    # ---- D: fit
    chimodel, fb = step_d(t, pro, oks, frozen, chi)
    np.savez_compressed(a.out / 'CHI1_MODEL.npz', classes=np.array(CLASSES), residues=np.array(RES20),
                        **{k.replace('__', '_'): v for k, v in chimodel.items()})
    fb.to_csv(a.out / 'FALLBACK_FRACTIONS.tsv', sep='\t', index=False)
    # ---- E: score
    err = predict_scores(d, frozen, chimodel, cdl_groups, cdl)
    for key in CONTROLS:
        e = err[err['term'] == key]
        if not np.allclose(e['v2chi'], e['v2'], atol=1e-6, rtol=0):
            sys.exit(f'STOP: control {key} changed under chi1 conditioning; run invalid')
    res, smae = contrasts(err)
    smae.to_csv(a.out / 'STRUCTURE_MAE_CHI1.tsv', sep='\t', index=False)
    (a.out / 'CONTRASTS.json').write_text(json.dumps(res, indent=2) + '\n')
    make_figure(res, a.out)
    reported = {term: ('v2 + chi1' if res['Delta_chi|REFMAC'][k].get('reading') == 'chi1 adds information' else 'v2')
                for k, term in PRIMARY.items()}
    lock = dict(spec='CHI1_FAIR_COMPARISON_SPEC_20261111.md v1.1', hashes=hashes, chi1_model_sha=sha(a.out / 'CHI1_MODEL.npz'),
                seed=SEED, draws=DRAWS, coverage=cov, rebuild_gate=gate, reported_mechlib_per_term=reported,
                runtime_s=round(time.time() - t0))
    (a.out / 'RUN_LOCK.json').write_text(json.dumps(lock, indent=2) + '\n')
    print('\nPRIMARY STRATUM (REFMAC)')
    for fam in ('Delta_chi', 'Delta_CDL'):
        for k in PRIMARY:
            r = res[f'{fam}|REFMAC'][k]
            if r.get('status') == 'ok':
                print(f"{fam:9s} {k}: {r['estimate']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
                      f"p_holm={r['p_holm']:.3g} -> {r['reading']}")
    print('Reported MechLib per term:', reported)
    print('Return the whole folder:', a.out)


# ------------------------------------------------------------------ self-test
def selftest():
    import tempfile
    rng = np.random.default_rng(3)
    # geometry maths: build a residue with known tau and chi1
    N, CA = np.array([1.458, 0, 0]), np.zeros(3)
    tau = np.radians(111.0)
    C = 1.525 * np.array([np.cos(tau), np.sin(tau), 0])
    CB = np.array([-0.53, -0.76, 1.21])
    def place(a, b, c, bond, ang, tor):
        bc = (c - b) / np.linalg.norm(c - b); n = np.cross(b - a, bc); n /= np.linalg.norm(n)
        m = np.cross(n, bc)
        d2 = np.array([-bond * np.cos(np.radians(ang)), bond * np.sin(np.radians(ang)) * np.cos(np.radians(tor)),
                       bond * np.sin(np.radians(ang)) * np.sin(np.radians(tor))])
        return c + d2[0] * bc + d2[1] * m + d2[2] * n
    CG = place(N, CA, CB, 1.52, 114.0, -63.0)
    at = {'N': N, 'CA': CA, 'C': C, 'CB': CB, 'CG': CG}
    t_, ncb, ccb, chi = residue_values('LEU', at)
    assert abs(t_ - 111.0) < 1e-6 and abs(chi + 63.0) < 1e-6, (t_, chi)
    assert chi1_class(-63) == 0 and chi1_class(178) == 1 and chi1_class(-178) == 1 and chi1_class(62) == 2 and chi1_class(0) == -1
    print('PASS geometry: tau, chi1 and rotamer classes')
    # parsers (with an altloc atom that must be ignored)
    def pdbline(name, resn, ch, seq, xyz, alt=' '):
        return f"ATOM  {1:5d} {name:<4s}{alt}{resn:>3s} {ch}{seq:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00"
    lines = [pdbline(n, 'LEU', 'A', 5, x) for n, x in at.items()] + [pdbline('CG', 'LEU', 'A', 5, CG + 5, 'B')]
    parsed = residues_from_pdb('\n'.join(lines))['auth']['A'][0]
    assert parsed[0] == 'LEU' and np.allclose(parsed[1]['CG'], np.round(CG, 3))
    cif = ['data_x', 'loop_'] + [f'_atom_site.{t}' for t in ('group_PDB', 'id', 'label_atom_id', 'label_alt_id', 'label_comp_id',
          'label_asym_id', 'label_seq_id', 'auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y', 'Cartn_z',
          'pdbx_PDB_model_num')]
    for k, (n, x) in enumerate(at.items()):
        cif.append(f'ATOM {k} {n} . LEU A 1 B 5 ? {x[0]:.3f} {x[1]:.3f} {x[2]:.3f} 1')
    cif.append(f'ATOM 9 CG A LEU A 1 B 5 ? 9 9 9 1')
    pc = residues_from_cif('\n'.join(cif) + '\n#\n')
    assert 'B' in pc['auth'] and 'A' in pc['label'] and np.allclose(pc['auth']['B'][0][1]['CG'], np.round(CG, 3))
    print('PASS parsers: PDB and mmCIF, first model, alternate locations ignored')
    # grids: rebuild equals itself; chi1 surfaces fall back to v2 when sparse
    n = 40000
    rows = pd.DataFrame({'res_name': rng.choice(['LEU', 'SER', 'GLY'], n), 'phi_deg': rng.uniform(-80, -40, n),
                         'psi_deg': rng.uniform(-60, -20, n)})
    chi = rng.choice([-60.0, 180.0, 60.0], n) + rng.normal(0, 10, n)
    cls = np.array([chi1_class(c) for c in chi])
    rows['tau_deg'] = 111 + 0.8 * (cls == 1) - 0.8 * (cls == 2) + rng.normal(0, 0.5, n)
    rows['angle_N_CA_CB'] = 110.5 + rng.normal(0, 0.5, n)
    rows['angle_C_CA_CB'] = 110.1 + rng.normal(0, 0.5, n)
    pro = pd.Series(rng.random(n) < 0.05)
    frozen = {}
    for term in PRIMARY.values():
        for s, arr in zip(('unsplit', 'v2', 'n_unsplit', 'n_split'), grids(term, rows, pro, 'clean')):
            frozen[f'{term}__{s}'] = arr
    again = grids('tau_deg', rows, pro, 'clean')
    assert all(compare(x, frozen[f'tau_deg__{s}'], s.startswith('n_'))[0] for x, s in zip(again, ('unsplit', 'v2', 'n_unsplit', 'n_split')))
    oks = {k: pd.Series(True, index=rows.index) & (rows['res_name'].ne('GLY') if k != 'TAU' else True) for k in PRIMARY}
    model, fb = step_d(rows, pro, oks, frozen, chi)
    print(fb.to_string())
    # scoring on fresh synthetic held-out data drawn the same way: chi1 must help tau, not the CB terms
    m = 20000
    h = pd.DataFrame({'pdb_id': [f'P{i % 400}' for i in range(m)], 'component': [f'K{i % 400}' for i in range(m)],
                      'refinement_stratum': np.where(np.arange(m) % 400 < 300, 'REFMAC', 'PHENIX_post_default'),
                      'res_i': rng.choice(['LEU', 'SER'], m), 'res_next': 'ALA', 'phi_i': rng.uniform(-75, -45, m),
                      'psi_i': rng.uniform(-55, -25, m)})
    hc = rng.choice([-60.0, 180.0, 60.0], m) + rng.normal(0, 10, m)
    hcls = np.array([chi1_class(c) for c in hc])
    h['chi1'] = hc
    h['tau_obs'] = 111 + 0.8 * (hcls == 1) - 0.8 * (hcls == 2) + rng.normal(0, 0.5, m)
    h['ncb_obs'] = 110.5 + rng.normal(0, 0.5, m); h['ccb_obs'] = 110.1 + rng.normal(0, 0.5, m)
    h['cacn_deg'] = 116.5 + rng.normal(0, 1, m); h['ocn_deg'] = 122.8 + rng.normal(0, 1, m)
    for term in CONTROLS.values():
        frozen[f'{term}__v2'] = np.full((2, 20, B, B), 116.5 if 'CaCN' in term else 122.8)
    groups = ['NonPGIV_nonxpro', 'NonPGIV_xpro']
    fake_cdl = {k: np.full((2, B, B), v) for k, v in {'TAU': 111.0, 'NCB': 110.5, 'CCB': 110.1, 'ACN': 116.5, 'OCN': 122.8}.items()}
    global DRAWS
    DRAWS = 1000
    err = predict_scores(h, frozen, model, groups, fake_cdl)
    for key in CONTROLS:
        e = err[err['term'] == key]
        assert np.allclose(e['v2chi'], e['v2'], atol=1e-6)
    res, _ = contrasts(err)
    r = res['Delta_chi|REFMAC']
    print('synthetic Delta_chi REFMAC:', {k: round(r[k]['estimate'], 3) for k in PRIMARY})
    assert r['TAU']['estimate'] < -0.1 and r['TAU']['reading'] == 'chi1 adds information'
    assert abs(r['NCB']['estimate']) < 0.05 and abs(r['CCB']['estimate']) < 0.05
    with tempfile.TemporaryDirectory() as tmp:
        make_figure(res, Path(tmp))
    print('PASS synthetic: chi1 effect recovered where built in, none where absent; controls unchanged')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='/mnt/c/Wei/PBM')
    ap.add_argument('--cdl', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not (a.cdl and a.out):
            ap.error('--cdl and --out are required (or --selftest)')
        main(a)
