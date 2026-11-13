"""Shared, dependency-light helpers (numpy only) for the CDL -> MechLib v2 swap test.

Used both by stage 1 (plain python) and inside PHENIX (run_swap.py), so the planned
edit and the applied edit come from the same function.
"""
import hashlib
from pathlib import Path

import numpy as np

NONPGIV = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS',
           'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR'}
RES20 = sorted(NONPGIV | {'ILE', 'VAL', 'GLY', 'PRO'})
B = 36
# CDL column -> clean v2 term
TERMS = ['mACN', 'mACO', 'mOCN']
COLS = {'mACN': 12, 'mACO': 10, 'mOCN': 14}
V2TERM = {'mACN': 'angle_CaCN', 'mACO': 'angle_CA_C_O', 'mOCN': 'angle_O_C_N_next'}
TARGETS_SHA = 'ef7b86b359c84492a0be791de52a1e64112a0153142b9f9f5a0417448d2731b1'
CDL_SHA = '79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_v2(path):
    if sha(path) != TARGETS_SHA:
        raise ValueError('MechLib v2 target file differs from the frozen ef7b86b3...')
    with np.load(path, allow_pickle=False) as z:
        if list(z['residues']) != RES20:
            raise ValueError('Unexpected residue axis')
        return {t: z[V2TERM[t] + '__v2'].reshape(40, B, B).astype(float) for t in TERMS}


def catmull_rom_point(G, g, x, y, x0=-175.0):
    u, v = (x - x0) / 10.0, (y - x0) / 10.0
    i0, j0 = int(np.floor(u)), int(np.floor(v))
    tx, ty = u - i0, v - j0

    def w(t):
        t2, t3 = t * t, t * t * t
        return [0.5 * (-t3 + 2 * t2 - t), 0.5 * (3 * t3 - 5 * t2 + 2), 0.5 * (-3 * t3 + 4 * t2 + t), 0.5 * (t3 - t2)]
    wx, wy = w(tx), w(ty)
    out = 0.0
    for a in range(4):
        for b in range(4):
            out += wx[a] * wy[b] * G[g, (i0 + a - 1) % B, (j0 + b - 1) % B]
    return float(out)


def v2_node(v2, term, res_i, res_next, key):
    """MechLib v2 target for the central residue at a CDL lookup node (phi, psi)."""
    g = RES20.index(res_i) + 20 * (res_next == 'PRO')
    return catmull_rom_point(v2[term], g, float(key[0]), float(key[1]))


def round_to_ten(x):
    t = int(round(float(x) / 10.0)) * 10
    return -180 if t == 180 else t


def cdl_group(res, nxt):
    g = 'NonPGIV' if res in NONPGIV else 'IleVal' if res in ('ILE', 'VAL') else \
        'Gly' if res == 'GLY' else 'Pro' if res == 'PRO' else None
    return None if g is None else g + ('_xpro' if nxt == 'PRO' else '_nonxpro')
