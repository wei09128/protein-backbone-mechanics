"""
collect_backbone_features.py v5 — Steric hindrance field + mechanical forces
=============================================================================
Architecture:
  GROUP A: Steric hindrance field (21 features) — WHERE backbone CAN go
  GROUP B: Five mechanical forces (14 features) — WHERE backbone DOES go
  GROUP C: Minimal context (22 features) — HOW forces transmit

  REMOVED: dipole, cn_prev/np1, ss_nm*, res_nm*, vec_cacb, contact_count,
           burial_proxy, sc_transmit, hb_mean_angle, hb_n_weak

  Outputs: phi_deg, psi_deg, omega_deg → sin/cos = 6 targets
  Total input features: ~57

Usage:
  python collect_backbone_features.py --pdb_dir ./pdb_cache --out features_v5.csv
"""

import argparse, collections, csv, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
from nerf_builder import NeRFBuilder
from hbond_finder import find_hbonds, E_WEAK
from molcore import PDBParser, dihedral_angle, AminoAcids
from molcore import BondGeometry

_SIDECHAIN_MASS = {'GLY':0.0,'ALA':15.0,'VAL':43.0,'LEU':57.0,'ILE':57.0,'PRO':42.0,'PHE':91.0,'TYR':107.0,'TRP':130.0,'SER':31.0,'THR':45.0,'CYS':47.0,'MET':75.0,'ASP':58.0,'ASN':58.0,'GLU':72.0,'GLN':72.0,'LYS':72.0,'ARG':100.0,'HIS':81.0}
_BB_CHARGES = {'N':-0.4157,'H':+0.2719,'CA':+0.0337,'C':+0.5973,'O':-0.5679,'CB':-0.0518}
_SC_CHARGE = {'ARG':+1.0,'LYS':+1.0,'ASP':-1.0,'GLU':-1.0,'HIS':+0.5,'SER':-0.1,'THR':-0.1,'TYR':-0.15,'ASN':-0.05,'GLN':-0.05,'CYS':-0.08}
_RES_ORDER = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
_RES_IDX = {r:i for i,r in enumerate(_RES_ORDER)}
_HAS_CHI2 = {'ARG','ASN','ASP','GLN','GLU','HIS','ILE','LEU','LYS','MET','PHE','TRP','TYR'}
_SC_N_HEAVY = {'GLY':0,'ALA':1,'VAL':3,'LEU':4,'ILE':4,'PRO':3,'PHE':7,'TYR':8,'TRP':10,'SER':2,'THR':3,'CYS':2,'MET':4,'ASP':4,'ASN':4,'GLU':5,'GLN':5,'LYS':5,'ARG':7,'HIS':6}
_SC_N_ROTATABLE = {'GLY':0,'ALA':0,'VAL':1,'LEU':2,'ILE':2,'PRO':0,'PHE':2,'TYR':2,'TRP':2,'SER':1,'THR':1,'CYS':1,'MET':3,'ASP':2,'ASN':2,'GLU':3,'GLN':3,'LYS':4,'ARG':4,'HIS':2}
_SC_BRANCHED = {'VAL','ILE','THR'}
_SC_AROMATIC = {'PHE','TYR','TRP','HIS'}

def _unit(v):
    n = np.linalg.norm(v)
    return v/n if n > 1e-10 else np.zeros(3)

def _torque(force, position, axis_point, axis_hat):
    return float(np.dot(np.cross(position - axis_point, force), axis_hat))

def _local_frame(N, CA, C):
    x = _unit(CA-N); c = _unit(C-CA)
    z = np.cross(x, c)
    if np.linalg.norm(z) < 1e-10: z = np.array([0.,0.,1.])
    else: z = _unit(z)
    y = np.cross(z, x)
    return np.stack([x, y, z])

def _ss_bin(p, q):
    if p > 0 and -20 <= q <= 80: return 5
    if -100<=p<=-40 and -60<=q<=20: return 0
    if p<=-90 and q>=90: return 1
    if -90<=p<=-50 and q>=120: return 2
    if -80<=p<=-30 and -40<=q<=0: return 3
    return 4

def _cg_position(CA, CB, chi1_rad):
    if chi1_rad is None or CB is None: return None
    if not (np.all(np.isfinite(CA)) and np.all(np.isfinite(CB))): return None
    b1 = _unit(CB-CA)
    perp = np.array([1.,0.,0.]) if abs(b1[0])<0.9 else np.array([0.,1.,0.])
    n_vec = _unit(np.cross(b1, perp)); m_vec = np.cross(b1, n_vec)
    ang = np.radians(111.0)
    d = np.array([-np.cos(ang), np.sin(ang)*np.cos(-chi1_rad), np.sin(ang)*np.sin(-chi1_rad)])
    return CB + 1.52*(d[0]*b1 + d[1]*m_vec + d[2]*n_vec)

def _extract_bfactors(builder):
    if hasattr(builder, 'bfactors') and builder.bfactors is not None: return builder.bfactors
    if not hasattr(builder, 'pdb_file') or builder.pdb_file is None: return None
    bf_by_res = {}; resnums = []
    try:
        with open(builder.pdb_file) as fh:
            for line in fh:
                if not line.startswith('ATOM  '): continue
                aname = line[12:16].strip()
                if aname not in ('N','CA','C','O'): continue
                try: bf = float(line[60:66].strip())
                except: bf = 0.0
                rk = line[17:27].strip()
                if rk not in bf_by_res: bf_by_res[rk] = {}; resnums.append(rk)
                bf_by_res[rk][aname] = bf
    except: return None
    n = len(builder.sequence) if hasattr(builder,'sequence') else 0
    if n == 0 or len(resnums) < n: return None
    result = {a: np.zeros(n) for a in ('N','CA','C','O')}
    for i in range(min(n, len(resnums))):
        d = bf_by_res.get(resnums[i], {})
        for a in ('N','CA','C','O'): result[a][i] = d.get(a, 0.0)
    if all(v.sum()==0 for v in result.values()): return None
    return result


# ══════════════════════════════════════════════════════════════════════════════
# GROUP B: Five mechanical forces (corrected torques from v2)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_torques(res_idx, coords, bonds, sequence, chi_list):
    # 1. Access existing coordinates
    N = coords['N'][res_idx]
    CA = coords['CA'][res_idx]
    C = coords['C'][res_idx]
    O = coords['O'][res_idx]
    # 2. Get/Place the Hydrogen (H)
    H = coords.get('H', np.full((len(sequence), 3), np.nan))[res_idx]
    # If H is missing/invalid and it's not Proline, place it virtually
    if sequence[res_idx] != 'PRO' and (np.all(np.isnan(H)) or np.linalg.norm(H) < 0.1):
        if res_idx > 0:
            # Use the previous residue's Carbon to define the plane
            prev_C = coords['C'][res_idx - 1]
            
            # Placement logic: H is placed 1.01A from N, 
            # roughly bisecting the C(prev)-N-CA angle (trans to Oxygen)
            v1 = _unit(N - prev_C)
            v2 = _unit(N - CA)
            H = N + _unit(v1 + v2) * BondGeometry.N_H 
        else:
            # For the first residue (N-term), just project away from CA
            H = N + _unit(N - CA) * BondGeometry.N_H
    CB = coords['CB'][res_idx] if 'CB' in coords else None
    phi_hat=_unit(CA-N); psi_hat=_unit(C-CA); phi_pt=N; psi_pt=CA
    res=sequence[res_idx]; sc_mass=_SIDECHAIN_MASS.get(res, 0.0)
    chi1 = None
    if chi_list and res_idx < len(chi_list) and chi_list[res_idx]:
        chi1 = float(chi_list[res_idx][0])
    CG = _cg_position(CA, CB, chi1) if CB is not None else None
    t = {k:0.0 for k in ['phi_bb_donor','psi_bb_donor','phi_bb_acc','psi_bb_acc',
         'phi_sc_hb','psi_sc_hb','phi_steric','psi_steric','phi_elec','psi_elec']}
    for bond in bonds:
        e=bond['energy']; vHO=np.array(bond['vec_HO']); Fm=abs(e)
        if bond['donor']==res_idx:
            F=Fm*vHO
            if np.all(np.isfinite(H)) and np.linalg.norm(H)>0.01:
                t['phi_bb_donor']+=_torque(F,H,phi_pt,phi_hat); t['psi_bb_donor']+=_torque(F,H,psi_pt,psi_hat)
            elif CB is not None and np.all(np.isfinite(CB)):
                sp=CB+_unit(CB-CA)*0.8; t['phi_sc_hb']+=_torque(F,sp,phi_pt,phi_hat); t['psi_sc_hb']+=_torque(F,sp,psi_pt,psi_hat)
        elif bond['acceptor']==res_idx:
            F=Fm*(-vHO)
            if np.all(np.isfinite(O)) and np.linalg.norm(O)>0.01:
                t['phi_bb_acc']+=_torque(F,O,phi_pt,phi_hat); t['psi_bb_acc']+=_torque(F,O,psi_pt,psi_hat)
            elif CB is not None and np.all(np.isfinite(CB)):
                sp=CB+_unit(CB-CA)*1.2; t['phi_sc_hb']+=_torque(F,sp,phi_pt,phi_hat); t['psi_sc_hb']+=_torque(F,sp,psi_pt,psi_hat)
    if CB is not None and np.all(np.isfinite(CB)) and sc_mass>0:
        Fcb=-_unit(CB-CA)*sc_mass/100.0
        t['phi_steric']+=_torque(Fcb,CB,phi_pt,phi_hat); t['psi_steric']+=_torque(Fcb,CB,psi_pt,psi_hat)
        if CG is not None and np.all(np.isfinite(CG)):
            Fcg=-_unit(CG-CA)*sc_mass/100.0
            t['phi_steric']+=_torque(Fcg,CG,phi_pt,phi_hat); t['psi_steric']+=_torque(Fcg,CG,psi_pt,psi_hat)
    cutoff=6.0
    for aname,positions in coords.items():
        if not isinstance(positions,np.ndarray) or positions.ndim!=2: continue
        bq=_BB_CHARGES.get(aname, 0.0)
        for j,pos in enumerate(positions):
            if j==res_idx: continue
            if not np.all(np.isfinite(pos)) or np.linalg.norm(pos)<0.01: continue
            rv=pos-CA; r=np.linalg.norm(rv)
            if r>cutoff or r<0.1: continue
            q=bq if aname in _BB_CHARGES else _SC_CHARGE.get(sequence[j],0.0)
            if abs(q)<1e-4: continue
            for ap,aq in [(N,_BB_CHARGES['N']),(CA,_BB_CHARGES['CA']),(C,_BB_CHARGES['C']),(O,_BB_CHARGES['O'])]:
                if not np.all(np.isfinite(ap)): continue
                dr=ap-pos; dist=np.linalg.norm(dr)
                if dist<0.5: continue
                Fe=aq*q/(dist**2)*_unit(dr)
                t['phi_elec']+=_torque(Fe,ap,phi_pt,phi_hat); t['psi_elec']+=_torque(Fe,ap,psi_pt,psi_hat)
    tpn=sum(t[k] for k in ['phi_bb_donor','phi_bb_acc','phi_sc_hb','phi_steric','phi_elec'])
    tqn=sum(t[k] for k in ['psi_bb_donor','psi_bb_acc','psi_sc_hb','psi_steric','psi_elec'])
    # NEW: per-bond detail for 2-spring decomposition
    hb_details = []
    for bond in bonds:
        if bond['donor'] == res_idx or bond['acceptor'] == res_idx:
            vec = np.array(bond['vec_HO'])
            hb_details.append({
                'd_HO':      round(float(np.linalg.norm(vec)), 3),
                'energy':    round(float(bond['energy']), 4),
                'category':  bond['category'],
                'role':      'donor' if bond['donor'] == res_idx else 'acceptor',
                'partner':   bond['acceptor'] if bond['donor'] == res_idx else bond['donor'],
                # individual torques BEFORE summing
                'tau_phi':   round(_torque(
                                 abs(bond['energy']) * (vec if bond['donor']==res_idx else -vec),
                                 H if bond['donor']==res_idx else coords['O'][res_idx],
                                 phi_pt, phi_hat), 5),
                'tau_psi':   round(_torque(
                                 abs(bond['energy']) * (vec if bond['donor']==res_idx else -vec),
                                 H if bond['donor']==res_idx else coords['O'][res_idx],
                                 psi_pt, psi_hat), 5),
            })
    return {
        'chi1_rad': round(chi1,5) if chi1 is not None else 0.0,
        'has_chi1': int(chi1 is not None and res not in ('GLY','ALA')),
        'tau_phi_correct':round(tpn,5), 'tau_psi_correct':round(tqn,5),
        'tau_phi_bb_donor':round(t['phi_bb_donor'],5), 'tau_psi_bb_donor':round(t['psi_bb_donor'],5),
        'tau_phi_bb_acc':round(t['phi_bb_acc'],5), 'tau_psi_bb_acc':round(t['psi_bb_acc'],5),
        'tau_phi_sc_hb':round(t['phi_sc_hb'],5), 'tau_psi_sc_hb':round(t['psi_sc_hb'],5),
        'tau_phi_steric':round(t['phi_steric'],5), 'tau_psi_steric':round(t['psi_steric'],5),
        'tau_phi_elec_corr':round(t['phi_elec'],5), 'tau_psi_elec_corr':round(t['psi_elec'],5),
        'hb_details': hb_details,   # list of dicts, one per bond
    }


# ══════════════════════════════════════════════════════════════════════════════
# GROUP A: Steric hindrance field (NEW in v5)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_steric_field(res_idx, coords, sequence, n_res, R):
    N  = coords['N'][res_idx]
    CA = coords['CA'][res_idx]
    C  = coords['C'][res_idx]
    O  = coords['O'][res_idx]
    bb_atoms = {'N': N, 'CA': CA, 'C': C, 'O': O}
    feat = {}
    rot_rad  = np.radians(10.0)   # 20° — detects real steric blockage
    clash_r  = 2.5

    # ── Collect atom pools ───────────────────────────────────────────────────
    # full_env  : everything except residue i itself  (for A1 shell counts)
    # far_env   : excluding i±1 as well               (for A2 asymmetry)
    # sc_self   : sidechain atoms of residue i         (for psi obstacle)
    exclude_self    = {res_idx}
    exclude_bonded  = {max(0, res_idx-1), res_idx, min(n_res-1, res_idx+1)}

    full_env_list, far_env_list, sc_self_list = [], [], []
    phi_moving_list, psi_moving_list          = [], []

    for aname, positions in coords.items():
        if not isinstance(positions, np.ndarray) or positions.ndim != 2:
            continue
        for j in range(len(positions)):
            pos = positions[j]
            if not np.all(np.isfinite(pos)) or np.linalg.norm(pos) < 0.01:
                continue

            # Environment pools
            if j not in exclude_self:
                full_env_list.append(pos)
            if j not in exclude_bonded:
                far_env_list.append(pos)

            # Moving-atom pools (residue i only)
            if j == res_idx:
                is_backbone = aname in ('N', 'CA', 'C', 'O', 'H')
                is_sidechain = not is_backbone

                if is_sidechain:
                    sc_self_list.append(pos)

                # phi rotation: N is the pivot — CA and everything beyond moves
                if aname not in ('N', 'H'):
                    phi_moving_list.append(pos)   # CA, C, O, CB, CG, ...

                # psi rotation: Cα is the pivot — C and O move (sidechain stays with CA)
                if aname in ('C', 'O'):
                    psi_moving_list.append(pos)

    # i+1 backbone atoms also move during psi (they're on the C-side)
    if res_idx + 1 < n_res:
        for aname in ('N', 'CA', 'C', 'O', 'H'):
            if aname not in coords:
                continue
            positions = coords[aname]
            if not isinstance(positions, np.ndarray) or res_idx+1 >= len(positions):
                continue
            pos = positions[res_idx + 1]
            if np.all(np.isfinite(pos)) and np.linalg.norm(pos) > 0.01:
                psi_moving_list.append(pos)

    full_env = np.array(full_env_list) if full_env_list else np.empty((0, 3))
    far_env  = np.array(far_env_list)  if far_env_list  else np.empty((0, 3))
    phi_moving = np.array(phi_moving_list) if phi_moving_list else np.empty((0, 3))
    psi_moving = np.array(psi_moving_list) if psi_moving_list else np.empty((0, 3))

    # psi environment: far_env + sidechain of residue i (fixed obstacle for C/O)
    psi_env = (np.vstack([far_env, sc_self_list])
               if sc_self_list else far_env)

    # ── A1: Contact shells ───────────────────────────────────────────────────
    if len(full_env) == 0:
        for a in ['N', 'CA', 'C', 'O']:
            for s in [3, 4, 5]:
                feat[f'steric_{a}_{s}A'] = 0
    else:
        for aname, apos in bb_atoms.items():
            dists = np.linalg.norm(full_env - apos, axis=1)
            for shell in [3.0, 4.0, 5.0]:
                feat[f'steric_{aname}_{shell:.0f}A'] = int(np.sum(dists <= shell))

    # ── A2: Directional asymmetry ────────────────────────────────────────────
    if len(far_env) > 0:
        dists_ca = np.linalg.norm(far_env - CA, axis=1)
        mask5 = dists_ca <= 5.0
        if mask5.sum() > 0:
            vecs = far_env[mask5] - CA
            w    = 1.0 / (dists_ca[mask5]**2 + 0.01)
            asym = R @ np.sum(vecs * w[:, np.newaxis], axis=0)
        else:
            asym = np.zeros(3)
    else:
        asym = np.zeros(3)

    feat['steric_asym_x'] = round(float(asym[0]), 4)
    feat['steric_asym_y'] = round(float(asym[1]), 4)
    feat['steric_asym_z'] = round(float(asym[2]), 4)

    # ── A3: Clash counts (corrected direction) ───────────────────────────────
    phi_hat = _unit(CA - N)
    psi_hat = _unit(C - CA)

    def _rodrigues(points, pivot, axis, angle):
        if len(points) == 0:
            return points
        r   = points - pivot
        ca_, sa_ = np.cos(angle), np.sin(angle)
        return (pivot
                + r * ca_
                + np.cross(axis, r) * sa_
                + axis * (r @ axis)[:, np.newaxis] * (1 - ca_))

    def _count_clashes(moving, env, pivot, axis, angle):
        if len(moving) == 0 or len(env) == 0:
            return 0
        rotated = _rodrigues(moving, pivot, axis, angle)
        # Vectorised: min dist from each rotated atom to any env atom
        # shape: (n_moving, n_env)
        diff = rotated[:, np.newaxis, :] - env[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        return int(np.sum(dists.min(axis=1) < clash_r))

    feat['steric_clash_phi_plus']  = _count_clashes(phi_moving, far_env,  N,  phi_hat,  rot_rad)
    feat['steric_clash_phi_minus'] = _count_clashes(phi_moving, far_env,  N,  phi_hat, -rot_rad)
    feat['steric_clash_psi_plus']  = _count_clashes(psi_moving, psi_env, CA,  psi_hat,  rot_rad)
    feat['steric_clash_psi_minus'] = _count_clashes(psi_moving, psi_env, CA,  psi_hat, -rot_rad)

    # ── A4: Improper dihedral at Cα ──────────────────────────────────────────
    try:
        fourth = None
        for aname in ('H', 'CB'):      # prefer H; fall back to CB
            if aname in coords:
                p = coords[aname][res_idx]
                if np.all(np.isfinite(p)) and np.linalg.norm(p) > 0.01:
                    fourth = p
                    break
        if fourth is not None:
            v1 = _unit(N - CA)
            v2 = _unit(fourth - CA)
            angle_deg = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
            feat['improper_ca'] = round(angle_deg - 109.5, 3)
        else:
            feat['improper_ca'] = 0.0
    except Exception:
        feat['improper_ca'] = 0.0

    # ── A5: Neighbour SC → my backbone min distance ──────────────────────────
    for offset, tag in [(-1, 'nm1'), (+1, 'np1')]:
        j = res_idx + offset
        min_d = 99.0
        if 0 <= j < n_res:
            for aname, positions in coords.items():
                if aname in ('N', 'CA', 'C', 'O', 'H'):
                    continue
                if not isinstance(positions, np.ndarray) or positions.ndim != 2:
                    continue
                if j >= len(positions):
                    continue
                pos = positions[j]
                if not np.all(np.isfinite(pos)) or np.linalg.norm(pos) < 0.01:
                    continue
                for bp in [N, CA, C, O]:
                    min_d = min(min_d, float(np.linalg.norm(pos - bp)))
        feat[f'sc_contact_{tag}_to_bb'] = round(min_d, 3)

    return feat

# ══════════════════════════════════════════════════════════════════════════════
# Per-residue feature extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_residue_features(res_idx, coords, sequence, phi, psi,
                              omega, bond_angles, bonds, chi_list,
                              pdb_id='', bfactors=None):
    N_all=coords['N']; CA_all=coords['CA']; C_all=coords['C']
    CB_all=coords.get('CB', None)
    n_res=len(sequence)
    ni=N_all[res_idx]; cai=CA_all[res_idx]; ci=C_all[res_idx]
    cbi=CB_all[res_idx] if CB_all is not None else np.full(3, np.nan)
    phi_i=float(np.degrees(phi[res_idx])); psi_i=float(np.degrees(psi[res_idx]))
    omega_deg = float(np.degrees(omega[res_idx])) if omega is not None else 180.0
    res = sequence[res_idx]
    R = _local_frame(ni, cai, ci)

    # Metadata + targets
    f = {'pdb_id':pdb_id, 'res_idx':res_idx, 'res_name':res, 'res_type':_RES_IDX.get(res,20),
         'phi_deg':round(phi_i,3), 'psi_deg':round(psi_i,3), 'omega_deg':round(omega_deg,3),
         'ss_bin':_ss_bin(phi_i, psi_i)}

    # GROUP A: Steric field
    f.update(_compute_steric_field(res_idx, coords, sequence, n_res, R))

    # GROUP B: Forces
    f.update(_compute_torques(res_idx, coords, bonds, sequence, chi_list))
    hbd = f.pop('hb_details')   # extract before writing CSV

    # Cancellation signal — the key new quantity
    phi_torques = [b['tau_phi'] for b in hbd]
    psi_torques = [b['tau_psi'] for b in hbd]
    
    f['hb_n_bonds']          = len(hbd)
    f['hb_tau_phi_sum']      = round(sum(phi_torques), 5)   # = existing tau_phi_bb_*
    f['hb_tau_phi_rms']      = round(float(np.sqrt(np.mean(np.array(phi_torques)**2))), 5) if hbd else 0.0
    f['hb_tau_phi_cancel']   = round(float(np.std(phi_torques)), 5) if len(hbd) > 1 else 0.0
    f['hb_tau_psi_rms']      = round(float(np.sqrt(np.mean(np.array(psi_torques)**2))), 5) if hbd else 0.0
    f['hb_tau_psi_cancel']   = round(float(np.std(psi_torques)), 5) if len(hbd) > 1 else 0.0
    f['hb_mean_d_HO']        = round(float(np.mean([b['d_HO'] for b in hbd])), 3) if hbd else 0.0
    f['hb_best_d_HO']        = round(min((b['d_HO'] for b in hbd), default=0.0), 3)
    f['hb_mean_energy']      = round(float(np.mean([b['energy'] for b in hbd])), 4) if hbd else 0.0
    f['hb_n_donor']          = sum(1 for b in hbd if b['role'] == 'donor')
    f['hb_n_acceptor']       = sum(1 for b in hbd if b['role'] == 'acceptor')

    # GROUP C: Context
    chi2 = None
    if chi_list and res_idx<len(chi_list) and chi_list[res_idx] and len(chi_list[res_idx])>=2:
        chi2 = float(chi_list[res_idx][1])
    f['chi2_rad'] = round(chi2,5) if chi2 is not None else 0.0
    f['has_chi2'] = int(chi2 is not None and res in _HAS_CHI2)

    sc_mass=_SIDECHAIN_MASS.get(res,0.0); sc_nh=_SC_N_HEAVY.get(res,0); sc_nr=_SC_N_ROTATABLE.get(res,0)
    f['sc_mass']=sc_mass; f['sc_n_heavy']=sc_nh; f['sc_n_rotatable']=sc_nr
    f['sc_rigidity']=round(sc_nh/max(sc_nr,1),2)
    f['sc_is_branched']=int(res in _SC_BRANCHED); f['sc_is_aromatic']=int(res in _SC_AROMATIC)

    sc_lever = 0.0
    if np.all(np.isfinite(cbi)) and np.linalg.norm(cbi)>0.01:
        sc_lever = float(np.linalg.norm(cbi-cai))
        chi1_v = None
        if chi_list and res_idx<len(chi_list) and chi_list[res_idx]: chi1_v=float(chi_list[res_idx][0])
        if chi1_v is not None and CB_all is not None:
            CG = _cg_position(cai, cbi, chi1_v)
            if CG is not None and np.all(np.isfinite(CG)):
                sc_lever = max(sc_lever, float(np.linalg.norm(CG-cai)))
    f['sc_lever_arm'] = round(sc_lever, 3)

    hb_all = [b for b in bonds if b['donor']==res_idx or b['acceptor']==res_idx]
    f['hb_n_strong'] = sum(1 for b in hb_all if b['category']=='strong')
    f['hb_best_e'] = round(min((b['energy'] for b in hb_all), default=0.0), 4)

    f['bfactor_ca'] = round(float(bfactors['CA'][res_idx]),2) if bfactors and 'CA' in bfactors else 0.0
    f['is_pro_np1'] = int(res_idx+1<n_res and sequence[res_idx+1]=='PRO')

    if bond_angles is not None and res_idx < len(bond_angles[0]):
        f['angle_NCaC']=round(float(np.degrees(bond_angles[0][res_idx])),3)
        f['angle_CaCN']=round(float(np.degrees(bond_angles[1][res_idx])),3)
        f['angle_CNCa']=round(float(np.degrees(bond_angles[2][res_idx])),3)
    else:
        f['angle_NCaC']=111.0; f['angle_CaCN']=117.0; f['angle_CNCa']=121.0

    if res_idx>=2: f['dist_ca_m2']=round(float(np.linalg.norm(CA_all[res_idx-2]-cai)),3)
    else: f['dist_ca_m2']=0.0
    if res_idx<n_res-2: f['dist_ca_p2']=round(float(np.linalg.norm(CA_all[res_idx+2]-cai)),3)
    else: f['dist_ca_p2']=0.0

    f['sc_mass_nm1']=_SIDECHAIN_MASS.get(sequence[res_idx-1],0.0) if res_idx>0 else 0.0
    f['sc_mass_np1']=_SIDECHAIN_MASS.get(sequence[res_idx+1],0.0) if res_idx<n_res-1 else 0.0

    return f


# ── PDB loader ────────────────────────────────────────────────────────────────
def _chain_ok(builder):
    if builder.data is None: return True, []
    breaks = []
    for i in range(len(builder.data['C'])-1):
        d = np.linalg.norm(builder.data['N'][i+1]-builder.data['C'][i])
        if d > 2.5: breaks.append((i,i+1,round(float(d),2)))
    return len(breaks)==0, breaks

def load_pdb_features(pdb_path):
    pdb_id = Path(pdb_path).stem.upper()
    print(f"  {pdb_id} ...", end=' ', flush=True)
    try: builder = NeRFBuilder(pdb_file=pdb_path)
    except Exception as e: print(f"SKIP ({e})"); return []
    if builder.data is None: print("SKIP (no backbone)"); return []
    n_res = builder.n_res
    if n_res < 5: print(f"SKIP ({n_res} res)"); return []
    for atom in ('N','CA','C','O'):
        if atom in builder.data and len(builder.data[atom])!=n_res:
            print(f"SKIP (mismatch {atom})"); return []
    ok, breaks = _chain_ok(builder)
    if not ok: i,j,d=breaks[0]; print(f"SKIP (break {i}→{j} {d}Å)"); return []
    try: phi,psi,omega,chi_list,angles,_ = builder.extract_angles()
    except Exception as e: print(f"SKIP ({e})"); return []
    if len(phi)!=n_res: print("SKIP (angle mismatch)"); return []
    bfactors = _extract_bfactors(builder)
    bond_angles = None
    if angles is not None:
        if isinstance(angles,(tuple,list)) and len(angles)>=3: bond_angles=angles[:3]
        elif isinstance(angles,np.ndarray) and angles.ndim==2 and angles.shape[1]>=3:
            bond_angles=(angles[:,0],angles[:,1],angles[:,2])
    # NEW:
    coords = builder.build(phi, psi, chi_list, angles)
    sequence = builder.sequence
    
    # Ensure H is present for every residue (except Proline)
    if 'H' not in coords:
        coords['H'] = np.full((n_res, 3), np.nan)
    
    for i in range(1, n_res):
        if sequence[i] == 'PRO':
            continue
        # If builder didn't place H, use molcore/nerf logic here
        if np.any(np.isnan(coords['H'][i])):
            # You can call a placement function here
            pass
        
    bonds=find_hbonds(coords,sequence,e_threshold=E_WEAK)
    ns=sum(1 for b in bonds if b['category']=='strong')
    nw=sum(1 for b in bonds if b['category']=='weak')
    print(f"{n_res}res {ns}s+{nw}w HB{' +Bf' if bfactors else ''}")
    rows = []
    for i in range(1, n_res-1):
        try:
            rows.append(extract_residue_features(
                i, coords, sequence, phi, psi, omega, bond_angles,
                bonds, chi_list, pdb_id=pdb_id, bfactors=bfactors))
        except Exception as e: print(f"    Warn res {i} ({sequence[i]}): {e}")
    return rows

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--pdb', type=str); g.add_argument('--pdb_dir', type=str)
    ap.add_argument('--out', default='features_v5.csv')
    ap.add_argument('--max_pdbs', type=int, default=None)
    args = ap.parse_args()
    pdb_files = [args.pdb] if args.pdb else sorted(Path(args.pdb_dir).glob('*.pdb'))
    if args.max_pdbs: pdb_files=pdb_files[:args.max_pdbs]
    print(f"\nCollecting from {len(pdb_files)} PDB(s) → {args.out}\n")
    all_rows = []
    for p in pdb_files: all_rows.extend(load_pdb_features(str(p)))
    if not all_rows: print("No data."); sys.exit(1)
    with open(args.out,'w',newline='') as fout:
        w=csv.DictWriter(fout, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nDone. {len(all_rows):,} residues → {args.out}")
    ss_n={0:'αR',1:'β',2:'PPII',3:'3₁₀',4:'loop',5:'αL'}
    counts=collections.Counter(r['ss_bin'] for r in all_rows)
    for k,v in sorted(counts.items()): print(f"  {ss_n.get(k,'?'):6s}: {v:6,} ({100*v/len(all_rows):.1f}%)")
    tp=[r['tau_phi_correct'] for r in all_rows if isinstance(r['tau_phi_correct'],(int,float))]
    tq=[r['tau_psi_correct'] for r in all_rows if isinstance(r['tau_psi_correct'],(int,float))]
    if tp: print(f"\n  τ_φ: mean={np.mean(tp):+.3f} std={np.std(tp):.3f}")
    if tq: print(f"  τ_ψ: mean={np.mean(tq):+.3f} std={np.std(tq):.3f}")
    for s in [3,4,5]:
        v=[r[f'steric_CA_{s}A'] for r in all_rows]; print(f"  steric_CA_{s}Å: mean={np.mean(v):.1f} max={max(v)}")
    for d in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
        v = [r[f'steric_clash_{d}'] for r in all_rows]
        print(f"  clash_{d}: mean={np.mean(v):.1f} max={max(v)}")
    n1=sum(1 for r in all_rows if r['has_chi1']); n2=sum(1 for r in all_rows if r['has_chi2'])
    print(f"\n  χ₁: {n1:,}/{len(all_rows):,} ({100*n1/len(all_rows):.0f}%)")
    print(f"  χ₂: {n2:,}/{len(all_rows):,} ({100*n2/len(all_rows):.0f}%)")
    om=[r['omega_deg'] for r in all_rows]
    print(f"  ω: mean={np.mean(om):.1f}° std={np.std(om):.1f}°")
    meta={'pdb_id','res_idx','res_name','res_type','phi_deg','psi_deg','omega_deg','ss_bin'}
    fc=len([k for k in all_rows[0] if k not in meta])
    print(f"\n  Input features: {fc}  |  Outputs: φ, ψ, ω → 6 sin/cos targets")

if __name__ == '__main__':
    main()