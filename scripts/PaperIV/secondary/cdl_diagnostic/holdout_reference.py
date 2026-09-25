#!/usr/bin/env python3
"""Two-pass, bounded-memory replication and pre-2020 mean baselines for Paper IV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

AMBER = {
    'tau_deg': (110.10, 63., True), 'angle_N_CA_CB': (109.70, 80., True),
    'angle_C_CA_CB': (111.10, 63., True), 'angle_CaCN': (116.6, 70., True),
    'angle_CNCa': (121.9, 50., True), 'angle_CA_C_O': (120.4, 80., True),
    'bond_N_CA': (1.449, 337., False), 'bond_CA_C': (1.522, 317., False),
    'bond_C_O': (1.229, 570., False), 'bond_C_N_next': (1.335, 490., False),
    'bond_CA_CB': (1.526, 317., False),
}
TERMS = list(AMBER)
RES = sorted('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split())
R = len(RES)
B = 36
ARMS = ['AMBER_FIXED', 'TRAIN_GLOBAL_MEAN', 'TRAIN_RESIDUE_MEAN', 'TRAIN_PHI_PSI_MECHLIB']
EXPECTED_COLUMNS = {'pdb_id', 'res_name', 'phi_deg', 'psi_deg', *TERMS}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def rows(path, dates, cutoff, chunk_size):
    for raw in pd.read_csv(path, usecols=sorted(EXPECTED_COLUMNS), chunksize=chunk_size,
                           low_memory=False):
        phi = pd.to_numeric(raw['phi_deg'], errors='coerce').to_numpy(dtype=float)
        psi = pd.to_numeric(raw['psi_deg'], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(phi) & np.isfinite(psi) & (phi >= -180) & (phi < 180) & (psi >= -180) & (psi < 180)
        if not valid.any():
            continue
        data = raw.loc[valid].copy()
        pb = np.floor((phi[valid] + 180) / 10).astype(np.int32)
        qb = np.floor((psi[valid] + 180) / 10).astype(np.int32)
        # Same left-closed 10-degree bins and cutoff as the archived script.
        release = data['pdb_id'].map(dates)
        dated = release.notna().to_numpy()
        if not dated.any():
            continue
        data = data.loc[dated]
        pb, qb = pb[dated], qb[dated]
        release = release.loc[dated]
        residue = pd.Categorical(data['res_name'], categories=RES).codes.astype(np.int32)
        if (residue < 0).any():
            bad = sorted(set(data.loc[residue < 0, 'res_name'].astype(str)))
            raise ValueError(f'Unrecognized residue names: {bad}; stop and audit mapping')
        groups = residue * B * B + qb * B + pb
        pooled = qb * B + pb
        train = (release < cutoff).to_numpy()
        yield data, groups, pooled, residue, train


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--dates', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cutoff', default='2020-01-01')
    ap.add_argument('--chunk-size', type=int, default=100000)
    ap.add_argument('--min-count', type=int, default=10)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    if args.preflight == args.execute:
        ap.error('choose exactly one of --preflight / --execute')
    for path in (args.csv, args.dates):
        if not path.is_file():
            raise FileNotFoundError(path)
    header = set(pd.read_csv(args.csv, nrows=0).columns)
    if not EXPECTED_COLUMNS.issubset(header):
        raise ValueError(f'Missing CSV columns: {EXPECTED_COLUMNS - header}')
    d = pd.read_csv(args.dates)
    if not {'pdb_id', 'release_date'}.issubset(d):
        raise ValueError('dates requires pdb_id,release_date')
    if d['pdb_id'].duplicated().any():
        raise ValueError('Duplicate release-date PDB ids')
    dates = pd.Series(pd.to_datetime(d['release_date'], errors='raise').values, index=d['pdb_id'])
    cutoff = pd.Timestamp(args.cutoff)
    if args.preflight:
        print(json.dumps({'status':'PREFLIGHT_PASS', 'csv_bytes':args.csv.stat().st_size,
                          'csv_header_terms':TERMS, 'date_rows':len(d), 'date_sha256':sha(args.dates),
                          'cutoff':str(cutoff.date()), 'full_csv_not_scanned':True}, indent=2))
        return
    if args.out.exists():
        raise FileExistsError(f'Output already exists: {args.out}')
    if args.min_count < 1 or args.chunk_size < 1:
        raise ValueError('Positive min-count and chunk-size required')
    n_group = R*B*B
    size_res = np.zeros(n_group, dtype=np.int64)
    sums_res = np.zeros((len(TERMS), n_group))
    counts_res = np.zeros_like(sums_res, dtype=np.int64)
    sums_pool = np.zeros((len(TERMS), B*B))
    counts_pool = np.zeros_like(sums_pool, dtype=np.int64)
    sums_type = np.zeros((len(TERMS), R))
    counts_type = np.zeros_like(sums_type, dtype=np.int64)
    total_sum = np.zeros(len(TERMS))
    total_count = np.zeros(len(TERMS), dtype=np.int64)
    counts = {'train':0,'test':0}
    structures = {'train':set(),'test':set()}
    print('PASS 1/2: fit means using dated, binned pre-cutoff training rows', flush=True)
    for data, groups, pooled, residue, train in rows(args.csv, dates, cutoff, args.chunk_size):
        if not train.any():
            continue
        counts['train'] += int(train.sum())
        structures['train'].update(data.loc[train, 'pdb_id'].tolist())
        structures['test'].update(data.loc[~train, 'pdb_id'].tolist())
        g, p, r = groups[train], pooled[train], residue[train]
        size_res += np.bincount(g, minlength=n_group)
        for j, term in enumerate(TERMS):
            values = pd.to_numeric(data.loc[train, term], errors='coerce').to_numpy(dtype=float)
            valid = np.isfinite(values)
            if not valid.any():
                continue
            v = values[valid]
            counts_res[j] += np.bincount(g[valid], minlength=n_group)
            sums_res[j] += np.bincount(g[valid], weights=v, minlength=n_group)
            counts_pool[j] += np.bincount(p[valid], minlength=B*B)
            sums_pool[j] += np.bincount(p[valid], weights=v, minlength=B*B)
            counts_type[j] += np.bincount(r[valid], minlength=R)
            sums_type[j] += np.bincount(r[valid], weights=v, minlength=R)
            total_sum[j] += v.sum()
            total_count[j] += len(v)
    if (total_count == 0).any():
        raise ValueError('At least one geometry term has no training observations')
    global_eq = total_sum / total_count
    totals = {scope: {arm: {'strain_sum':0.,'residue_rows':0} for arm in ARMS}
              for scope in ('train','test')}
    print('PASS 2/2: score identical rows for all four arms, train and test', flush=True)
    for data, groups, pooled, residue, train in rows(args.csv, dates, cutoff, args.chunk_size):
        counts['test'] += int((~train).sum())
        structures['train'].update(data.loc[train, 'pdb_id'].tolist())
        structures['test'].update(data.loc[~train, 'pdb_id'].tolist())
        e = {arm:np.zeros(len(data)) for arm in ARMS}
        for j, term in enumerate(TERMS):
            obs = pd.to_numeric(data[term], errors='coerce').to_numpy(dtype=float)
            valid = np.isfinite(obs)
            if not valid.any():
                continue
            fixed, k, angle = AMBER[term]
            coeff = 0.5*k*((np.pi/180)**2 if angle else 1.)
            global_target = global_eq[j]
            type_target = np.divide(sums_type[j, residue], counts_type[j, residue],
                                    out=np.full(len(data), global_target),
                                    where=counts_type[j, residue]>0)
            pool_target = np.divide(sums_pool[j, pooled], counts_pool[j, pooled],
                                    out=np.full(len(data), global_target),
                                    where=counts_pool[j, pooled]>0)
            local = np.divide(sums_res[j, groups], counts_res[j, groups],
                              out=pool_target.copy(),
                              where=(size_res[groups]>=args.min_count) & (counts_res[j, groups]>0))
            for arm, target in zip(ARMS, (fixed, global_target, type_target, local)):
                e[arm][valid] += coeff * (obs[valid] - (target if np.isscalar(target) else target[valid]))**2
        for scope, mask in (('train',train), ('test',~train)):
            for arm in ARMS:
                totals[scope][arm]['strain_sum'] += float(e[arm][mask].sum())
                totals[scope][arm]['residue_rows'] += int(mask.sum())
    for scope in totals:
        for arm in ARMS:
            q = totals[scope][arm]
            q['mean_strain'] = q['strain_sum']/q['residue_rows'] if q['residue_rows'] else None
            q['structure_count'] = len(structures[scope])
            q['reduction_vs_AMBER_percent'] = 100*(1-q['mean_strain']/totals[scope]['AMBER_FIXED']['mean_strain'])
    checks = {'train_reduction_rounded_1_decimal_matches_38_7':
              round(totals['train']['TRAIN_PHI_PSI_MECHLIB']['reduction_vs_AMBER_percent'],1)==38.7,
              'test_reduction_rounded_1_decimal_matches_40_6':
              round(totals['test']['TRAIN_PHI_PSI_MECHLIB']['reduction_vs_AMBER_percent'],1)==40.6,
              'train_rows_match_archive': counts['train']==1269812,
              'test_rows_match_archive': counts['test']==428750}
    status = 'PASS' if all(checks.values()) else 'REPLICATION_MISMATCH_DO_NOT_INTERPRET_BASELINES'
    report = {'status':status,'scope':'same-row pre-2020-trained geometry-strain baselines',
              'cutoff':args.cutoff,'min_count':args.min_count,'k_angle_N_CA_CB':80,
              'csv_path':str(args.csv.resolve()),'csv_size_bytes':args.csv.stat().st_size,
              'csv_sha256_not_computed':True,'dates_sha256':sha(args.dates),
              'counts':counts,'checks':checks,'arms':totals,
              'limitations':'Crystal-coordinate strain is not independent force-field validation; global and residue means use training coordinates only.'}
    args.out.mkdir(parents=True)
    (args.out/'HOLDOUT_BASELINE_SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HOLDOUT_BASELINES='+status, json.dumps(checks), flush=True)
    if status != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
