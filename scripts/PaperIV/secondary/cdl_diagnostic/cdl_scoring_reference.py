#!/usr/bin/env python3
"""Ten-term post-2020 crystal-coordinate strain comparison against cctbx CDL."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from holdout_reference import AMBER, RES, R, B, rows, sha
import holdout_reference as ref

from mmtbx.conformation_dependent_library import cdl_setup, cdl_utils, utils
from mmtbx.conformation_dependent_library.cdl_database import cdl_database
cdl_module=importlib.import_module('mmtbx.conformation_dependent_library.cdl_database')


# Indices into the installed 26-field CDL row; verified against cdl_setup.headers.
TERMS_TO_COLUMN = {
    'tau_deg': (6, 'N(0)  - Ca(0) - C(0)'),
    'angle_N_CA_CB': (4, 'N(0)  - Ca(0) - Cb(0)'),
    'angle_C_CA_CB': (8, 'Cb(0) - Ca(0) - C(0)'),
    'angle_CaCN': (12, 'Ca(0) - C(0)  - N(+1)'),
    'angle_CNCa': (2, 'C(-1) - N(0)  - Ca(0)'),
    'angle_CA_C_O': (10, 'Ca(0) - C(0)  - O(0)'),
    'bond_N_CA': (18, 'N(0)  - Ca(0)'),
    'bond_CA_C': (22, 'Ca(0) - C(0)'),
    'bond_C_O': (24, 'C(0)  - O(0)'),
    'bond_CA_CB': (20, 'Ca(0) - Cb(0)'),
}
TERMS = list(TERMS_TO_COLUMN)
ARMS = ['AMBER_FIXED','TRAIN_GLOBAL_MEAN','TRAIN_RESIDUE_MEAN',
        'TRAIN_PHI_PSI_MECHLIB','CCTBX_CDL_2026_7']
SOURCE_SHA256 = '79c197e5692986cde26db9aa492cecfb5b8b7d5b1f3cba4dc222bb2e04d23c68'


def validate_implementation():
    assert importlib.metadata.version('cctbx-base') == '2026.7'
    assert sha(Path(cdl_module.__file__)) == SOURCE_SHA256, 'CDL database differs from audited 2026.7 version'
    assert len(cdl_database) == 8
    for term,(idx,header) in TERMS_TO_COLUMN.items():
        assert cdl_setup.headers[idx] == header, (term, idx, cdl_setup.headers[idx])
    assert all(len(table)==1296 for table in cdl_database.values())
    assert 'bond_C_N_next' not in TERMS


def class_code(data, class_lookup):
    flag = data['is_pro_np1'].astype(str).str.strip().str.lower()
    numeric = {'true':True,'1':True,'1.0':True,'false':False,'0':False,'0.0':False}
    if not flag.isin(numeric).all():
        examples = flag[~flag.isin(numeric)].value_counts().head(5).to_dict()
        raise ValueError(f'Unknown is_pro_np1 values: {examples}')
    return np.fromiter((class_lookup[(res,bool(numeric[f]))]
                        for res,f in zip(data['res_name'],flag)),dtype=np.int16,count=len(data))


def build_cdl():
    classes = sorted(cdl_database)
    labels = {(res, p): cdl_utils.get_res_type_group(res, 'PRO' if p else 'ALA')
              for res in RES for p in (False,True)}
    if any(v not in classes for v in labels.values()):
        raise ValueError(f'Unmapped residue classes: {labels}')
    lookup = {key:classes.index(value) for key,value in labels.items()}
    matrix = np.full((len(classes), B, B, len(TERMS)), np.nan)
    flags = {}
    for i,name in enumerate(classes):
        table=cdl_database[name]
        for a in range(B):
            for z in range(B):
                row=table[(-180+10*a,-180+10*z)]
                assert len(row)==26
                flags[row[0]]=flags.get(row[0],0)+1
                for j,term in enumerate(TERMS):
                    value=float(row[TERMS_TO_COLUMN[term][0]])
                    if value>0:
                        matrix[i,a,z,j]=value
    return matrix,lookup,labels,flags


def cdl_targets(data,matrix,class_lookup):
    # Nearest CDL grid, equivalent to default cctbx interpolate=False.
    phi = pd.to_numeric(data['phi_deg'],errors='raise').to_numpy(dtype=float)
    psi = pd.to_numeric(data['psi_deg'],errors='raise').to_numpy(dtype=float)
    key_phi = np.fromiter((utils.round_to_ten(float(x)) for x in phi),dtype=np.int32,count=len(data))
    key_psi = np.fromiter((utils.round_to_ten(float(x)) for x in psi),dtype=np.int32,count=len(data))
    x=((key_phi+180)//10)%B
    y=((key_psi+180)//10)%B
    return matrix[class_code(data,class_lookup),x,y,:]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',type=Path,required=True)
    ap.add_argument('--dates',type=Path,required=True)
    ap.add_argument('--baseline-report',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--chunk-size',type=int,default=100000)
    args=ap.parse_args()
    if args.preflight==args.execute:
        ap.error('select exactly one of --preflight / --execute')
    validate_implementation()
    original=json.loads(args.baseline_report.read_text())
    if original['status']!='PASS' or original['csv_size_bytes']!=args.csv.stat().st_size or original['dates_sha256']!=sha(args.dates):
        raise ValueError('Preceding all-term baseline replication or input provenance mismatch')
    d=pd.read_csv(args.dates)
    assert not d.pdb_id.duplicated().any()
    dates=pd.Series(pd.to_datetime(d.release_date).values,index=d.pdb_id)
    matrix,class_lookup,labels,flags=build_cdl()
    ref.TERMS=TERMS
    ref.EXPECTED_COLUMNS={'pdb_id','res_name','phi_deg','psi_deg','is_pro_np1',*TERMS}
    if not ref.EXPECTED_COLUMNS.issubset(pd.read_csv(args.csv,nrows=0).columns):
        raise ValueError('Required geometry or next-Pro field absent')
    if args.preflight:
        data=pd.read_csv(args.csv,nrows=100,usecols=sorted(ref.EXPECTED_COLUMNS))
        class_code(data,class_lookup)
        print(json.dumps({'status':'CDL_PREFLIGHT_PASS','matched_terms':TERMS,
                          'unmatched_term':'bond_C_N_next: CDL stores C(previous)-N(current) at the next residue phi/psi',
                          'class_names':sorted(cdl_database),'n_example_rows':len(data),
                          'database_sha256':SOURCE_SHA256},indent=2))
        return
    if args.out.exists():
        raise FileExistsError(f'Output already exists: {args.out}')
    if args.chunk_size<1:
        raise ValueError('Positive chunk size required')
    n=R*B*B
    group_size=np.zeros(n,dtype=np.int64)
    sums=np.zeros((len(TERMS),n)); nobs=np.zeros_like(sums,dtype=np.int64)
    pooled_sum=np.zeros((len(TERMS),B*B)); pooled_n=np.zeros_like(pooled_sum,dtype=np.int64)
    type_sum=np.zeros((len(TERMS),R)); type_n=np.zeros_like(type_sum,dtype=np.int64)
    global_sum=np.zeros(len(TERMS));global_n=np.zeros(len(TERMS),dtype=np.int64)
    cutoff=pd.Timestamp('2020-01-01')
    print('PASS 1/2: train matched ten-term MechLib and mean baselines',flush=True)
    for data,g,p,r,train in rows(args.csv,dates,cutoff,args.chunk_size):
        if not train.any(): continue
        g,p,r=g[train],p[train],r[train]
        group_size+=np.bincount(g,minlength=n)
        for j,term in enumerate(TERMS):
            v=pd.to_numeric(data.loc[train,term],errors='coerce').to_numpy(dtype=float)
            ok=np.isfinite(v)
            if not ok.any():continue
            nobs[j]+=np.bincount(g[ok],minlength=n)
            sums[j]+=np.bincount(g[ok],weights=v[ok],minlength=n)
            pooled_n[j]+=np.bincount(p[ok],minlength=B*B)
            pooled_sum[j]+=np.bincount(p[ok],weights=v[ok],minlength=B*B)
            type_n[j]+=np.bincount(r[ok],minlength=R)
            type_sum[j]+=np.bincount(r[ok],weights=v[ok],minlength=R)
            global_sum[j]+=v[ok].sum();global_n[j]+=int(ok.sum())
    if (global_n==0).any():raise ValueError('Missing a training term')
    global_eq=global_sum/global_n
    acc={s:{arm:{'strain_sum':0.,'residue_rows':0} for arm in ARMS} for s in ('train','test')}
    coverage={s:{term:{'observed':0,'cdl_scored':0} for term in TERMS} for s in ('train','test')}
    sets={s:set() for s in ('train','test')}
    counts={'train':0,'test':0}
    test_by_structure={}
    print('PASS 2/2: score five arms on the same available ten-term measurements',flush=True)
    for data,g,p,r,train in rows(args.csv,dates,cutoff,args.chunk_size):
        target=cdl_targets(data,matrix,class_lookup)
        energies={arm:np.zeros(len(data)) for arm in ARMS}
        for j,term in enumerate(TERMS):
            obs=pd.to_numeric(data[term],errors='coerce').to_numpy(dtype=float)
            seen=np.isfinite(obs)
            good=seen & np.isfinite(target[:,j])
            for scope,mask in (('train',train),('test',~train)):
                coverage[scope][term]['observed']+=int((seen & mask).sum())
                coverage[scope][term]['cdl_scored']+=int((good & mask).sum())
            if not good.any():continue
            fixed,k,angle=AMBER[term]
            coeff=0.5*k*((np.pi/180)**2 if angle else 1.)
            global_target=global_eq[j]
            by_type=np.divide(type_sum[j,r],type_n[j,r],out=np.full(len(data),global_target),where=type_n[j,r]>0)
            pool=np.divide(pooled_sum[j,p],pooled_n[j,p],out=np.full(len(data),global_target),where=pooled_n[j,p]>0)
            local=np.divide(sums[j,g],nobs[j,g],out=pool.copy(),where=(group_size[g]>=10)&(nobs[j,g]>0))
            for arm,eq in zip(ARMS,(fixed,global_target,by_type,local,target[:,j])):
                energies[arm][good]+=coeff*(obs[good]-(eq if np.isscalar(eq) else eq[good]))**2
        for scope,mask in (('train',train),('test',~train)):
            counts[scope]+=int(mask.sum())
            sets[scope].update(data.loc[mask,'pdb_id'].tolist())
            for arm in ARMS:
                acc[scope][arm]['strain_sum']+=float(energies[arm][mask].sum())
                acc[scope][arm]['residue_rows']+=int(mask.sum())
        if (~train).any():
            ids, ix = np.unique(data.loc[~train,'pdb_id'].astype(str).to_numpy(), return_inverse=True)
            counts_by_id=np.bincount(ix,minlength=len(ids))
            for pos, pdb_id in enumerate(ids):
                target_row=test_by_structure.setdefault(pdb_id, np.zeros(len(ARMS)+1))
                target_row[0]+=counts_by_id[pos]
            for j,arm in enumerate(ARMS):
                sums_by_id=np.bincount(ix,weights=energies[arm][~train],minlength=len(ids))
                for pos,pdb_id in enumerate(ids):
                    test_by_structure[pdb_id][j+1]+=sums_by_id[pos]
    for scope in acc:
        fixed=acc[scope]['AMBER_FIXED']['strain_sum']/counts[scope]
        for arm in ARMS:
            q=acc[scope][arm]
            q['mean_strain']=q['strain_sum']/counts[scope]
            q['reduction_vs_matched_AMBER_percent']=100*(1-q['mean_strain']/fixed)
            q['structures']=len(sets[scope])
    tests={'all_term_original_PASS':original['status']=='PASS',
           'train_rows_match_original':counts['train']==original['counts']['train'],
           'test_rows_match_original':counts['test']==original['counts']['test'],
           'train_structures_match_original':len(sets['train'])==original['arms']['train']['AMBER_FIXED']['structure_count'],
           'test_structures_match_original':len(sets['test'])==original['arms']['test']['AMBER_FIXED']['structure_count']}
    ordered_ids=sorted(test_by_structure)
    by_structure=np.array([test_by_structure[k] for k in ordered_ids])
    delta=(by_structure[:,ARMS.index('CCTBX_CDL_2026_7')+1]
           -by_structure[:,ARMS.index('TRAIN_PHI_PSI_MECHLIB')+1])
    n_boot=2000
    rng=np.random.default_rng(20260925)
    idx=rng.integers(0,len(ordered_ids),size=(n_boot,len(ordered_ids)))
    draws=delta[idx].sum(axis=1)/by_structure[idx,0].sum(axis=1)
    cdl_minus_mechlib={
        'definition':'mean CDL strain minus mean MechLib strain; positive favors MechLib',
        'mean_on_matched_rows':float(delta.sum()/by_structure[:,0].sum()),
        'structure_cluster_bootstrap_ci95':np.quantile(draws,[.025,.975]).tolist(),
        'bootstrap_replicates':n_boot,'bootstrap_seed':20260925,
        'status':'exploratory_not_registered_endpoint',
    }
    status='PASS' if all(tests.values()) and all(coverage['test'][t]['cdl_scored']>0 for t in TERMS) else 'AUDIT_REQUIRED'
    report={'status':status,'scope':'exploratory_crystal_coordinate_ten_term_comparison',
            'cdl_module_sha256':SOURCE_SHA256,'cdl_flags_in_grid':flags,
            'columns':{k:v[0] for k,v in TERMS_TO_COLUMN.items()},
            'excluded':['bond_C_N_next: requires next-residue φ/ψ; excluded from every arm'],
            'class_map':{f'{k[0]} next_pro={k[1]}':v for k,v in labels.items()},
            'coverage':coverage,'counts':counts,'checks':tests,'arms':acc,
            'test_cdl_minus_mechlib':cdl_minus_mechlib,
            'baseline_report_sha256':sha(args.baseline_report),
            'interpretation':'CDL table entries for each raw phi/psi nearest 10-degree grid; no 2D interpolation; 10 common terms scored on identical observations per term. Crystal reference and potential refinement imprint persist. This is not QM or NMR validation.'}
    args.out.mkdir(parents=True)
    (args.out/'CDL_MATCHED_SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CDL_MATCHED='+status,'test_reduction=',acc['test']['CCTBX_CDL_2026_7']['reduction_vs_matched_AMBER_percent'],flush=True)
    if status!='PASS':raise SystemExit(2)


if __name__=='__main__':main()
