#!/usr/bin/env python3
"""Post hoc ten-term CDL comparison by geometry term and refinement software."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import cdl_scoring_reference as cdl
import holdout_reference as ref


ARMS = cdl.ARMS + ['REFMAC_TRAIN_PHI_PSI_MECHLIB']
SUBSETS = ['ALL', 'REFMAC', 'PHENIX', 'NONREFMAC_KNOWN', 'UNKNOWN']


def identify(path):
    try:
        program=[]
        with path.open(errors='replace') as handle:
            for line in handle:
                if line.startswith('REMARK   3   PROGRAM'):
                    program.append(line[19:].strip())
                if line.startswith('CRYST1'):
                    break
    except OSError:
        return 'UNKNOWN'
    text=' '.join(program).upper()
    if 'PHENIX' in text: return 'PHENIX'
    if 'REFMAC' in text: return 'REFMAC'
    if text: return 'OTHER_KNOWN'
    return 'UNKNOWN'


def prepare_map(pdb_ids,pdb_dir):
    output={}
    for pdb_id in pdb_ids:
        name=str(pdb_id)
        path=pdb_dir/(name+'.pdb')
        if not path.exists():path=pdb_dir/(name.lower()+'.pdb')
        output[pdb_id]=identify(path) if path.is_file() else 'UNKNOWN'
    return output


def new_stats():
    n=ref.R*ref.B*ref.B
    k=len(cdl.TERMS)
    return dict(group_size=np.zeros(n,dtype=np.int64),sums=np.zeros((k,n)),
                counts=np.zeros((k,n),dtype=np.int64),pool_sum=np.zeros((k,ref.B*ref.B)),
                pool_count=np.zeros((k,ref.B*ref.B),dtype=np.int64),
                type_sum=np.zeros((k,ref.R)),type_count=np.zeros((k,ref.R),dtype=np.int64),
                global_sum=np.zeros(k),global_count=np.zeros(k,dtype=np.int64),nrows=0)


def update(stats,data,g,p,r,select):
    if not select.any():return
    stats['nrows']+=int(select.sum())
    g,p,r=g[select],p[select],r[select]
    stats['group_size']+=np.bincount(g,minlength=len(stats['group_size']))
    for j,term in enumerate(cdl.TERMS):
        obs=pd.to_numeric(data.loc[select,term],errors='coerce').to_numpy(dtype=float)
        ok=np.isfinite(obs)
        if not ok.any():continue
        v=obs[ok]
        stats['counts'][j]+=np.bincount(g[ok],minlength=len(stats['group_size']))
        stats['sums'][j]+=np.bincount(g[ok],weights=v,minlength=len(stats['group_size']))
        stats['pool_count'][j]+=np.bincount(p[ok],minlength=ref.B*ref.B)
        stats['pool_sum'][j]+=np.bincount(p[ok],weights=v,minlength=ref.B*ref.B)
        stats['type_count'][j]+=np.bincount(r[ok],minlength=ref.R)
        stats['type_sum'][j]+=np.bincount(r[ok],weights=v,minlength=ref.R)
        stats['global_sum'][j]+=v.sum();stats['global_count'][j]+=len(v)


def targets(stats,j,g,p,r):
    global_eq=stats['global_sum'][j]/stats['global_count'][j]
    by_res=np.divide(stats['type_sum'][j,r],stats['type_count'][j,r],
                     out=np.full(len(g),global_eq),where=stats['type_count'][j,r]>0)
    pool=np.divide(stats['pool_sum'][j,p],stats['pool_count'][j,p],
                   out=np.full(len(g),global_eq),where=stats['pool_count'][j,p]>0)
    local=np.divide(stats['sums'][j,g],stats['counts'][j,g],out=pool.copy(),
                    where=(stats['group_size'][g]>=10)&(stats['counts'][j,g]>0))
    return global_eq,by_res,local


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',type=Path,required=True)
    ap.add_argument('--dates',type=Path,required=True)
    ap.add_argument('--pdb-dir',type=Path,required=True)
    ap.add_argument('--baseline-report',type=Path,required=True)
    ap.add_argument('--cdl-report',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--chunk-size',type=int,default=100000)
    ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    if args.preflight==args.execute:ap.error('select exactly one mode')
    cdl.validate_implementation()
    base=json.loads(args.baseline_report.read_text()); prior=json.loads(args.cdl_report.read_text())
    if base['status']!='PASS' or prior['status']!='PASS':raise ValueError('Upstream PASS reports required')
    if base['csv_size_bytes']!=args.csv.stat().st_size or base['dates_sha256']!=ref.sha(args.dates):
        raise ValueError('Data differs from frozen passed baseline')
    if prior['cdl_module_sha256']!=cdl.SOURCE_SHA256 or prior['baseline_report_sha256']!=ref.sha(args.baseline_report):
        raise ValueError('Prior CDL run provenance mismatch')
    if not args.pdb_dir.is_dir():raise FileNotFoundError(args.pdb_dir)
    date_table=pd.read_csv(args.dates)
    if date_table.pdb_id.duplicated().any():raise ValueError('Duplicate PDB date')
    dates=pd.Series(pd.to_datetime(date_table.release_date).values,index=date_table.pdb_id)
    software=prepare_map(date_table.pdb_id.tolist(),args.pdb_dir)
    by_era={era:{name:int(((date_table.release_date.lt('2020-01-01') if era=='train' else
                             date_table.release_date.ge('2020-01-01')) &
                             date_table.pdb_id.map(software).eq(name)).sum())
                 for name in ('REFMAC','PHENIX','OTHER_KNOWN','UNKNOWN')}
            for era in ('train','test')}
    matrix,classes,_,_=cdl.build_cdl()
    ref.TERMS=cdl.TERMS
    ref.EXPECTED_COLUMNS={'pdb_id','res_name','phi_deg','psi_deg','is_pro_np1',*cdl.TERMS}
    if args.preflight:
        print(json.dumps({'status':'CDL_DIAGNOSTIC_PREFLIGHT_PASS','software_by_era_from_date_inventory':by_era,
                          'pdb_header_rule':'REMARK 3 PROGRAM before CRYST1; PHENIX and REFMAC categories are proxies',
                          'target_terms':cdl.TERMS,'source_sha256':cdl.SOURCE_SHA256},indent=2))
        return
    if args.out.exists():raise FileExistsError(args.out)
    full, rm = new_stats(),new_stats()
    print('PASS 1/2: all-program and REFMAC-only pre-2020 training targets',flush=True)
    for data,g,p,r,train in ref.rows(args.csv,dates,pd.Timestamp('2020-01-01'),args.chunk_size):
        label=data.pdb_id.map(software).fillna('UNKNOWN').to_numpy()
        update(full,data,g,p,r,train)
        update(rm,data,g,p,r,train & (label=='REFMAC'))
    if (full['global_count']==0).any() or (rm['global_count']==0).any():
        raise ValueError('Missing training terms in either training scope')
    summaries={s:{t:{arm:{'strain_sum':0.,'n_observed':0} for arm in ARMS}
                  for t in cdl.TERMS} for s in SUBSETS}
    strata_rows={s:0 for s in SUBSETS}
    by_structure=defaultdict(lambda:np.zeros(3))
    counts={'train':0,'test':0}
    print('PASS 2/2: test on matched terms, split by refinement program',flush=True)
    for data,g,p,r,train in ref.rows(args.csv,dates,pd.Timestamp('2020-01-01'),args.chunk_size):
        counts['train']+=int(train.sum());counts['test']+=int((~train).sum())
        if train.all():continue
        labels=data.pdb_id.map(software).fillna('UNKNOWN').to_numpy()
        try: cdl_target=cdl.cdl_targets(data,matrix,classes)
        except Exception as e:raise ValueError(f'CDL target mapping: {e}') from e
        masks={'ALL':~train,'REFMAC':(~train)&(labels=='REFMAC'),
               'PHENIX':(~train)&(labels=='PHENIX'),
               'NONREFMAC_KNOWN':(~train)&(~np.isin(labels,['REFMAC','UNKNOWN'])),
               'UNKNOWN':(~train)&(labels=='UNKNOWN')}
        for s,mask in masks.items():strata_rows[s]+=int(mask.sum())
        if masks['REFMAC'].any():
            names,nper=np.unique(data.loc[masks['REFMAC'],'pdb_id'].astype(str).to_numpy(),return_counts=True)
            for pdb_id,number in zip(names,nper):by_structure[pdb_id][0]+=int(number)
        for j,term in enumerate(cdl.TERMS):
            obs=pd.to_numeric(data[term],errors='coerce').to_numpy(dtype=float)
            good=np.isfinite(obs)&np.isfinite(cdl_target[:,j])
            if not good.any():continue
            fixed,k,angle=ref.AMBER[term]
            coeff=.5*k*((np.pi/180)**2 if angle else 1.)
            glob,by_type,local=targets(full,j,g,p,r)
            rm_local=targets(rm,j,g,p,r)[2]
            eqs=(fixed,glob,by_type,local,cdl_target[:,j],rm_local)
            energies={arm:np.zeros(len(data)) for arm in ARMS}
            for arm,eq in zip(ARMS,eqs):
                energies[arm][good]=coeff*(obs[good]-(eq if np.isscalar(eq) else eq[good]))**2
            for s,mask in masks.items():
                selected=good&mask
                if not selected.any():continue
                for arm in ARMS:
                    q=summaries[s][term][arm]
                    q['strain_sum']+=float(energies[arm][selected].sum())
                    q['n_observed']+=int(selected.sum())
            selected=good&masks['REFMAC']
            if selected.any():
                ids,inv=np.unique(data.loc[selected,'pdb_id'].astype(str).to_numpy(),return_inverse=True)
                e_cdl=np.bincount(inv,weights=energies['CCTBX_CDL_2026_7'][selected],minlength=len(ids))
                e_mech=np.bincount(inv,weights=energies['TRAIN_PHI_PSI_MECHLIB'][selected],minlength=len(ids))
                for ix,pdb_id in enumerate(ids):
                    z=by_structure[pdb_id];z[1]+=e_cdl[ix];z[2]+=e_mech[ix]
    # Reconstruct full-cohort ten-term sums from per-term contributions.
    checks={'train_rows_match_prior':counts['train']==prior['counts']['train'],
            'test_rows_match_prior':counts['test']==prior['counts']['test'],
            'test_matched_rows_match_prior':strata_rows['ALL']==prior['counts']['test']}
    for arm in cdl.ARMS:
        total=sum(summaries['ALL'][t][arm]['strain_sum'] for t in cdl.TERMS)
        prior_total=prior['arms']['test'][arm]['strain_sum']
        checks['reproduce_prior_test_'+arm]=abs(total-prior_total)<1e-6
    detail=[]
    totals={}
    for s in SUBSETS:
        totals[s]={arm:sum(summaries[s][t][arm]['strain_sum'] for t in cdl.TERMS) for arm in ARMS}
        for t in cdl.TERMS:
            x=summaries[s][t]
            n=x['CCTBX_CDL_2026_7']['n_observed']
            detail.append({'subset':s,'term':t,'n_observed':n,
                           'amber_strain_sum':x['AMBER_FIXED']['strain_sum'],
                           'cdl_strain_sum':x['CCTBX_CDL_2026_7']['strain_sum'],
                           'mechlib_strain_sum':x['TRAIN_PHI_PSI_MECHLIB']['strain_sum'],
                           'cdl_minus_mechlib_strain_per_residue_row':
                           ((x['CCTBX_CDL_2026_7']['strain_sum']-x['TRAIN_PHI_PSI_MECHLIB']['strain_sum'])/strata_rows[s]
                            if strata_rows[s] else None),
                           'cdl_minus_mechlib_per_observed_term':
                           ((x['CCTBX_CDL_2026_7']['strain_sum']-x['TRAIN_PHI_PSI_MECHLIB']['strain_sum'])/n
                            if n else None)})
    bootstrap=None
    if by_structure:
        v=np.array(list(by_structure.values()))
        rng=np.random.default_rng(20260925)
        idx=rng.integers(0,len(v),size=(2000,len(v)))
        draws=((v[idx,1]-v[idx,2]).sum(axis=1)/v[idx,0].sum(axis=1))
        bootstrap={'n_refmac_structures':len(v),
                   'cdl_minus_mechlib_mean':float((v[:,1]-v[:,2]).sum()/v[:,0].sum()),
                   'cluster_bootstrap_ci95':np.quantile(draws,[.025,.975]).tolist(),
                   'bootstrap_replicates':2000,'seed':20260925}
    status='PASS' if all(checks.values()) else 'REPRODUCTION_MISMATCH'
    report={'status':status,'scope':'post_hoc_ten_term_crystal_coordinate_refinement_sensitivity',
            'checks':checks,'software_counts_from_dates':by_era,'test_rows_by_subset':strata_rows,
            'training_rows_all':full['nrows'],'training_rows_refmac':rm['nrows'],
            'test_strain_sums_by_subset':totals,'term_results':detail,
            'refmac_only_cluster_bootstrap':bootstrap,
            'prior_cdl_report_sha256':ref.sha(args.cdl_report),
            'refinement_label_limitation':'REMARK 3 PROGRAM is a proxy and does not establish which CDL restraints were active.',
            'interpretation':'Full training with REFMAC test and a separate REFMAC-trained MechLib arm are both reported. Not a fresh registered endpoint.'}
    args.out.mkdir(parents=True)
    (args.out/'CDL_DIAGNOSTIC_SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CDL_DIAGNOSTIC='+status,'refmac_minus_mechlib=',
          None if bootstrap is None else bootstrap['cdl_minus_mechlib_mean'],flush=True)
    if status!='PASS':raise SystemExit(2)


if __name__=='__main__':main()
