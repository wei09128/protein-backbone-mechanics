#!/usr/bin/env python3
"""Post hoc NMR sensitivity: discard first 200 ps of each 1 ns production trajectory."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

CASES={'M2-GB3':('M2E_GB3_TableS1_transcription_v1.tsv',50),
       'M2-UBQ':('M2E_UBQ_TableS3_HNHA_transcription_v1.tsv',63)}
ARMS=('AMBER_FIXED','AMBER_MECHLIB_V1')
SEEDS=('2026092201','2026092202','2026092203')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def require(ok,msg):
    if not ok:raise ValueError(msg)

def write_tsv(p,rows,cols):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,delimiter='\t');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--lock',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args()
    require(a.preflight != a.execute,'choose exactly one: --preflight or --execute')
    root=a.root.resolve();out=a.out.resolve();lock=json.loads(a.lock.read_text())
    paths={'input_manifest':root/'M2E_INPUT_SHA256_v1.tsv',
           'contract':root/'M2E_SCORING_CONTRACT_FROZEN_v1.md',
           'scorer':root/'M2E_score_scalar_couplings_v1.py'}
    for key,p in paths.items():
        require(sha(p)==lock[key+'_sha256'],f'lock hash mismatch: {key}: {p}')
    with paths['input_manifest'].open(newline='') as f:
        inventory=list(csv.DictReader(f,delimiter='\t'))
    require(len(inventory)==259,'input inventory must have 259 rows')
    require(len({r['relative_path'] for r in inventory})==259,'duplicate input paths')
    from collections import Counter
    require(Counter(r['kind'] for r in inventory)=={
        'integrity_report':1,'experimental_source_pdf':2,'topology':2,
        'nmr_reference_table':2,'audit':12,'trajectory':120,'thermodynamic_report':120},
        'manifest file counts changed')
    for row in inventory:
        p=(root/row['relative_path']).resolve()
        require(root in p.parents and p.is_file() and p.stat().st_size==int(row['size_bytes']) and
                sha(p)==row['sha256'],f'input integrity mismatch: {p}')

    import mdtraj as md
    import numpy as np
    require(md.__version__=='1.10.3' and np.__version__=='2.2.6',
            f'environment mismatch: mdtraj={md.__version__}, numpy={np.__version__}')
    index={};obs={}
    for case,(filename,expected) in CASES.items():
        pdb=root/'frozen_inputs'/case/'prepared.pdb'
        model=md.load_pdb(str(pdb));phi_idx,_=md.compute_phi(model)
        mapping={}
        for i,quad in enumerate(phi_idx):
            residue=model.topology.atom(int(quad[2])).residue
            if residue.is_protein:
                require(residue.resSeq not in mapping,f'duplicate phi mapping: {case} {residue.resSeq}')
                mapping[residue.resSeq]=i
        tab=root/'M2E_TRAJECTORY_CHECK_v1_0_1'/filename
        with tab.open(newline='') as f:lines=list(csv.DictReader(f,delimiter='\t'))
        primary=[r for r in lines if r['primary_non_gly']=='YES']
        require(len(primary)==expected and len({int(r['residue_number']) for r in primary})==expected,
                f'primary observation count or numbering: {case}')
        for row in primary:
            n=int(row['residue_number']);res=model.topology.atom(int(phi_idx[mapping[n],2])).residue
            require(res.code==row['residue'] and row['residue'] not in ('G','P'),
                    f'experimental row mapping: {case} {n}')
            require(math.isfinite(float(row['J_HN_Halpha_Hz'])),f'nonfinite observation: {case} {n}')
            if case=='M2-UBQ':require(row['method']=='HNHA',f'wrong method: {case} {n}')
        index[case]=(pdb,phi_idx,[mapping[int(r['residue_number'])] for r in primary])
        obs[case]=primary
    if a.preflight:
        print('M2E_LAST800_PREFLIGHT=PASS; 259 file hashes; 113 mapped non-gly NMR rows; no DCD frames scored')
        return
    require(not out.exists(),f'output already exists: {out}')
    out.mkdir(parents=True)
    residue_rows=[];lineage_rows=[];block_rows=[]
    for case in CASES:
        pdb,phi_idx,selection=index[case]
        exp=np.array([float(r['J_HN_Halpha_Hz']) for r in obs[case]])
        for arm in ARMS:
            for seed in SEEDS:
                folder=root/'m2d_pilot_out_v101_GB3_corruption_rerun_v1'/case/arm/('seed_'+seed)
                sums=np.zeros(len(exp));blocks=[]
                for chunk in range(2,10):
                    traj=md.load_dcd(str(folder/f'trajectory_chunk_{chunk:02d}.dcd'),top=str(pdb))
                    require(traj.n_frames==10,f'frame count: {case} {arm} {seed} {chunk}')
                    phi=md.compute_dihedrals(traj,phi_idx)[:,selection]
                    require(np.isfinite(phi).all(),f'phi: {case} {arm} {seed} {chunk}')
                    c=np.cos(phi-np.pi/3)
                    predicted=6.7*c*c-1.3*c+1.5
                    require(np.isfinite(predicted).all(),f'predicted J: {case} {arm} {seed} {chunk}')
                    sums+=predicted.sum(axis=0)
                    blocks.append(predicted.mean(axis=0))
                    block_rows.append({'case':case,'arm':arm,'seed':seed,'chunk':chunk,
                                       'mse_hz2':float(np.mean((blocks[-1]-exp)**2))})
                mean=sums/80
                mse=float(np.mean((mean-exp)**2))
                lineage_rows.append({'case':case,'arm':arm,'seed':seed,'n_observations':len(exp),
                                     'n_frames':80,'mse_hz2':mse,'rmse_hz':math.sqrt(mse)})
                for i,row in enumerate(obs[case]):
                    residue_rows.append({'case':case,'arm':arm,'seed':seed,
                                         'residue_number':row['residue_number'],'residue':row['residue'],
                                         'observed_hz':float(exp[i]),'predicted_hz':float(mean[i]),
                                         'residual_hz':float(mean[i]-exp[i])})
    lookup={(r['case'],r['arm'],r['seed']):r for r in lineage_rows}
    contrasts=[]
    for case in CASES:
        for seed in SEEDS:
            fixed=lookup[case,'AMBER_FIXED',seed];modified=lookup[case,'AMBER_MECHLIB_V1',seed]
            contrasts.append({'case':case,'seed':seed,'amber_mse_hz2':fixed['mse_hz2'],
                              'mechlib_mse_hz2':modified['mse_hz2'],
                              'delta_mse_hz2':fixed['mse_hz2']-modified['mse_hz2']})
    means={case:sum(r['delta_mse_hz2'] for r in contrasts if r['case']==case)/3 for case in CASES}
    pooled=sum(means.values())/2
    if pooled>0 and all(v>0 for v in means.values()):status='PILOT_CONCORDANT_POSITIVE'
    elif pooled<0 and all(v<0 for v in means.values()):status='PILOT_CONCORDANT_NEGATIVE'
    else:status='PILOT_MIXED_OR_ZERO'
    write_tsv(out/'lineages.tsv',lineage_rows,list(lineage_rows[0]))
    write_tsv(out/'residues.tsv',residue_rows,list(residue_rows[0]))
    write_tsv(out/'blocks.tsv',block_rows,list(block_rows[0]))
    write_tsv(out/'paired_contrasts.tsv',contrasts,list(contrasts[0]))
    summary={'status':status,'scope':'post_hoc_last_800_ps_of_1_ns_pilot_not_Gate_M',
             'case_mean_delta_mse_hz2':means,'equal_protein_mean_delta_mse_hz2':pooled,
             'positive_favors':'MechLib','n_proteins':2,'n_seeds_per_protein':3,
             'n_primary_rows_by_protein':{k:len(v) for k,v in obs.items()},
             'lock_sha256':sha(a.lock),'mdtraj_version':md.__version__,
             'no_universal_validation_claim':True,'n_production_frames_discarded':20,'registered_full_trajectory_score_unchanged':True}
    (out/'POSTHOC_LAST800_SCORE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('M2E_POSTHOC_LAST800PS='+status,'equal_protein_mean_delta_mse_hz2='+str(pooled),'output='+str(out))

if __name__=='__main__':main()
