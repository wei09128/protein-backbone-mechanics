"""Three-angle response, fitted on isolated interventions only. Degrees throughout."""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
import numpy as np
from audit_base import geometry,coordinates,angle,require,check_reflections,final_r

TERMS=['mACN','mACO','mOCN']
EDITS={'cacn':{'mACN':1.},'aco':{'mACO':1.},'ocn':{'mOCN':1.},'compensated':{'mACN':1.,'mOCN':-1.}}
REUSED=['stock','cacn_plus','cacn_minus','compensated_plus','compensated_minus']

def identify(key):
 kind,atoms=key
 if kind!='angle':return None
 a,b,c=atoms
 if b[0]!='C':return None
 if {a[0],c[0]}=={'CA','O'} and a[2:]==b[2:]==c[2:]:return 'mACO'
 if a[0]=='N':a,c=c,a
 if c[0]=='N' and a[0] in ('CA','O') and a[2:]==b[2:] and b[3]==c[3] and b[4:]!=c[4:]:
  return 'mACN' if a[0]=='CA' else 'mOCN'
 return None

def audit_targets(stock,other,edit,sign):
 require(stock.keys()==other.keys(),'Initial restraint identity changed')
 changed={t:set() for t in edit}
 for key,vals in stock.items():
  nv=other[key];require(vals[1:]==nv[1:],('Initial coordinates/sigma/weight changed',key))
  delta=float(nv[0]-vals[0])
  if delta:
   t=identify(key);require(t in edit and abs(delta-sign*edit[t])<1e-10,('Unexpected target edit',key,delta))
   changed[t].add(key)
 require(all(changed.values()),('Missing intended term',edit))
 return changed

def triplets(stock,eligible):
 by_c={}
 for key in stock:
  t=identify(key)
  if t and all(not a[1] for a in key[1]):by_c.setdefault(key[1][1],{}).setdefault(t,[]).append(key)
 out=[]
 for c,terms in sorted(by_c.items()):
  if set(TERMS)<=terms.keys() and all(len(terms[t])==1 for t in TERMS):
   keys=[terms[t][0] for t in TERMS]
   acn,aco,ocn=[k[1] for k in keys]
   def named(atoms,name):return next(a for a in atoms if a[0]==name)
   if named(acn,'N')!=named(ocn,'N') or named(acn,'CA')!=named(aco,'CA') or named(aco,'O')!=named(ocn,'O'):continue
   if all(k in eligible[t] for t,k in zip(TERMS,keys)):out.append((c,keys))
 require(len(out)>=30,('Insufficient common blank-altloc carbonyl sites',len(out)))
 return out

def response(plus,minus,h=1.):return (np.asarray(plus)-np.asarray(minus))/(2*h)

def main(a):
 a.out.mkdir(parents=True,exist_ok=False)
 names=['cacn','compensated'] if a.partial else list(EDITS)
 arms=['stock']+[f'{n}_{s}' for n in names for s in ['plus','minus']]
 pref={x:a.root/x/f'{a.pdb}_refine_001' for x in arms}
 geos={x:geometry(p.with_suffix('.geo')) for x,p in pref.items()};changed={}
 for n in names:
  pp=audit_targets(geos['stock'],geos[n+'_plus'],EDITS[n],1)
  mm=audit_targets(geos['stock'],geos[n+'_minus'],EDITS[n],-1)
  require(pp==mm,('Signed coverage mismatch',n));changed[n]=pp
 require(changed['cacn']['mACN']==changed['compensated']['mACN'],'Compensated ACN coverage differs')
 coords={x:coordinates(p.with_suffix('.pdb')) for x,p in pref.items()}
 for x,p in pref.items():
  require(coords[x].keys()==coords['stock'].keys(),('Atom coverage mismatch',x))
  require(re.search(r'PBM discrete CDL calls=[1-9]\d*',(p.parent/'RUN.log').read_text()) is not None,('Missing live lookup trace',x))
 if not a.partial:
  require(a.mtz is not None,'Full audit needs --mtz')
  check_reflections(a.mtz,[p.with_suffix('.mtz') for p in pref.values()],a.pdb)
 # mACO coverage is derived from the initial stock in the historical partial analysis.
 eligible={'mACN':changed['cacn']['mACN'],'mOCN':changed['compensated']['mOCN'],'mACO':set(k for k in geos['stock'] if identify(k)=='mACO')}
 if not a.partial:
  require(changed['ocn']['mOCN']==changed['compensated']['mOCN'],'Isolated/compensated OCN coverage differs')
  eligible['mACO'] &=changed['aco']['mACO'];eligible['mOCN'] &=changed['ocn']['mOCN']
 sites=triplets(geos['stock'],eligible);rows=[];matrices=[];predictions=[];observations=[];even=[]
 for c,keys in sites:
  vals={x:np.array([angle(coords[x],key[1]) for key in keys]) for x in arms}
  vecs={n:response(vals[n+'_plus'],vals[n+'_minus']) for n in names}
  row={'carbonyl_id':repr(c),'stock_angle_sum_deg':float(sum(vals['stock']))}
  for n in names:
   for j,t in enumerate(TERMS):
    row[n+'_response_'+t]=float(vecs[n][j])
    row[n+'_even_'+t]=float((vals[n+'_plus'][j]+vals[n+'_minus'][j])/2-vals['stock'][j])
    even.append(abs(row[n+'_even_'+t]))
  if not a.partial:
   T=np.column_stack([vecs[n] for n in ['cacn','aco','ocn']]);pred=T@np.array([1.,0.,-1.]);obs=vecs['compensated']
   matrices.append(T);predictions.append(pred);observations.append(obs)
   for j,t in enumerate(TERMS):row['heldout_pred_'+t]=float(pred[j]);row['heldout_obs_'+t]=float(obs[j]);row['heldout_error_'+t]=float(obs[j]-pred[j])
  rows.append(row)
 with (a.out/'site_response.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
 report={'status':'PARTIAL_EXISTING_PILOT' if a.partial else 'PASS_MATRIX_INPUT_AUDITS','structure':a.pdb,'source_root':str(a.root.resolve()),'blank_altloc_common_sites':len(sites),'term_order':TERMS,'input_target_amplitude_deg':1.,'response_definition':'(q_plus-q_minus)/2; all target amplitudes 1 degree','protocol_scope':'One-cycle finite protocol response, not converged Hessian response','heldout_arm':'compensated; target vector [1,0,-1] excluded from isolated matrix fit','max_abs_even_response_deg':max(even),'mean_responses':{n:[float(np.mean([r[n+'_response_'+t] for r in rows])) for t in TERMS] for n in names},'R_work_R_free':{x:list(map(float,final_r(p.with_suffix('.log')))) for x,p in pref.items()},'reflection_audit':'not rerun in portable historical partial analysis; original pilot audit remains separate' if a.partial else 'PASS','limitations':['Uniform edits estimate local outputs under GLOBAL target edits, not a single-site causal response','PDB coordinates rounded to 0.001 Angstrom; tiny responses have a serialization floor','Three carbonyl angles need not sum exactly to 360 degrees when nonplanar','No truth geometry, experimental information fraction or calibrated confidence interval inferred']}
 if not a.partial:
  Tmean=np.mean(matrices,axis=0);u,s,vh=np.linalg.svd(Tmean);pred=np.array(predictions);obs=np.array(observations);err=obs-pred
  report.update({'mean_response_matrix':Tmean.tolist(),'svd_of_mean_matrix':{'singular_values':s.tolist(),'input_modes_rows_Vt':vh.tolist(),'output_modes_columns_U':u.tolist()},'heldout_errors':{t:{'mae_deg':float(np.mean(abs(err[:,j]))),'rmse_deg':float(np.sqrt(np.mean(err[:,j]**2))),'mean_observed_deg':float(obs[:,j].mean()),'mean_predicted_deg':float(pred[:,j].mean()),'zero_response_baseline_mae_deg':float(np.mean(abs(obs[:,j])))} for j,t in enumerate(TERMS)},'heldout_mae_all_terms_deg':float(np.mean(abs(err))),'zero_response_baseline_mae_all_terms_deg':float(np.mean(abs(obs))),'model_identity':'Three angle terms in degrees; no bond/angle scaling mixture'})
  # Mechanical feasibility gate fixed before the new arms: compare with zero prediction.
  report['predictive_gate']='PASS' if report['heldout_mae_all_terms_deg']<report['zero_response_baseline_mae_all_terms_deg'] else 'FAIL'
  report['predictive_gate_scope']='Descriptive pilot only; no population inference or calibrated tolerance'
 hashes={str(p.relative_to(a.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.root.rglob('*') if p.is_file() and p.suffix in ('.phil','.geo','.pdb') and p.parent.name in arms or p.is_file() and p.name=='PHENIX_VERSION.txt'}
 report['source_file_sha256']=hashes
 (a.out/'MATRIX_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='source_file_sha256'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pdb',choices=['4aqo','4yca'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mtz',type=Path);p.add_argument('--partial',action='store_true');main(p.parse_args())
