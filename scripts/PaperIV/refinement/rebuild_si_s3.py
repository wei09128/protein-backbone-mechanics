from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
log=(Path(__file__).resolve().parent / 'logs/paper4_check_refinement_software_confound.log').read_text()
ref=(Path(__file__).resolve().parent / 'logs/paper4_refmac_only_sensitivity.log').read_text()
software=[]
for label in ('REFMAC','PHENIX','CNS/X-PLOR','OTHER','BUSTER','TNT'):
 m=re.search(r'^\s*'+re.escape(label)+r'\s*:\s*n_structures=\s*([0-9,]+).*?reduction=\s*([\d.]+)%\s*95% CI=\[([\d.]+)%,\s*([\d.]+)%\]',log,re.M)
 if not m:raise ValueError(label)
 software.append((label,*map(float,m.groups())))
secondary=[]
for label in ('alphaR','beta','PPII','alphaL','coil'):
 m=re.search(r'^\s*'+re.escape(label)+r'\s*:\s*n_structures=\s*([0-9,]+).*?reduction=\s*([\d.]+)%\s*95% CI=\[([\d.]+)%,\s*([\d.]+)%\]',ref,re.M)
 if not m:raise ValueError(label)
 secondary.append((label,*map(float,m.groups())))
fig,ax=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
for a,rows,title,color in [(ax[0],software,'A  Refinement software','#28659f'),(ax[1],secondary,'B  REFMAC by secondary structure','#228f87')]:
 names=[x[0] for x in rows];val=[x[2] for x in rows];lower=[x[2]-x[3] for x in rows];upper=[x[4]-x[2] for x in rows]
 a.bar(range(len(rows)),val,color=color,alpha=.9,width=.72)
 a.errorbar(range(len(rows)),val,yerr=[lower,upper],fmt='none',ecolor='#17263d',capsize=3,linewidth=1)
 a.set_xticks(range(len(rows)),names,rotation=25 if a==ax[0] else 0,ha='right' if a==ax[0] else 'center')
 a.set_ylim(0,65 if a==ax[0] else 55);a.set_ylabel('Strain reduction (%)');a.set_title(title,loc='left',fontweight='bold')
 a.spines[['top','right']].set_visible(False)
 for i,v in enumerate(val):a.text(i,v+2,f'{v:.1f}',ha='center',fontsize=9)
import sys
fig.savefig(sys.argv[1] if len(sys.argv)>1 else 'figureS3_refinement_k80.png',dpi=250)
plt.close(fig)
print('REFINEMENT_FIG=PASS software',software,'secondary',secondary)
