from pathlib import Path
import json,re,html,shutil,csv
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
r=Path(__file__).parent; w=Path.cwd();s=(r/'manuscript_source.md').read_text();meta=json.loads((r/'reference_metadata.json').read_text())

import unicodedata
def clean(t):return re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',str(t)))).strip()
def year(k):
 m=meta[k]['metadata'];return m.get('published-print',m.get('published',m.get('issued',{}))).get('date-parts',[[int(k[-4:])]])[0][0]
BERMAN=[{'family':f,'given':g} for f,g in [('Berman','H. M.'),('Westbrook','J.'),('Feng','Z.'),('Gilliland','G.'),('Bhat','T. N.'),('Weissig','H.'),('Shindyalov','I. N.'),('Bourne','P. E.')]]
def auth(k):
 a=BERMAN if k=='Berman2000' else meta[k]['metadata'].get('author',[])
 return [v for v in a if 'et al' not in (v.get('family',v.get('name','')))]
def truncated(k):return any('et al' in v.get('family',v.get('name','')) for v in meta[k]['metadata'].get('author',[]))
def fams(k):return [v.get('family',v.get('name','')) for v in auth(k)]
JABBR={'Acta Crystallographica Section D Biological Crystallography':'Acta Cryst. D','Acta Crystallographica Section D Structural Biology':'Acta Cryst. D','Acta Crystallographica Section A Foundations of Crystallography':'Acta Cryst. A','Acta Crystallographica Section A Foundations and Advances':'Acta Cryst. A','Acta Crystallographica Section A':'Acta Cryst. A','Journal of Applied Crystallography':'J. Appl. Cryst.'}
def jname(j):
 for kk,v in JABBR.items():
  if j.startswith(kk):return v
 if j.startswith('Acta Crystallographica Section D'):return 'Acta Cryst. D'
 if j.startswith('Acta Crystallographica Section A'):return 'Acta Cryst. A'
 return j
def intext(k):
 f=fams(k);y=year(k)
 n=f[0] if len(f)==1 else (f'{f[0]} & {f[1]}' if len(f)==2 else f'{f[0]} et al.')
 return n,y
def initials(g):
 if re.match(r'^([A-Z]\.\s?)+$',g.strip()):return g.strip()
 parts=re.findall(r'[^\W\d_]+(?:-[^\W\d_]+)?',g,re.UNICODE)
 out=[]
 for x in parts:
  out.append('-'.join(y[0]+'.' for y in x.split('-')))
 return ' '.join(out)
def fullref(k):
 m=meta[k]['metadata'];a=auth(k)
 names=[(v.get('family',v.get('name',''))+', '+initials(v.get('given',''))).rstrip(', ') for v in a[:10]]
 au=(', '.join(names[:-1])+' & '+names[-1]) if len(names)>1 else names[0]
 if len(a)>10 or truncated(k): au=', '.join(names)+' et al.'
 y=year(k);journal=jname(clean(m.get('container-title',[''])[0]));vol=m.get('volume','');page=m.get('page',m.get('article-number',''))
 title=clean(m['title'][0])
 if k=='Smith2020':title='Psi4 1.4: Open-source software for high-throughput quantum chemistry'
 if k=='ChenLu2026':return f'{au} ({y}). A sampling-matched test of whether local peptide mechanics explains protein backbone bond-angle coupling. Submitted.'
 if not vol: 
  url='https://doi.org/'+m['DOI'] if m.get('DOI') else m.get('URL','')
  return f'{au} ({y}). {title}. {journal}. {url}'.replace('. .','.')
 return f'{au} ({y}). {journal}, {vol}'+(', '+page.replace('-', '–') if page else '')+'.'
def resolve(text):
 used=[]
 # duplicate author-year disambiguation over whole metadata of cited keys
 pat2=r'([.,;:]?)\s?\[([A-Za-z][A-Za-z0-9, ]*)\]'
 def sub(mt):
  punct=mt.group(1);kk=[k.strip() for k in mt.group(2).split(',')]
  if not all(k in meta for k in kk):return mt.group(0)
  for k in kk:
   if k not in used:used.append(k)
  c='; '.join(f'{n}, {y}' for n,y in (intext(k) for k in kk))
  return '\x00'+c+'\x01'+punct
 out=re.sub(pat2,sub,text)
 # merge into preceding (Figure ...) parenthesis when directly adjacent
 out=re.sub(r'\(((?:Figure|Table|Fig)[^()]*)\)\x00([^\x01]*)\x01',lambda m:'('+m.group(1)+'; '+m.group(2)+')',out)
 out=out.replace('\x00',' (').replace('\x01',')')
 out=re.sub(r'  +',' ',out)
 return out,used
resolved,used=resolve(s)
print('references cited',len(used))
assert len(used)>=50
refs=sorted(used,key=lambda k:(unicodedata.normalize('NFKD',fams(k)[0]).lower(),len(fams(k))>2,fams(k)[1:2] if len(fams(k))==2 else [],year(k)))
resolved=resolved.replace('{REFERENCES}','\n\n'.join(fullref(k) for k in refs))
keys=used
(r/'manuscript_iucrj.md').write_text(resolved)
# Current source-owned figures; Figure 2 stays unfilled rather than using an obsolete render.
figs={1:r/'figures/Figure1_controlled_response.png',2:r/'figures/Figure2_library_change.png',3:r/'figures/Figure3_archive_fingerprints.png',4:r/'figures/Figure4_three_rules.png'}
fd=r/'figures';fd.mkdir(exist_ok=True)
for n,p in figs.items():
 if p.resolve()!=(fd/p.name).resolve():shutil.copy2(p,fd/p.name)
def setup(title):
 d=Document();sec=d.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
 sec.top_margin=sec.bottom_margin=Inches(.85);sec.left_margin=sec.right_margin=Inches(1)
 for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
  st=d.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
  st.font.size=Pt({'Title':17,'Subtitle':10,'Heading 1':14,'Heading 2':12,'Heading 3':11.5,'Caption':10.5}.get(name,11.5))
  st.paragraph_format.space_after=Pt(6)
  if name.startswith('Heading'):
   st.font.bold=True;st.paragraph_format.space_before=Pt(12);st.paragraph_format.keep_with_next=True
 # Remove inherited theme fonts and decorative title borders.
 for st in d.styles:
  if st.type==1:
   for el in list(st.element.xpath('.//w:pBdr')):el.getparent().remove(el)
  for rf in st.element.xpath('./w:rPr/w:rFonts'):
   for key in list(rf.attrib):
    if 'theme' in key.lower():del rf.attrib[key]
   for key in ['ascii','hAnsi','eastAsia','cs']:rf.set(qn('w:'+key),'Times New Roman')
 d.styles['Caption'].font.bold=False
 d.styles['Caption'].font.italic=False
 d.styles['Normal'].paragraph_format.line_spacing=1.5
 lnn=OxmlElement('w:lnNumType');lnn.set(qn('w:countBy'),'1');lnn.set(qn('w:restart'),'continuous');sec._sectPr.append(lnn)
 d.styles['Normal'].paragraph_format.widow_control=True
 foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.RIGHT
 rr=foot.add_run('');rr.font.size=Pt(9)
 fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');foot._p.append(fld)
 d.core_properties.title=title;d.core_properties.author='Wei Chen and Xiaohong Lu';d.core_properties.subject='Manuscript submitted to IUCrJ'
 return d
def addtext(d,txt,style=None):
 p=d.add_paragraph(style=style)
 # keep all scientific text literal; no unsupported markdown syntax in final Word
 for chunk in re.split(r'(\*\*.*?\*\*)',txt):
  z=p.add_run(chunk[2:-2]if chunk.startswith('**')else chunk)
  if chunk.startswith('**'):z.bold=True
 if '[E6 RESULT PENDING:' in txt or '[SOURCE RECORD PENDING:'in txt or '[QC RECORD PENDING:'in txt or '[MD SOURCE RECORD PENDING:'in txt or '[PUBLIC DEPOSITION PENDING:'in txt:
  for z in p.runs:z.italic=True
 return p
def maketable(d,lines):
 data=[[x.strip()for x in line.strip().strip('|').split('|')]for line in lines]
 data=[x for x in data if not all(re.fullmatch(r':?-+:?',v)for v in x)]
 t=d.add_table(rows=0,cols=len(data[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 widths=([2.0,4.5]if len(data[0])==2 else [.9,.9,.9,1.0,1.55,1.25]if len(data[0])==6 else [1.1,1.1,1.65,1.05,1.6]if len(data[0])==5 else [1.8,1.05,1.8,1.85]if len(data[0])==4 else [1.15,2.45,2.9])
 for j,width in enumerate(widths):t.columns[j].width=Inches(width)
 for i,row in enumerate(data):
  cells=t.add_row().cells
  for j,txt in enumerate(row):
   cells[j].width=Inches(widths[j]);cells[j].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   p=cells[j].paragraphs[0];p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(3);p.paragraph_format.line_spacing=1.0
   z=p.add_run(txt);z.font.size=Pt(9);z.bold=i==0
   tcPr=cells[j]._tc.get_or_add_tcPr(); borders=OxmlElement('w:tcBorders')
   for edge in ['top','left','bottom','right']:
    e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
   tcPr.append(borders)
   if i==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E9E9E9');tcPr.append(sh)
  if i==0:
   trPr=t.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');trPr.append(repeat)
  # no row splits across pages
  pr=t.rows[-1]._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
 d.add_paragraph()
legend_refs=[False]
def convert(text,name,main=False):
 d=setup(text.splitlines()[0].lstrip('# '));parts=text.split('\n\n');legend=False
 for part in parts:
  part=part.strip()
  if not part:continue
  if part.startswith('|'):
   maketable(d,part.splitlines());continue
  if part.startswith('# '):
   d.add_paragraph(part[2:],'Title');continue
  if part.startswith('## '):
   h=part[3:]
   if h in ['Figure legends','References'] or (not main and (h.startswith('Table S8') or h.startswith('Table S9'))):d.add_page_break()
   d.add_heading(h,level=1);legend=h=='Figure legends';legend_refs[0]=h in('References','SI references')
   if not main and h.startswith('Figure S1'):
    p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run().add_picture(str(r/'figures/FigureS1_refmac_era_and_altloc.png'),width=Inches(6.0))
   continue
  if part.startswith('### '):
   h=part[4:]
   if main and h.startswith('Figure '):
    n=int(h.split()[1]);d.add_page_break()if n>1 else None
    d.add_heading(h,level=2)
    if n in figs:
     p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run().add_picture(str(figs[n]),width=Inches(6.35));p.paragraph_format.keep_with_next=True
    else:addtext(d,'[Figure 2 placeholder: insert the current audited blank-altloc, grouped-inference render from the figure owner. The legend below specifies the required evidence. Do not substitute the earlier ungrouped event-study image.]','Caption')
   else:d.add_heading(h,level=2)
   continue
  p=addtext(d,part,'Caption'if legend else None)
  if legend_refs[0]:
   p.paragraph_format.line_spacing=1.0;p.paragraph_format.left_indent=Inches(.25);p.paragraph_format.first_line_indent=Inches(-.25)
   for z in p.runs:z.font.size=Pt(10)
 d.save(r/name)
convert(resolved,'Chen_Lu_IUCrJ_manuscript.docx',True)
si,siused=resolve((r/'supporting_information_source.md').read_text())
si=si.rstrip()+'\n\n## SI references\n\n'+'\n\n'.join(fullref(k) for k in sorted(siused,key=lambda k:fams(k)[0].lower()))
convert(si,'Chen_Lu_IUCrJ_supporting_information.docx')
print('Created manuscript and supporting information; references',len(keys))
