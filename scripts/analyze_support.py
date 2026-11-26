"""Supporting analyses and plotting summaries from the original public files."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.special import expit
from scipy.optimize import brentq
from scipy.stats import chi2,norm,binomtest
from patsy import dmatrix,build_design_matrices
from sklearn.metrics import roc_auc_score,cohen_kappa_score
from data_loader import build_A,build_B,PRED,FIG,XLS,ROOT

A,B=build_A(),build_B()
raw=pd.read_csv(FIG,encoding='utf-8-sig')
al=pd.read_excel(XLS,sheet_name='1. Allocation')
imt=pd.read_excel(XLS,sheet_name='2. IMT',na_values=['.'])
M=sm.Logit(A.outcome_CAD,sm.add_constant(A[PRED])).fit(disp=0)
cc=B[PRED].notna().all(axis=1)
pair=cc&B.IMT_seg.notna()
out={'duplicates':int(raw.duplicated().sum()),'parameters':M.params.to_dict(),'coefficient_ci':M.conf_int().to_dict(),'coefficient_p':M.pvalues.to_dict()}
def metrics(lp,y):
    p=expit(lp)
    a=brentq(lambda v: expit(lp+v).mean()-np.mean(y),-40,40)
    slope=sm.Logit(y,sm.add_constant(lp)).fit(disp=0).params[1]
    return [float(roc_auc_score(y,p)),float(np.mean((y-p)**2)),float(a),float(slope)]
both=B.IMT_seg.notna()
yd=(B.loc[both,'maxIMT']>=1.1).astype(int).to_numpy();ys=(B.loc[both,'IMT_seg']>=1.1).astype(int).to_numpy()
cells=[int(((yd==i)&(ys==j)).sum()) for i,j in [(1,1),(1,0),(0,1),(0,0)]]
a,b,c,d=cells
out['agreement']={'n':len(yd),'cells':cells,'agreement':(a+d)/len(yd),'kappa':float(cohen_kappa_score(yd,ys)),'positive_specific':2*a/(2*a+b+c),'negative_specific':2*d/(2*d+b+c),'mcnemar_p':float(binomtest(b,b+c,.5).pvalue),'correlation':float(B.loc[both,['maxIMT','IMT_seg']].corr().iloc[0,1])}
out['baseline']={}
for var in PRED:
    aa=A[var].dropna();bb=B[var].dropna()
    out['baseline'][var]={'A_n':len(aa),'B_n':len(bb),'A_mean':float(aa.mean()),'B_mean':float(bb.mean()),'A_sd':float(aa.std()),'B_sd':float(bb.std()),'A_q':aa.quantile([.25,.5,.75]).tolist(),'B_q':bb.quantile([.25,.5,.75]).tolist(),'B_missing':int(B[var].isna().sum()),'SMD':float((bb.mean()-aa.mean())/np.sqrt((aa.var()+bb.var())/2))}
out['statin_fields']={c:{str(k):int(v) for k,v in al[c].value_counts(dropna=False).items()} for c in al if 'statin' in str(c).lower()}
out['splines']={}
for var in ['HDL','age']:
    X=dmatrix(f'cr({var},df=3)-1',A,return_type='dataframe')
    fit=sm.Logit(A.outcome_CAD,X).fit(disp=0)
    lin=sm.Logit(A.outcome_CAD,sm.add_constant(A[[var]])).fit(disp=0)
    df=int(np.linalg.matrix_rank(X)-np.linalg.matrix_rank(sm.add_constant(A[[var]])))
    lr=float(2*(fit.llf-lin.llf)); pv=float(chi2.sf(lr,df))
    xx=np.linspace(*np.percentile(A[var],[2,98]),120)
    ref=float(A[var].median())
    Xg=np.asarray(build_design_matrices([X.design_info],pd.DataFrame({var:xx}))[0])
    Xref=np.asarray(build_design_matrices([X.design_info],pd.DataFrame({var:[ref]}))[0])
    diff=Xg-Xref; eta=diff@fit.params.to_numpy(); se=np.sqrt(np.einsum('ij,jk,ik->i',diff,fit.cov_params().to_numpy(),diff))
    out['splines'][var]={'x':xx.tolist(),'or':np.exp(eta).tolist(),'lo':np.exp(eta-1.96*se).tolist(),'hi':np.exp(eta+1.96*se).tolist(),'reference':ref,'lr':lr,'df':df,'p':pv,'basis_rank':int(np.linalg.matrix_rank(X))}

models={'Ten variables':M,'Age only':sm.Logit(A.outcome_CAD,sm.add_constant(A[['age']])).fit(disp=0),'Age + smoking':sm.Logit(A.outcome_CAD,sm.add_constant(A[['age','smoke_ever']])).fit(disp=0)}
scores={name: np.asarray(fit.predict(sm.add_constant(B.loc[cc,fit.params.index.drop('const')],has_constant='add'),which='linear')) for name,fit in models.items()}
X=dmatrix('cr(age,df=3)-1',A,return_type='dataframe');fit=sm.Logit(A.outcome_CAD,X).fit(disp=0)
scores['Nonlinear age']=np.asarray(build_design_matrices([X.design_info],B.loc[cc,['age']])[0])@fit.params.to_numpy()
y=(B.loc[cc,'maxIMT']>=1.1).astype(int).to_numpy()
out['comparators404']={k:metrics(v,y) for k,v in scores.items()}
out['comparator_bootstrap']={'n':len(y),'reps':2000,'seed':20261124,'differences':{}}
rng=np.random.default_rng(20261124)
draws={k:[] for k in scores if k!='Ten variables'}
for _ in range(2000):
    idx=rng.integers(0,len(y),len(y)); yy=y[idx]
    if len(np.unique(yy))<2:continue
    pp=expit(scores['Ten variables'][idx]);a1=roc_auc_score(yy,pp);b1=np.mean((yy-pp)**2)
    for k in draws:
        pc=expit(scores[k][idx]);draws[k].append([a1-roc_auc_score(yy,pc),b1-np.mean((yy-pc)**2)])
for k,v in draws.items():out['comparator_bootstrap']['differences'][k]={'delta':(np.array(out['comparators404']['Ten variables'][:2])-out['comparators404'][k][:2]).tolist(),'ci':np.percentile(v,[2.5,97.5],axis=0).tolist(),'usable':len(v)}
bm=sm.Logit(y,sm.add_constant(B.loc[cc,PRED])).fit(disp=0)
z=(M.params-bm.params)/np.sqrt(M.bse**2+bm.bse**2)
ps=2*norm.sf(abs(z[PRED]));adj=multipletests(ps,method='holm')[1]
out['association_comparison']={k:{'A_OR':float(np.exp(M.params[k])),'B_OR':float(np.exp(bm.params[k])),'ratio_OR':float(np.exp(M.params[k]-bm.params[k])),'p':float(ps[i]),'holm_p':float(adj[i])} for i,k in enumerate(PRED)}
original=M.predict(sm.add_constant(B.loc[pair,PRED])).to_numpy()
rev=A.copy();rev['male']=1-rev['male'];rm=sm.Logit(rev.outcome_CAD,sm.add_constant(rev[PRED])).fit(disp=0)
bp=B.loc[pair,PRED].copy();pb=rm.predict(sm.add_constant(bp)).to_numpy();bp['male']=1-bp['male'];pa=rm.predict(sm.add_constant(bp)).to_numpy()
out['sex_sensitivity']={'both_reversed_max_diff':float(abs(pa-original).max()),'A_only_reversed_max_diff':float(abs(pb-original).max()),'original_auc':float(roc_auc_score((B.loc[pair,'maxIMT']>=1.1),original)),'A_only_auc':float(roc_auc_score((B.loc[pair,'maxIMT']>=1.1),pb))}
keep=~raw.duplicated();md=sm.Logit(A.loc[keep,'outcome_CAD'],sm.add_constant(A.loc[keep,PRED])).fit(disp=0)
ld=md.predict(sm.add_constant(B.loc[pair,PRED]),which='linear').to_numpy()
out['duplicate_sensitivity']={'development_n':int(keep.sum()),'evaluation_n':int(pair.sum()),'delta':(np.array(metrics(ld,(B.loc[pair,'maxIMT']>=1.1).astype(int).to_numpy()))-np.array(metrics(ld,(B.loc[pair,'IMT_seg']>=1.1).astype(int).to_numpy()))).tolist()}
flags=original>=.10
out['threshold10']={}
for name,col in [('Designated','maxIMT'),('Segment','IMT_seg')]:
    yy=(B.loc[pair,col]>=1.1).to_numpy();tp=int((flags&yy).sum());fp=int((flags&~yy).sum());fn=int((~flags&yy).sum());tn=int((~flags&~yy).sum())
    out['threshold10'][name]={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'ppv':tp/(tp+fp),'npv':tn/(tn+fn)}
for group,mask in [('Both positive',(B.loc[pair,'maxIMT']>=1.1)&(B.loc[pair,'IMT_seg']>=1.1)),('Designated only',(B.loc[pair,'maxIMT']>=1.1)&(B.loc[pair,'IMT_seg']<1.1)),('Segment only',(B.loc[pair,'maxIMT']<1.1)&(B.loc[pair,'IMT_seg']>=1.1)),('Both negative',(B.loc[pair,'maxIMT']<1.1)&(B.loc[pair,'IMT_seg']<1.1))]:
    out.setdefault('reclassification_profile',{})[group]={'n':int(mask.sum()),'mean_predicted_probability':float(original[mask].mean())}
(ROOT/'results/support.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'spline_tests':{k:{t:v[t] for t in ['df','lr','p']} for k,v in out['splines'].items()},'agreement':out['agreement'],'duplicate_sensitivity':out['duplicate_sensitivity'],'sex_sensitivity':out['sex_sensitivity']},indent=2))
