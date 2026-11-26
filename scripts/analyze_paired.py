"""Rebuild endpoint substitution on observed endpoints and one frozen 682-row score."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from patsy import dmatrix, build_design_matrices

ROOT = Path(__file__).resolve().parents[1]
from data_loader import build_A, build_B, PRED, XLS
A, B, P = build_A(), build_B(), PRED
(ROOT/'derived').mkdir(exist_ok=True)
A.to_csv(ROOT/'derived/harmonized_A.csv',index=False)
B.to_csv(ROOT/'derived/harmonized_B.csv',index=False)
raw = pd.read_excel(XLS, sheet_name='2. IMT', na_values=['.'])
segments = [c for c in raw if c.startswith('0M_') and c.endswith('_max_IMT')]
assert len(segments) == 6
al = pd.read_excel(XLS, sheet_name='1. Allocation')
counts = al[['ID']].merge(raw[['ID']+segments], on='ID', how='left')[segments].notna().sum(axis=1)
B['segment_count'] = counts.to_numpy()
eligible = B[P].notna().all(axis=1) & B['maxIMT'].notna() & B['IMT_seg'].notna()
idx = B.index[eligible]
model = sm.Logit(A['outcome_CAD'].astype(float), sm.add_constant(A[P].astype(float))).fit(disp=0)
lp_all = model.params['const'] + B.loc[idx,P].to_numpy() @ model.params[P].to_numpy()
y_d_all = (B.loc[idx,'maxIMT'].to_numpy() >= 1.1).astype(int)
y_s_all = (B.loc[idx,'IMT_seg'].to_numpy() >= 1.1).astype(int)

def met(lp, y):
    p = expit(lp)
    k = int(y.sum())
    a = brentq(lambda x: expit(lp+x).sum()-k, -40, 40)
    slope = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
    return np.array([roc_auc_score(y,p), np.mean((p-y)**2), a, slope], dtype=float)

def evaluate(lp, yd, ys, reps=0, seed=20261012):
    d,s=met(lp,yd),met(lp,ys)
    result={'n':len(lp),'counts':[int(yd.sum()),int(ys.sum())], 'prevalence':[float(yd.mean()),float(ys.mean())],
            'designated':d.tolist(), 'segment':s.tolist(), 'delta':(d-s).tolist(),
            'cells':[int(((yd==1)&(ys==1)).sum()),int(((yd==1)&(ys==0)).sum()),
                     int(((yd==0)&(ys==1)).sum()),int(((yd==0)&(ys==0)).sum())]}
    if reps:
        rng=np.random.default_rng(seed); draws=[]; parts=[]; absolute=[]
        for _ in range(reps):
            j=rng.integers(0,len(lp),len(lp)); a,b=yd[j],ys[j]
            if len(np.unique(a))<2 or len(np.unique(b))<2: continue
            first, second = met(lp[j],a), met(lp[j],b)
            draws.append(first-second)
            absolute.append([first,second])
            z=a-b; parts.append([float(z.mean()),float(-2*np.mean(expit(lp[j])*z))])
        result['bootstrap']={'requested':reps,'usable':len(draws),'ci':np.percentile(draws,[2.5,97.5],axis=0).tolist(),
                             'absolute_ci':np.percentile(absolute,[2.5,97.5],axis=0).tolist(),
                             'brier_parts_ci':np.percentile(parts,[2.5,97.5],axis=0).tolist()}
    z=yd-ys
    result['brier_parts']=[float(z.mean()),float(-2*np.mean(expit(lp)*z))]
    return result

out={'primary':evaluate(lp_all,y_d_all,y_s_all,2000),
     'flow':{'all':len(B),'predictor_complete':int(B[P].notna().all(axis=1).sum()),
             'eligible':len(idx),'missing_segment_after_predictor_screen':int((B[P].notna().all(axis=1)&B['IMT_seg'].isna()).sum())},
     'model':{'n':len(A),'parameters':model.params.to_dict()}}
out['six_complete']=evaluate(lp_all[B.loc[idx,'segment_count'].to_numpy()==6],
                              y_d_all[B.loc[idx,'segment_count'].to_numpy()==6],
                              y_s_all[B.loc[idx,'segment_count'].to_numpy()==6],1000,20261117)
out['extreme_excluded']=evaluate(lp_all[B.loc[idx,'maxIMT'].to_numpy()<=4],
                                 y_d_all[B.loc[idx,'maxIMT'].to_numpy()<=4],
                                 y_s_all[B.loc[idx,'maxIMT'].to_numpy()<=4],1000,20261118)
known = ~B.loc[idx, 'smoke_unknown'].to_numpy()
out['flow']['smoking_unknown_all'] = int(B['smoke_unknown'].sum())
out['flow']['smoking_unknown_paired'] = int((~known).sum())
out['smoking_known'] = evaluate(lp_all[known], y_d_all[known], y_s_all[known], 1000, 20261125)
out['smoking_unknown_ever'] = evaluate(
    lp_all + (~known) * model.params['smoke_ever'], y_d_all, y_s_all, 1000, 20261126
)
out['thresholds']={}
for th in [.8,1.,1.1,1.2,1.5]:
    out['thresholds'][str(th)]=evaluate(lp_all,(B.loc[idx,'maxIMT'].to_numpy()>=th).astype(int),
                                        (B.loc[idx,'IMT_seg'].to_numpy()>=th).astype(int))

scores={'Age only':None,'Age + smoking':None,'Nonlinear age':None,'Ten variables':lp_all}
age=sm.Logit(A['outcome_CAD'],sm.add_constant(A[['age']])).fit(disp=0)
scores['Age only']=age.params['const']+B.loc[idx,'age'].to_numpy()*age.params['age']
ages=sm.Logit(A['outcome_CAD'],sm.add_constant(A[['age','smoke_ever']])).fit(disp=0)
scores['Age + smoking']=ages.params['const']+B.loc[idx,['age','smoke_ever']].to_numpy() @ ages.params[['age','smoke_ever']].to_numpy()
design=dmatrix('cr(age, df=3) - 1', A,return_type='dataframe')
spline=sm.Logit(A['outcome_CAD'],design).fit(disp=0)
scores['Nonlinear age']=np.asarray(build_design_matrices([design.design_info],B.loc[idx,['age']])[0]) @ spline.params.to_numpy()
out['scores']={k:evaluate(np.asarray(v),y_d_all,y_s_all,1000,20261116+i) for i,(k,v) in enumerate(scores.items())}
out['at_10pct']={'flags':int((expit(lp_all)>=.1).sum())}
(ROOT/'results/paired.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:out[k] for k in ['flow','primary','six_complete','extreme_excluded','at_10pct']},indent=2))
