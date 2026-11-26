"""Refitted published-variable comparison with conventional bootstrap optimism correction."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from data_loader import build_A,FIG,ROOT,PRED
A=build_A();raw=pd.read_csv(FIG,encoding='utf-8-sig');A=A.loc[~raw.duplicated()].reset_index(drop=True)
y=A.outcome_CAD.to_numpy();reps=1000;seed=20261012
columns={'Published five-variable specification':['age','NAFLD','smoke_ever','HDL','LDL'],'Ten-variable specification':PRED}
def fit_predict(X,y,Xtest):
    fit=sm.GLM(y,X,family=sm.families.Binomial()).fit(maxiter=200,tol=1e-10)
    assert fit.converged
    return fit.predict(X),fit.predict(Xtest)
out={'n':len(y),'events':int(y.sum()),'reps':reps,'seed':seed,'method':'Bootstrap refit evaluated on bootstrap training sample and original sample; mean train-minus-original AUC is subtracted from original apparent AUC.','models':{}}
pred={}
for name,cols in columns.items():
    X=sm.add_constant(A[cols]).to_numpy();original,_=fit_predict(X,y,X);pred[name]=original
    apparent=roc_auc_score(y,original);opt=[];rng=np.random.default_rng(seed)
    for _ in range(reps):
        idx=rng.integers(0,len(y),len(y));yy=y[idx]
        if len(np.unique(yy))<2:continue
        pp,po=fit_predict(X[idx],yy,X)
        opt.append(roc_auc_score(yy,pp)-roc_auc_score(y,po))
    out['models'][name]={'apparent_auc':float(apparent),'optimism':float(np.mean(opt)),'optimism_corrected_auc':float(apparent-np.mean(opt)),'usable':len(opt)}
    print(name,out['models'][name],flush=True)
rng=np.random.default_rng(seed);delta=[]
for _ in range(1000):
    idx=rng.integers(0,len(y),len(y))
    delta.append(roc_auc_score(y[idx],pred['Ten-variable specification'][idx])-roc_auc_score(y[idx],pred['Published five-variable specification'][idx]))
out['apparent_auc_difference']={'point':float(roc_auc_score(y,pred['Ten-variable specification'])-roc_auc_score(y,pred['Published five-variable specification'])),'ci':np.percentile(delta,[2.5,97.5]).tolist()}
coef={'age':.093,'NAFLD':.457,'smoke_ever':.900,'HDL':-.792,'LDL':.316}
out['published_coefficient_rank_auc']=float(roc_auc_score(y,sum(A[k]*v for k,v in coef.items())))
(ROOT/'results/nomogram.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
