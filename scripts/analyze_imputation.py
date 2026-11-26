"""Outcome-inclusive chained-equation sensitivity for associations only."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
from data_loader import build_B,PRED,ROOT
B=build_B();cols=PRED+['outcome_imt11'];X=B[cols].to_numpy();q=[];u=[]
for i in range(30):
    Z=IterativeImputer(max_iter=10,sample_posterior=True,random_state=200+i).fit_transform(X)
    fit=sm.GLM(Z[:,-1],sm.add_constant(Z[:,:-1]),family=sm.families.Binomial()).fit(maxiter=200,tol=1e-10)
    assert fit.converged
    q.append(fit.params);u.append(fit.bse**2)
out={'method':'IterativeImputer BayesianRidge with posterior sampling, 10 iterations, outcome included; Rubin pooling with t-based inference. Only association estimates are imputed; the endpoint comparison remains complete-case.','seed_scheme':'200+i','n':len(B),'results':{}}
for m in [10,20,30]:
    qq=np.array(q[:m]);uu=np.array(u[:m]);qb=qq.mean(0);ub=uu.mean(0);bb=qq.var(0,ddof=1);tv=ub+(1+1/m)*bb
    df=np.clip((m-1)*(1+ub/((1+1/m)*np.maximum(bb,1e-30)))**2,1,1e6)
    t=stats.t.ppf(.975,df)
    out['results'][str(m)]={k:{'or':float(np.exp(qb[j])),'lo':float(np.exp(qb[j]-t[j]*np.sqrt(tv[j]))),'hi':float(np.exp(qb[j]+t[j]*np.sqrt(tv[j]))),'p':float(2*stats.t.sf(abs(qb[j]/np.sqrt(tv[j])),df[j])),'df':float(df[j])} for j,k in enumerate(['const']+PRED)}
print(json.dumps({m:x['smoke_ever'] for m,x in out['results'].items()},indent=2))
(ROOT/'results/imputation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
