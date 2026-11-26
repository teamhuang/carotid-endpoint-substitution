"""Independent GLM-vs-count-solver check for each replicate of the headline cell."""
import json
import numpy as np
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.special import expit
from simulation_core import ROOT,SEED,draw_outcome,expand,misclass,make_lp
n=2000;reps=2000;base=.646;dpi=.223;auc=.63
z,beta=make_lp(n,auc);zz=np.repeat(z[None,:],reps,axis=0)
b0=brentq(lambda v:expit(v+beta*z).mean()-base,-12,4)
rng=np.random.default_rng(SEED+round(dpi*1000)+round(auc*100)+n)
y1=draw_outcome(zz,b0,beta,rng);target=np.full(reps,round(n*(base+dpi)))
ys={key:expand(y1,zz,target,how,rng) for key,how in [('M1','desc'),('M2','rand'),('M3','asc')]}
ys['M4']=misclass(y1,.9,.9,rng)
shift=np.log((base+dpi)/(1-base-dpi))-np.log(base/(1-base))
ys['M5']=draw_outcome(zz,b0+shift,beta,rng)
count_cache={}
def from_count(k):
    k=int(k)
    if k not in count_cache:count_cache[k]=brentq(lambda a:expit(z+a).sum()-k,-40,40,xtol=1e-13)
    return count_cache[k]
def from_labels(y):return float(sm.GLM(y,np.ones((n,1)),family=sm.families.Binomial(),offset=z).fit(tol=1e-12).params[0])
a1=np.array([from_labels(y) for y in y1]);c1=np.array([from_count(y.sum()) for y in y1])
out={'cell':{'n':n,'base_auc':auc,'target_dpi':dpi},'seed':SEED,'reps_checked':reps,'mechanisms':{}}
for key,Y in ys.items():
    aa=np.array([from_labels(y) for y in Y]);cc=np.array([from_count(y.sum()) for y in Y])
    residual=(a1-aa)-(c1-cc);maximum=float(abs(residual).max())
    assert maximum<1e-8,(key,maximum)
    out['mechanisms'][key]={'max_abs_identity_residual':maximum,'mean_base_prevalence':float(y1.mean()),'mean_alternative_prevalence':float(Y.mean()),'mean_signed_prevalence_change':float(Y.mean()-y1.mean())}
    print(key,maximum,flush=True)
assert out['mechanisms']['M4']['mean_signed_prevalence_change']<0
(ROOT/'results/identity.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
