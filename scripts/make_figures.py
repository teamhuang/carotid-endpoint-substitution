"""Render the five manuscript figures and each panel as 600 dpi PNG and vector PDF."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from data_loader import ROOT, build_A, build_B

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
    'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
    'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'savefig.facecolor':'white'})
def data(name):
    return json.loads((ROOT/f'results/{name}.json').read_text(encoding='utf-8'))

P=data('paired')
S=data('support')
sim=[x for x in data('simulation')['results'] if (x['target_dpi'],x['base_auc'],x['n'])==(.223,.63,2000)]
A,B=build_A(),build_B()
blue,orange,ink='#0072B2','#D55E00','#262626'
def save(fig, name, kind):
    folder=ROOT/'figures'/kind
    folder.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(folder/f'{name}.{ext}',dpi=600)
    plt.close(fig)

def flow(ax):
    ax.set_axis_off()
    def box(x,y,label):
        ax.text(x,y,label,ha='center',va='center',transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=.5',fc='white',ec='#717171',lw=.75),linespacing=1.35)
    def arrow(x1,y1,x2,y2):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->',lw=.75,color='#474747'))
    box(.19,.82,'Cohort A\n682 released records')
    box(.19,.48,'Fit logistic score\nFreeze coefficients')
    box(.65,.82,'Cohort B\n463 baseline participants')
    box(.55,.48,'Scorable records\n404')
    box(.84,.48,'Both IMT fields\n453 for agreement')
    box(.55,.14,'Scorable with both fields\n400 for paired evaluation')
    arrow(.19,.70,.19,.60);arrow(.61,.70,.55,.59);arrow(.71,.70,.83,.59)
    arrow(.55,.36,.55,.25);arrow(.32,.46,.43,.46)
    ax.text(.69,.28,'4 had no segment value',ha='left',transform=ax.transAxes)

def agreement(ax):
    a,b,c,d=S['agreement']['cells']
    arr=np.array([[a,b],[c,d]])
    edges=np.arange(3)-.5
    ax.pcolormesh(edges,edges,arr,cmap='Blues',vmin=0,vmax=320)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    for (i,j),v in np.ndenumerate(arr):
        ax.text(j,i,str(v),ha='center',va='center',color='white' if v>150 else ink,fontsize=10)
    ax.set(xticks=[0,1],xticklabels=['Positive','Negative'],yticks=[0,1],
           yticklabels=['Positive','Negative'],xlabel='Segment-derived',ylabel='Designated',
           title='Endpoint agreement')
    ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)

def differences(ax):
    ax.set_axis_off();ax.set_title('Paired metric differences',pad=7)
    z=P['primary'];lo,hi=z['bootstrap']['ci']
    names=['AUC','Brier score','Offset intercept','Recalibration slope']
    limits=[(-.16,.10),(-.02,.09),(-2.1,.4),(-.5,.5)]
    for j,(name,lim) in enumerate(zip(names,limits)):
        yy=.79-j*.24
        ax.text(0,yy+.078,name,transform=ax.transAxes,va='bottom')
        inner=ax.inset_axes([.04,yy,.91,.06])
        v=z['delta'][j]
        inner.axvline(0,color='#929292',lw=.7,ls='--')
        inner.errorbar(v,0,xerr=[[v-lo[j]],[hi[j]-v]],fmt='o',ms=3.5,color=blue,capsize=2,lw=1)
        inner.set(xlim=lim,ylim=(-1,1),yticks=[])
        inner.spines[['left','bottom']].set_visible(False)
        inner.tick_params(axis='x',length=2,pad=1)
        inner.set_xticks(np.linspace(lim[0],lim[1],3),labels=[f'{x:.2f}' for x in np.linspace(*lim,3)])

def count(ax):
    z=P['primary'];x=z['counts'];y=[z['designated'][2],z['segment'][2]]
    ax.plot(x,y,color='#828282',lw=.9,zorder=1)
    ax.scatter(x,y,c=[blue,orange],s=21,zorder=2)
    ax.annotate('Designated',(x[0],y[0]),xytext=(6,3),textcoords='offset points')
    ax.annotate('Segment-derived',(x[1],y[1]),xytext=(-6,-12),ha='right',textcoords='offset points')
    ax.set(xlim=(246,364),ylim=(-.3,1.85),xticks=x,yticks=[0,.5,1,1.5],
           xlabel='Positive labels',ylabel='Offset intercept',title='Calibration and positive count')

def simulation(key,label):
    def plot(ax):
        rows=np.array([x[key] for x in sim])
        y=np.arange(len(rows))[::-1]
        ax.axvline(0,color='#929292',lw=.7,ls='--')
        for j,(m,l,h) in enumerate(rows):
            ax.errorbar(m,y[j],xerr=[[m-l],[h-m]],fmt='o',ms=4,
                        color=orange if j==3 else blue,capsize=2,lw=1)
        ax.set(yticks=y,yticklabels=[f'M{k}' for k in range(1,6)],ylim=(-.65,4.65),xlabel=label)
    return plot

def density(variable,label):
    def plot(ax):
        combined=np.r_[A[variable].dropna(),B[variable].dropna()]
        x=np.linspace(*np.percentile(combined,[.25,99.75]),180)
        for frame,color,name,ls in [(A,blue,'Cohort A','-'),(B,orange,'Cohort B','--')]:
            y=gaussian_kde(frame[variable].dropna())(x)
            ax.plot(x,y,color=color,label=name,ls=ls,lw=1.25)
        ax.set(xlabel=label,ylabel='Density',ylim=(0,None))
        if variable=='age':ax.legend(frameon=False,loc='upper left',fontsize=8)
    return plot

def maxima(ax):
    ax.boxplot(B.maxIMT.dropna(),orientation='horizontal',positions=[0],widths=.45,showfliers=False,
               medianprops={'color':blue},boxprops={'color':ink},whiskerprops={'color':ink})
    v=B.maxIMT[B.maxIMT>4]
    ax.scatter(v,np.linspace(-.12,.12,len(v)),s=8,color=orange,zorder=3)
    ax.axvline(4,color='#888888',ls='--',lw=.7)
    ax.set(xlabel='Designated IMT (mm)',yticks=[],ylim=(-.45,.5),xlim=(0,13.6))
    ax.spines['left'].set_visible(False)

def spline(variable,label):
    def plot(ax):
        v=S['splines'][variable]
        ax.fill_between(v['x'],v['lo'],v['hi'],color=blue,alpha=.16,linewidth=0)
        ax.plot(v['x'],v['or'],color=blue,lw=1.35)
        ax.axhline(1,color='#858585',lw=.7,ls='--')
        ax.axvline(v['reference'],color='#858585',lw=.6,ls=':')
        ax.set(xlabel=label,ylabel='Odds ratio',yscale='log')
    return plot

def build(name,draw,ncols,width,height,single=(3.25,2.5)):
    n=len(draw);nrows=(n+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(width,height),squeeze=False)
    fig.subplots_adjust(left=.12 if ncols<3 else .116,right=.974,bottom=.17 if nrows==1 else .12,
                        top=.81 if nrows==1 and n>1 else .91,wspace=.58 if ncols<3 else .69,hspace=.71)
    for i,(ax,fun) in enumerate(zip(axes.flat,draw)):
        fun(ax)
        if n>1:
            pos=ax.get_position(original=True)
            fig.text(pos.x0-.055,pos.y1+(.087 if nrows==1 else .033),chr(65+i),
                     fontsize=10,fontweight='bold',ha='left',va='bottom')
        panel,axis=plt.subplots(figsize=single)
        panel.subplots_adjust(left=.23,right=.94,bottom=.23,top=.87)
        fun(axis)
        save(panel,f'{name}{chr(65+i) if n>1 else ""}','panels')
    save(fig,name,'combined')

build('Figure_1',[flow],1,6.75,2.65,(6.75,2.65))
build('Figure_2',[agreement,differences,count],3,7.05,3.12,(3.25,2.85))
build('Figure_3',[simulation(k,l) for k,l in [('dAUC','Δ AUC'),('dSlope','Δ slope'),
    ('dBrier','Δ Brier score'),('dCITL','Δ offset intercept')]],2,6.75,4.55)
build('Figure_S1',[density('age','Age (years)'),density('HbA1c','HbA1c (%)'),
    density('HDL','HDL-C (mmol/L)'),maxima],2,6.75,4.55)
build('Figure_S2',[spline('HDL','HDL-C (mmol/L)'),spline('age','Age (years)')],2,6.75,2.55)
