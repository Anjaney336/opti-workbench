"""Matched empirical CVaR benchmark: SciPy LP versus CVXPY, no cardinality claim."""
from pathlib import Path
import csv,json,time
import numpy as np
from scipy.optimize import linprog
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def cvar(returns,weights,alpha=.95):
    losses=-np.asarray(returns)@np.asarray(weights);eta=np.quantile(losses,alpha)
    return float(eta+np.maximum(losses-eta,0).mean()/(1-alpha))


def solve(returns,method='LP',alpha=.95,cap=.3):
    returns=np.asarray(returns,float)
    if returns.ndim!=2 or not np.isfinite(returns).all() or not 0<alpha<1 or not 0<cap<=1 or cap*returns.shape[1]<1:raise ValueError('invalid or infeasible problem')
    n,m=returns.shape
    if method=='LP':
        objective=np.r_[np.zeros(m),1.,np.ones(n)/(n*(1-alpha))]
        inequalities=np.c_[-returns,-np.ones(n),-np.eye(n)]
        result=linprog(objective,A_ub=inequalities,b_ub=np.zeros(n),A_eq=np.array([np.r_[np.ones(m),np.zeros(n+1)]]),b_eq=[1.],bounds=[(0,cap)]*m+[(None,None)]+[(0,None)]*n,method='highs')
        if not result.success:raise RuntimeError(result.message)
        w=result.x[:m]
    elif method=='CVXPY':
        wv=cp.Variable(m);eta=cp.Variable();z=cp.Variable(n)
        problem=cp.Problem(cp.Minimize(eta+cp.sum(z)/(n*(1-alpha))),[cp.sum(wv)==1,wv>=0,wv<=cap,z>=0,z>=-returns@wv-eta]);problem.solve(solver='CLARABEL')
        if problem.status not in ['optimal','optimal_inaccurate']:raise RuntimeError(problem.status)
        w=np.asarray(wv.value)
    else:raise ValueError('unknown solver')
    if abs(w.sum()-1)>1e-6 or w.min()<-1e-6 or w.max()>cap+1e-6:raise AssertionError('solver infeasible')
    return w


def main():
    out=Path(__file__).parent/'results/matched';out.mkdir(parents=True,exist_ok=True);rows=[];gaps=[]
    for seed in [41,42,43]:
        rng=np.random.default_rng(seed);history=rng.normal(.001,.02,(140,10))+rng.normal(0,.01,(140,1))
        for method in ['LP','CVXPY','EQUAL']:
            held=np.zeros(10);wealth=1.
            for start in range(60,140,20):
                window=history[start-60:start];begin=time.perf_counter();w=np.ones(10)/10 if method=='EQUAL' else solve(window,method);seconds=time.perf_counter()-begin
                gaps.append({'seed':seed,'fold':start,'method':method,'training_cvar':cvar(window,w),'seconds':seconds})
                for t in range(start,start+20):
                    target=w if t==start else held;turnover=np.abs(target-held).sum();gross=float(target@history[t]);net=(1-.001*turnover)*(1+gross)-1;held=target*(1+history[t])/(1+gross)
                    if t==139:net=(1+net)*(1-.001*held.sum())-1
                    wealth*=1+net;rows.append({'seed':seed,'method':method,'period':t,'net_return':net,'wealth':wealth})
    comparisons=[]
    for seed in [41,42,43]:
        for start in range(60,140,20):
            cell={r['method']:r['training_cvar'] for r in gaps if r['seed']==seed and r['fold']==start};comparisons.append(abs(cell['LP']-cell['CVXPY']))
    assert max(comparisons)<1e-6
    for name,records in [('ledger',rows),('solver_audit',gaps)]:
        with (out/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    report={'max_matched_objective_gap':max(comparisons),'seeds':[41,42,43],'folds_per_seed':4,'heldout_periods_per_seed':80,'returns':{str(s):{m:next(r['wealth']-1 for r in reversed(rows) if r['seed']==s and r['method']==m) for m in ['LP','CVXPY','EQUAL']} for s in [41,42,43]},'source':'Seeded synthetic correlated returns','source_url':'https://github.com/Anjaney336/opti-workbench/blob/main/matched_benchmark.py','access_date':'2026-10-24','assumptions':{'alpha':.95,'cap':.3,'turnover_cost':.001,'initial_and_terminal_costs':True},'limitations':'All methods use the same 10-asset long-only feasible set; equal weight is not a CVaR optimizer. No cardinality or MILP. Synthetic data are not calibrated. Solver agreement is numerical verification, not evidence of investment superiority.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2));fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,seed in zip(axes,[41,42,43]):
        for m in ['LP','CVXPY','EQUAL']:
            rs=[r for r in rows if r['seed']==seed and r['method']==m];ax.plot([r['period'] for r in rs],[r['wealth'] for r in rs],label=m)
        ax.set(title=f'Synthetic seed {seed}',xlabel='Held-out period',ylabel='Wealth');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'matched.svg');plt.close(fig);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
