"""Small train-only synthetic evaluation harness for the existing optimizer."""
from pathlib import Path
import json,csv,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optimizer import PortfolioOptimizer,PortfolioRequest
from dataclasses import replace

class ResearchOptimizer(PortfolioOptimizer):
    def _calibrate_problem_to_target_risk(self,problem):
        # Disable the dashboard fixed-risk target; estimate covariance on training data.
        return replace(problem,cov=np.cov(np.asarray(self.request.historical_returns),rowvar=False))


def train_weights(window,method,seed=17):
    opt=ResearchOptimizer(PortfolioRequest(asset_count=10,backtest_lookback=11,active_assets_target=3,max_allocation=.4,budget=5.,historical_returns=window.tolist(),method=method,seed=seed,exact_cvar_seed=False,exact_discrete_selection=False,monte_carlo_paths=200,return_engine='RAW'))
    raw=opt._raw_solve(method)
    w=np.asarray(raw['weights'],float)
    assert abs(w.sum()-1)<1e-6 and w.min()>-1e-6 and w.max()<=.40001
    return w

def evaluate_step(weights,previous,returns,fee=.001):
    turnover=float(np.abs(weights-previous).sum());gross=float(weights@returns)
    net=(1-fee*turnover)*(1+gross)-1
    drift=weights*(1+returns)/(1+gross)
    return net,drift,turnover

def main():
    rng=np.random.default_rng(20261024);common=rng.normal(0,.015,(36,1));history=rng.normal(.004,.025,(36,10))+common
    out=Path(__file__).parent/'results/audit';out.mkdir(parents=True,exist_ok=True);rows=[]
    # Four six-period held-out blocks, trained on the immediately preceding 12 periods.
    for method in ['QP','PSO','GA','HYBRID','EQUAL']:
        print('Evaluating',method,flush=True)
        previous=np.ones(10)/10;wealth=1.
        for fold,start in enumerate([12,18,24,30]):
            begin=time.perf_counter();w=np.ones(10)/10 if method=='EQUAL' else train_weights(history[start-12:start],method)
            seconds=time.perf_counter()-begin
            # Hold weights with drift inside each block; no free periodic rebalancing.
            for i in range(start,start+6):
                target=w if i==start else previous
                net,previous,turnover=evaluate_step(target,previous,history[i]);wealth*=1+net
                rows.append(dict(method=method,fold=fold,period=i,net_return=net,wealth=wealth,turnover=turnover,solve_seconds=seconds if i==start else 0))
    with (out/'walk_forward.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    report={m:{'periods':24,'net_cumulative_return':next(x['wealth'] for x in reversed(rows) if x['method']==m)-1,'turnover':sum(x['turnover'] for x in rows if x['method']==m),'solve_seconds':sum(x['solve_seconds'] for x in rows if x['method']==m)} for m in ['QP','PSO','GA','HYBRID','EQUAL']}
    payload={'data':'Synthetic correlated 10-asset return panel, not FHFA or market history','source_url':'https://github.com/Anjaney336/opti-workbench/blob/main/research_harness.py','access_date':'2026-10-24','seed':20261024,'train':12,'test':6,'folds':4,'fee_per_traded_notional':.001,'report':report,'limits':'Small uncalibrated synthetic experiment. Dashboard fixed-risk calibration disabled; covariance estimated on the training window. CVaR exact-seed and discrete search disabled for this benchmark; QP uses its surrogate. Equal-weight uses ten assets, whereas optimized portfolios target three; not matched constraints. Solver objectives differ. No statistical superiority or exact-MILP claim. Costs exclude initial deployment and final liquidation; benchmark rebalances at same fold boundaries.'}
    (out/'summary.json').write_text(json.dumps(payload,indent=2))
    fig,ax=plt.subplots(figsize=(9,4))
    for m in report:
        z=[x for x in rows if x['method']==m];ax.plot([x['period'] for x in z],[x['wealth'] for x in z],label=m)
    ax.set(xlabel='Synthetic period',ylabel='Wealth from 1',title='Opti-Workbench: train-only allocation, drift and 10 bp turnover cost');ax.legend(ncol=5);fig.tight_layout();fig.savefig(out/'walk_forward.svg');plt.close(fig)
    print(json.dumps(payload,indent=2))
if __name__=='__main__':main()




