"""Separate four-vs-eight-component compression from supervised sample size.

RBF SVC is a classical diagnostic. All angles/preprocessing are the existing
train-only exports. No quantum model is trained or selected here.
"""
from pathlib import Path
import argparse,time
import numpy as np,pandas as pd
from sklearn.svm import SVC
from run_checks import save_npz
from sklearn.metrics import roc_auc_score,average_precision_score


def run(root,out,seeds):
    rows=[];baseline=pd.read_csv(root/'compact_results/results_all.csv')
    for seed in seeds:
        d=dict(np.load(root/'datasets'/f'qml_dataset_source_degree_matched_seed{seed}.npz'))
        for start in [0,1]:
            b=dict(np.load(root/'compact_results/blocks'/f'graph{seed}_start{start}.npz'))
            for dim in [4,8]:
                for pool in ['subset256','full_pool']:
                    idx=b['train_indices'] if pool=='subset256' else np.arange(len(d['y_train']))
                    X=d['X_train'][idx,:dim].astype(float);y=d['y_train'][idx];T=d['X_test'][:,:dim].astype(float)
                    t=time.perf_counter();m=SVC(C=1.,kernel='rbf',gamma='scale',class_weight='balanced');m.fit(X,y)
                    s=m.decision_function(T);auc=roc_auc_score(d['y_test'],s)
                    if dim==4 and pool=='subset256':
                        ref=baseline[(baseline.graph_seed==seed)&(baseline.start==start)&(baseline.model=='rbf_svc')].iloc[0]
                        assert abs(auc-ref.roc_auc)<1e-12
                    key=f'seed{seed}_start{start}_svc{dim}_{pool}'
                    save_npz(out/'fits'/f'{key}.npz',support_vectors=m.support_vectors_,dual_coef=m.dual_coef_,
                        intercept=m.intercept_,gamma=np.array(m._gamma),test_inputs=T,y_test=d['y_test'],test_score=s,
                        train_indices=idx,dimension=np.array(dim),seed=np.array(seed),start=np.array(start))
                    rows.append({'seed':seed,'start':start,'dimension':dim,'pool':pool,'train_n':len(X),
                                 'roc_auc':auc,'average_precision':average_precision_score(d['y_test'],s),'seconds':time.perf_counter()-t})
    d=pd.DataFrame(rows);d.to_csv(out/'compression_budget_all.csv',index=False)
    d.groupby(['dimension','pool','seed']).mean(numeric_only=True).groupby(['dimension','pool']).agg(
        auc_mean=('roc_auc','mean'),auc_sd=('roc_auc','std'),ap_mean=('average_precision','mean'),ap_sd=('average_precision','std')
        ).to_csv(out/'compression_budget_summary.csv')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--supplement',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(42,62)));a=p.parse_args();t=time.perf_counter();run(a.supplement,a.out,a.seeds)
    print(f'Compression/sample-size diagnostic: {time.perf_counter()-t:.1f} seconds')
