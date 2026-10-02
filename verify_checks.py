"""Independently verify stored follow-up predictions, ranks and loss separation."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from numpy_autoencoder import Autoencoder,gradient_check
from run_checks import operators,ranking,summarize_ranks,integrity,save_json,node_inputs
from compact_vqc import Circuit
from project_features import graph_features


def verify(root,out):
    raw=dict(np.load(root/'numeric_data/host_graph_content.npz'));n=len(raw['content_svd32'])
    full=set(map(tuple,raw['full_edges'].tolist()));F=np.zeros((n,n),bool);F[tuple(raw['full_edges'].T)]=True
    metrics=pd.read_csv(out/'pair_results_all.csv').query("stratum=='all'");ranks=pd.read_csv(out/'ranking_all.csv')
    config=json.loads((out/'configuration.json').read_text());maxerror=0.;count=0;svc_count=0
    for seed in config['seeds']:
        pairs=dict(np.load(root/'pairs'/f'pairs_seed{seed}.npz'));integrity(pairs,full,n)
        A,Ao,Ai=operators(n,pairs['train_pos'])
        for f in sorted((out/'fits').glob(f'seed{seed}_*.npz')):
            d=dict(np.load(f))
            if 'support_vectors' in d:
                svc_count+=1
                from scipy.spatial.distance import cdist
                K=np.exp(-float(d['gamma'])*cdist(d['test_inputs'],d['support_vectors'],'sqeuclidean'))
                score=K@d['dual_coef'].ravel()+d['intercept'][0]
                assert np.max(np.abs(score-d['test_score']))<1e-10
                continue
            q=d['train_pairs'];test=d['test_pairs'];count+=1
            trainset=set(map(tuple,q));testset=set(map(tuple,test))
            assert not trainset&testset
            neg=set(map(tuple,q[d['y_train']==0]));assert not neg&full
            if 'param_src' in d:
                X=node_inputs(A,raw['content_svd32'],str(d['input_view']))
                m=Autoencoder(X.shape[1],int(d['hidden']),24,bool(d['variational']),bool(d['messages']),seed)
                m.P={k[6:]:v for k,v in d.items() if k.startswith('param_')}
                z,_=m.forward(X,Ao,Ai);U,V=m.factors(z)
                score=U@V.T/np.sqrt(24)+m.P['bias']
                maxerror=max(maxerror,float(np.max(np.abs(score[tuple(test.T)]-d['test_logits']))))
                model=f.stem[len(f'seed{seed}_'):]
                policy=next(p for p in ['source_replacement','drop_fallback_pairs','original'] if model.endswith('_'+p))
                model=model[:-(len(policy)+1)]
                # Full pair-logit derivative has support only at explicit train pairs.
                from scipy.special import expit
                G=np.zeros((n,n));np.add.at(G,tuple(q.T),(expit(score[tuple(q.T)])-d['y_train'])/len(q))
                assert np.max(np.abs(G[tuple(test.T)]))==0
                if policy=='original':
                    rr=ranking(score,pairs['test_pos'],F,raw['semantic_group'])
                    for v in summarize_ranks(rr):
                        saved=ranks[(ranks.seed==seed)&(ranks.model==model)&(ranks.stratum==v['stratum'])].iloc[0]
                        for key in ['mrr','source_macro_mrr','hits1','hits3','hits10','median_rank']:
                            assert abs(saved[key]-v[key])<1e-12
            else:
                Z=(np.column_stack([np.log1p(np.asarray(A.sum(0)).ravel()),np.log1p(np.asarray(A.sum(1)).ravel())])
                   if str(d['feature_view'])=='four_degree' else np.hstack([graph_features(n,pairs['train_pos'],seed),raw['content_svd32']]))
                u,v=test.T;a=Z[u];b=Z[v];X=np.hstack([a,b,a*b,np.abs(a-b)])
                logits=((X-d['scaler_mean'])/d['scaler_scale'])@d['coef'].ravel()+d['intercept'][0]
                maxerror=max(maxerror,float(np.max(np.abs(logits-d['test_logits']))))
                model='four_degree_logreg' if 'four_degree' in f.stem else 'multimodal_logreg';policy=f.stem.split('_logreg_')[1]
            row=metrics[(metrics.seed==seed)&(metrics.model==model)&(metrics.policy==policy)].iloc[0]
            assert abs(roc_auc_score(d['y_test'],d['test_logits'])-row.roc_auc)<1e-12
            assert abs(average_precision_score(d['y_test'],d['test_logits'])-row.average_precision)<1e-12
        # Independent verification of degree ranking against original archived results.
        din=np.asarray(A.sum(0)).ravel();dout=np.asarray(A.sum(1)).ravel();S=dout[:,None]*din[None,:]
        rr=ranking(S,pairs['test_pos'],F,raw['semantic_group'])
        row=ranks[(ranks.seed==seed)&(ranks.model=='preferential_attachment')&(ranks.stratum=='all')].iloc[0]
        assert abs(np.mean([1/v['rank'] for v in rr])-row.mrr)<1e-12
    assert maxerror<1e-10
    # Check parameter-shift derivative against central finite differences on a
    # stored trained circuit, rather than reusing the gradient implementation.
    c=Circuit();fit=dict(np.load(root/'compact_results/fits'/'g42_s0_vqc_zz_cobyla_exact.npz'))
    b=dict(np.load(root/'compact_results/blocks'/'graph42_start0.npz'));E=c.encode(b['X_train'][:32],'zz');theta=fit['theta'];err=0.
    for j in range(16):
        e=np.eye(16)[j];ps=(c.probabilities(E,theta+e*np.pi/2)-c.probabilities(E,theta-e*np.pi/2))/2
        fd=(c.probabilities(E,theta+e*1e-5)-c.probabilities(E,theta-e*1e-5))/2e-5
        err=max(err,float(np.max(np.abs(ps-fd))))
    assert err<1e-8
    # Every additional quantum exact AUC is reproducible from retained weights.
    qd=pd.read_csv(out/'quantum_diagnostics_all.csv')
    for row in qd.itertuples():
        f=dict(np.load(root/'compact_results/fits'/f'{row.fit_id}.npz'))
        b=dict(np.load(root/'compact_results/blocks'/f'graph{row.seed}_start{row.start}.npz'))
        p=c.probabilities(c.encode(b['X_test'],row.encoding),f['theta'])
        assert abs(roc_auc_score(b['y_test'],p)-row.exact_test_auc)<1e-12
    # Recalculate descriptive graph-level summaries from individual records.
    pair=pd.read_csv(out/'pair_results_all.csv');sm=pd.read_csv(out/'pair_summary.csv')
    calc=pair.groupby(['model','policy','stratum']).roc_auc.agg(['mean','std'])
    for row in sm.itertuples():
        v=calc.loc[(row.model,row.policy,row.stratum)]
        assert abs(v['mean']-row.auc_mean)<1e-12 and abs(v['std']-row.auc_sd)<1e-12
    qr=qd.groupby(['encoding','optimizer','mode','seed']).mean(numeric_only=True).groupby(['encoding','optimizer','mode']).agg(['mean','std'])
    qs=pd.read_csv(out/'quantum_diagnostics_summary.csv',header=[0,1],index_col=[0,1,2])
    assert np.max(np.abs(qr[qs.columns].to_numpy()-qs.to_numpy()))<1e-10
    return {'saved_classical_fits_checked':count,'compression_svc_fits_checked':svc_count,'quantum_fits_checked':len(qd),'largest_prediction_error':maxerror,
            'parameter_shift_vs_finite_difference_max_error':err,'analytic_autoencoder_gradient_checks':gradient_check(),
            'direct_heldout_loss_gradient':0,'metric_and_rank_reproduction':True,'summary_reproduction':True,
            'no_quantum_retraining':True}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--supplement',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=verify(a.supplement,a.out);save_json(a.out/'verification.json',result);print(json.dumps(result,indent=2,default=float))
