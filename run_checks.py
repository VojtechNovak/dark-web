#!/usr/bin/env python3
"""Focused follow-up studies. No VQC retraining, HTML access or network needed."""
from pathlib import Path
import argparse,json,time,hashlib,platform,sys
import numpy as np,pandas as pd,scipy,sklearn
from scipy import sparse
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from numpy_autoencoder import Autoencoder,gradient_check
from project_features import graph_features,pair_features,pagerank
from compact_vqc import Circuit

DIR_GROUPS=[7,10]
def save_npz(path,**arrays):
    path=Path(path);temporary=path.with_suffix('.writing.npz')
    np.savez_compressed(temporary,**arrays);temporary.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def metric(y,p):return {'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p))}
def stack(p):return np.vstack([p['train_pos'],p['train_neg']]),np.r_[np.ones(len(p['train_pos'])),np.zeros(len(p['train_neg']))]
def tests(p):return np.vstack([p['test_pos'],p['test_neg']]),np.r_[np.ones(len(p['test_pos'])),np.zeros(len(p['test_neg']))]

def integrity(p,full,n,allow_duplicates=False):
    ss={k:set(map(tuple,a.tolist())) for k,a in p.items()}
    assert ss['train_pos']|ss['test_pos']==full
    assert not ss['train_pos']&ss['test_pos']
    assert not any((v,u) in ss['train_pos'] for u,v in ss['test_pos'])
    assert not (ss['train_neg']|ss['test_neg'])&full
    assert not ss['train_neg']&ss['test_neg']
    assert all(np.all((a>=0)&(a<n)) and np.all(a[:,0]!=a[:,1]) for a in p.values())
    if not allow_duplicates:assert all(len(ss[k])==len(a) for k,a in p.items())
    train,y=stack(p);test,yt=tests(p)
    # Explicit derivative of BCE with respect to the full pair-logit matrix.
    rng=np.random.default_rng(901);L=rng.normal(size=(n,n));G=np.zeros((n,n))
    np.add.at(G,tuple(train.T),(expit(L[tuple(train.T)])-y)/len(y))
    assert np.max(np.abs(G[tuple(test.T)]))==0
    old=np.mean(np.logaddexp(0,L[tuple(train.T)])-y*L[tuple(train.T)])
    L[tuple(test.T)]+=rng.normal(size=len(test))*5
    new=np.mean(np.logaddexp(0,L[tuple(train.T)])-y*L[tuple(train.T)])
    assert old==new
    return {'n_train_pairs':len(train),'n_unique_train_pairs':len(set(map(tuple,train))),
            'n_test_pairs':len(test),'direct_test_logit_gradient_max':float(np.max(np.abs(G[tuple(test.T)]))),
            'loss_change_after_heldout_logit_perturbation':float(new-old),
            'train_test_pair_overlap':len(set(map(tuple,train))&set(map(tuple,test))),
            'reverse_positive_overlap':0,'negative_observed_edge_overlap':0}

def operators(n,edges):
    A=sparse.csr_matrix((np.ones(len(edges)),tuple(edges.T)),shape=(n,n))
    Ao=sparse.diags(1/np.maximum(np.asarray(A.sum(1)).ravel(),1))@A
    Ai=sparse.diags(1/np.maximum(np.asarray(A.sum(0)).ravel(),1))@A.T
    return A,Ao.tocsr(),Ai.tocsr()

def node_inputs(A,content,view):
    din=np.asarray(A.sum(0)).ravel();dout=np.asarray(A.sum(1)).ravel()
    S=StandardScaler().fit_transform(np.column_stack([din,dout,np.log1p(din),np.log1p(dout),pagerank(A),np.ones(A.shape[0])]))
    C=StandardScaler().fit_transform(content)
    return C if view=='content' else S if view=='structure' else np.hstack([S,C])

def ranking(S,pos,full_matrix,groups):
    rows=[];n=len(S)
    for u,v in pos:
        keep=~full_matrix[u].copy();keep[u]=False;keep[v]=True
        a=S[u,keep];t=S[u,v]
        rank=1+np.sum(a>t)+.5*max(0,np.isclose(a,t,rtol=1e-12,atol=1e-12).sum()-1)
        rows.append({'source':int(u),'target':int(v),'rank':float(rank),'candidate_n':int(keep.sum()),
                     'source_group':int(groups[u]),'directory_source':bool(groups[u] in DIR_GROUPS)})
    return rows

def summarize_ranks(rows):
    d=pd.DataFrame(rows);out=[]
    for name,m in [('all',np.ones(len(d),bool)),('directory',d.directory_source),('other',~d.directory_source)]:
        q=d[m];r=q['rank'].to_numpy()
        if not len(q):continue
        macro=q.assign(rr=1/q['rank']).groupby('source').rr.mean().mean()
        out.append({'stratum':name,'mrr':float(np.mean(1/r)),'source_macro_mrr':float(macro),
                    'hits1':float(np.mean(r<=1)),'hits3':float(np.mean(r<=3)),
                    'hits10':float(np.mean(r<=10)),'median_rank':float(np.median(r)),
                    'n_queries':len(r),'n_sources':q.source.nunique()})
    return out

def stratify(y,score,te,groups):
    out=[];directory=np.isin(groups[te[:,0]],DIR_GROUPS)
    for name,m in [('all',np.ones(len(y),bool)),('directory',directory),('other',~directory)]:
        if len(np.unique(y[m]))<2:continue
        out.append({'stratum':name,**metric(y[m],score[m]),'n_pairs':int(m.sum()),
                    'n_positive':int(y[m].sum()),'n_sources':len(np.unique(te[m,0]))})
    return out

VARIANTS={
 'gae_full':(False,True,'both',48,.001),
 'vgae_full':(True,True,'both',48,.001),
 'gae_structure_only':(False,True,'structure',48,.001),
 'gae_content_only':(False,True,'content',48,.001),
 'gae_no_messages':(False,False,'both',48,.001),
 'gae_no_messages_capacity':(False,False,'both',143,.001),
 'vgae_no_messages_capacity':(True,False,'both',143,.001),
 'vgae_no_kl':(True,True,'both',48,0.),
}

def strict_replacement(p,A,full_matrix,seed):
    rng=np.random.default_rng(seed+170000);din=np.asarray(A.sum(0)).ravel();n=A.shape[0]
    reserved=set(map(tuple,p['test_neg']));used=set(reserved);out=np.empty_like(p['train_pos'])
    for i in sorted(range(len(out)),key=lambda i:din[p['train_pos'][i,1]],reverse=True):
        u,v=p['train_pos'][i]
        base=[w for w in range(n) if w!=u and not full_matrix[u,w] and (u,w) not in reserved]
        unique=[w for w in base if (u,w) not in used];pool=unique if unique else base
        assert pool,'No source-preserving nonedge after reserving tests'
        delta=np.abs(din[pool]-din[v]);pool=np.asarray(pool)[delta==delta.min()]
        w=int(rng.choice(pool));out[i]=[u,w];used.add((u,w))
    return out

def fit_ae(seed,variant,policy,p,training,A,Ao,Ai,raw,args,out):
    var,msg,view,h,kl=VARIANTS[variant]
    X=node_inputs(A,raw['content_svd32'],view);tr,y=training;te,yt=tests(p)
    modelseed=seed+(1000 if var else 0)
    model=Autoencoder(X.shape[1],h,24,var,msg,modelseed)
    t0=time.perf_counter();z,hist,best=model.train(X,Ao,Ai,tr,y,args.epochs,kl)
    U,V=model.factors(z);S=U@V.T/np.sqrt(24)+model.P['bias']
    key=f'seed{seed}_{variant}_{policy}'
    save_npz(out/'fits'/f'{key}.npz',**{'param_'+k:v for k,v in model.P.items()},
        input_view=np.array(view),bias=model.P['bias'],history=hist,train_pairs=tr,y_train=y,test_pairs=te,y_test=yt,
        test_logits=S[tuple(te.T)],seed=np.array(seed),best_epoch=np.array(best),hidden=np.array(h),
        variational=np.array(var),messages=np.array(msg),kl_weight=np.array(kl))
    row={'seed':seed,'model':variant,'policy':policy,**metric(yt,S[tuple(te.T)]),
         'train_auc':roc_auc_score(y,S[tuple(tr.T)]),'parameters':model.n_parameters(),
         'seconds':time.perf_counter()-t0,'selected_train_epoch':best,'epochs':args.epochs}
    return S,row

def fit_lr(Z,tr,y,seed,queries,n):
    model=LogisticRegression(max_iter=4000,class_weight='balanced',random_state=seed)
    scaler=StandardScaler();model.fit(scaler.fit_transform(pair_features(Z,tr)),y)
    S=np.zeros((n,n))
    for u in queries:
        pairs=np.column_stack([np.full(n,u),np.arange(n)])
        S[u]=model.decision_function(scaler.transform(pair_features(Z,pairs)))
    return S,model,scaler

def classical(args,root,out,raw,full,F):
    pair_rows=[];rank_rows=[];audit_rows=[];sampling_rows=[]
    for seed in args.seeds:
        p=dict(np.load(root/'pairs'/f'pairs_seed{seed}.npz'));audit=integrity(p,full,len(F))
        A,Ao,Ai=operators(len(F),p['train_pos']);Z=np.hstack([graph_features(len(F),p['train_pos'],seed),raw['content_svd32']])
        tr,y=stack(p);te,yt=tests(p);groups=raw['semantic_group'];queries=np.unique(p['test_pos'][:,0])
        changed=p['train_pos'][:,0]!=p['train_neg'][:,0]
        audit_rows.append({'seed':seed,**audit,'fallback_pairs':int(changed.sum())})
        dropped=(np.vstack([p['train_pos'][~changed],p['train_neg'][~changed]]),np.r_[np.ones((~changed).sum()),np.zeros((~changed).sum())])
        replacement=strict_replacement(p,A,F,seed);pr={**p,'train_neg':replacement}
        integrity(pr,full,len(F),allow_duplicates=True)
        policies={'original':(tr,y),'source_replacement':stack(pr),'drop_fallback_pairs':dropped}
        din=np.asarray(A.sum(0)).ravel();dout=np.asarray(A.sum(1)).ravel()
        for policy,(q,labels) in policies.items():
            npos=int(labels.sum());neg=q[labels==0];pos=q[labels==1]
            sampling_rows.append({'seed':seed,'policy':policy,'n_train_positive':npos,'n_train_negative':len(neg),
               'unique_train_negative':len(set(map(tuple,neg))),'train_source_changed':int(np.sum(neg[:,0]!=pos[:,0])),
               'train_test_overlap':len(set(map(tuple,q))&set(map(tuple,te))),
               'target_log_degree_wasserstein':scipy.stats.wasserstein_distance(np.log1p(din[pos[:,1]]),np.log1p(din[neg[:,1]]))})
            # The drop sensitivity retains the original training adjacency and
            # content; only the supervised reconstruction loss pool changes.
            for variant in VARIANTS if policy=='original' else ['gae_full','vgae_full']:
                S,row=fit_ae(seed,variant,policy,p,(q,labels),A,Ao,Ai,raw,args,out)
                for extra in stratify(yt,S[tuple(te.T)],te,groups):pair_rows.append({**row,**extra})
                if policy=='original':
                    rr=ranking(S,p['test_pos'],F,groups)
                    for extra in summarize_ranks(rr):rank_rows.append({'seed':seed,'model':variant,**extra})
                    pd.DataFrame(rr).to_csv(out/'ranks'/f'seed{seed}_{variant}.csv',index=False)
            S,lr,sc=fit_lr(Z,q,labels,seed,queries,len(F))
            for extra in stratify(yt,S[tuple(te.T)],te,groups):pair_rows.append({'seed':seed,'model':'multimodal_logreg','policy':policy,**extra})
            save_npz(out/'fits'/f'seed{seed}_logreg_{policy}.npz',coef=lr.coef_,intercept=lr.intercept_,
                 scaler_mean=sc.mean_,scaler_scale=sc.scale_,feature_view=np.array('multimodal'),train_pairs=q,y_train=labels,test_pairs=te,y_test=yt,test_logits=S[tuple(te.T)])
            if policy=='original':
                for name,Scores in [('multimodal_logreg',S),('preferential_attachment',dout[:,None]*din[None,:])]:
                    rr=ranking(Scores,p['test_pos'],F,groups)
                    for extra in summarize_ranks(rr):rank_rows.append({'seed':seed,'model':name,**extra})
                    pd.DataFrame(rr).to_csv(out/'ranks'/f'seed{seed}_{name}.csv',index=False)
                # All four endpoint degrees test degree channels not directly
                # controlled by the primary source-out/target-in matching.
                deg=np.column_stack([np.log1p(din),np.log1p(dout)])
                Sd,md,sd=fit_lr(deg,q,labels,seed,queries,len(F))
                save_npz(out/'fits'/f'seed{seed}_four_degree_logreg_original.npz',coef=md.coef_,intercept=md.intercept_,
                    scaler_mean=sd.mean_,scaler_scale=sd.scale_,feature_view=np.array('four_degree'),train_pairs=q,y_train=labels,test_pairs=te,y_test=yt,test_logits=Sd[tuple(te.T)])
                for extra in stratify(yt,Sd[tuple(te.T)],te,groups):pair_rows.append({'seed':seed,'model':'four_degree_logreg','policy':policy,**extra})
        pd.DataFrame(pair_rows).to_csv(out/'pair_results_all.csv',index=False)
        pd.DataFrame(rank_rows).to_csv(out/'ranking_all.csv',index=False)
        pd.DataFrame(audit_rows).to_csv(out/'loss_separation_audit.csv',index=False)
        pd.DataFrame(sampling_rows).to_csv(out/'sampling_diagnostics.csv',index=False)
        print(f'Classical seed {seed} completed',flush=True)
    pairdf=pd.DataFrame(pair_rows);rankdf=pd.DataFrame(rank_rows)
    pairdf.groupby(['model','policy','stratum']).agg(auc_mean=('roc_auc','mean'),auc_sd=('roc_auc','std'),
        ap_mean=('average_precision','mean'),ap_sd=('average_precision','std'),n_splits=('seed','size')).to_csv(out/'pair_summary.csv')
    rankdf.groupby(['model','stratum']).agg(mrr_mean=('mrr','mean'),mrr_sd=('mrr','std'),
        macro_mrr_mean=('source_macro_mrr','mean'),macro_mrr_sd=('source_macro_mrr','std'),
        hits10_mean=('hits10','mean'),hits10_sd=('hits10','std'),n_splits=('seed','size')).to_csv(out/'ranking_summary.csv')
    # Descriptive paired differences on the same graph holdouts, no independence
    # or population-significance claims for repeated splits of one crawl.
    base=pairdf.query("stratum=='all' and policy=='original'").set_index(['seed','model'])
    diffs=[]
    for name in ['gae_full','vgae_full','multimodal_logreg']:
        for policy in ['source_replacement','drop_fallback_pairs']:
            alt=pairdf.query("stratum=='all' and model==@name and policy==@policy").set_index('seed')
            delta=alt.roc_auc-base.xs(name,level='model').roc_auc
            diffs.append({'model':name,'policy':policy,'auc_delta_mean':delta.mean(),'auc_delta_sd':delta.std()})
    pd.DataFrame(diffs).to_csv(out/'sampling_paired_differences.csv',index=False)


def quantum(args,root,out):
    df=pd.read_csv(root/'compact_results/results_all.csv');df=df[(df.family=='vqc')&df.graph_seed.isin(args.seeds)]
    c=Circuit();rows=[];geometry=[]
    for seed in args.seeds:
        for start in [0,1]:
            b=dict(np.load(root/'compact_results/blocks'/f'graph{seed}_start{start}.npz'))
            X,y=b['X_train'].astype(float),b['y_train'];T,yt=b['X_test'].astype(float),b['y_test']
            kernels={}
            for enc in ['ry','zz']:
                E=c.encode(X,enc);ET=c.encode(T,enc);K=np.abs(E@E.conj().T)**2
                kernels[enc]=(E,ET)
                Kc=K-K.mean(0)[None,:]-K.mean(1)[:,None]+K.mean();a=2*y-1;L=np.outer(a,a)
                alignment=float(np.sum(Kc*L)/max(np.linalg.norm(Kc)*np.linalg.norm(L),1e-15))
                permrng=np.random.default_rng(810000+seed*10+start)
                perm=np.column_stack([permrng.permutation(a) for _ in range(128)])
                null=np.sum(perm*(Kc@perm),axis=0)/max(np.linalg.norm(Kc)*len(y),1e-15)
                geometry.append({'seed':seed,'start':start,'encoding':enc,'centered_kernel_alignment':alignment,'permuted_alignment_mean':float(null.mean()),
                    'permuted_alignment_sd':float(null.std(ddof=1)),'alignment_minus_permuted_mean':float(alignment-null.mean()),
                    'offdiag_fidelity_mean':float(K[~np.eye(len(K),dtype=bool)].mean()),
                    'same_label_fidelity_mean':float(K[(y[:,None]==y[None,:])&~np.eye(len(K),dtype=bool)].mean()),
                    'different_label_fidelity_mean':float(K[y[:,None]!=y[None,:]].mean())})
            for row in df[(df.graph_seed==seed)&(df.start==start)].itertuples():
                fit=dict(np.load(root/'compact_results/fits'/f'{row.fit_id}.npz'))
                E,ET=kernels[row.model];t=fit['theta'];t0=fit['initial_theta']
                p=c.probabilities(E,t);p0=c.probabilities(E,t0);pt=c.probabilities(ET,t)
                def bce(q):q=np.clip(q,1e-9,1-1e-9);return float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q)))
                def grad(theta):
                    q=np.clip(c.probabilities(E,theta),1e-9,1-1e-9);d=(q-y)/(q*(1-q)*len(y))
                    D=np.column_stack([(c.probabilities(E,theta+np.eye(16)[j]*np.pi/2)-c.probabilities(E,theta-np.eye(16)[j]*np.pi/2))/2 for j in range(16)])
                    return d@D
                gi,gf=grad(t0),grad(t)
                rng=np.random.default_rng(900000+seed*100+start*10+(row.model=='zz')*4+(row.optimizer=='spsa')*2+(row.mode=='shots'))
                aucmc=[roc_auc_score(yt,rng.binomial(1024,pt)/1024) for _ in range(args.mc_repeats)]
                delta=float(pt[yt==1].mean()-pt[yt==0].mean())
                sensitivity=[]
                for j in range(4):
                    plus=T.copy();minus=T.copy();plus[:,j]+=1e-4;minus[:,j]-=1e-4
                    d=(c.probabilities(c.encode(plus,row.model),t)-c.probabilities(c.encode(minus,row.model),t))/2e-4
                    sensitivity.append(float(np.sqrt(np.mean(d*d))))
                rows.append({'fit_id':row.fit_id,'seed':seed,'start':start,'encoding':row.model,'optimizer':row.optimizer,'mode':row.mode,
                    'initial_exact_train_bce':bce(p0),'final_exact_train_bce':bce(p),'initial_train_auc':roc_auc_score(y,p0),
                    'final_train_auc':roc_auc_score(y,p),'exact_test_auc':roc_auc_score(yt,pt),
                    'reported_test_auc':row.roc_auc,'initial_gradient_norm':np.linalg.norm(gi),'final_gradient_norm':np.linalg.norm(gf),
                    'test_positive_negative_probability_gap':delta,'test_probability_sd':np.std(pt),
                    'single_example_shot_se_rms':float(np.sqrt(np.mean(pt*(1-pt)/1024))),
                    'conditional_resampled_test_auc_sd':np.std(aucmc,ddof=1),
                    'input_derivative_rms_mean':np.mean(sensitivity),**{f'input{j}_derivative_rms':v for j,v in enumerate(sensitivity)},
                    'nfev':row.nfev})
        print(f'Quantum diagnostics seed {seed} completed',flush=True)
    d=pd.DataFrame(rows);d.to_csv(out/'quantum_diagnostics_all.csv',index=False)
    keys=['encoding','optimizer','mode'];cols=[k for k in d if k not in keys+['fit_id','seed','start']]
    g=d.groupby(keys+['seed'])[cols].mean();g.groupby(keys).agg(['mean','std']).to_csv(out/'quantum_diagnostics_summary.csv')
    geo=pd.DataFrame(geometry);geo.to_csv(out/'quantum_geometry_all.csv',index=False)
    geo.groupby(['encoding','seed']).mean(numeric_only=True).groupby('encoding').agg(['mean','std']).to_csv(out/'quantum_geometry_summary.csv')
    contrasts=[]
    for enc in ['ry','zz']:
        for opt in ['cobyla','spsa']:
            q=d[(d.encoding==enc)&(d.optimizer==opt)].pivot(index=['seed','start'],columns='mode',values='exact_test_auc')
            delta=(q.shots-q.exact).groupby('seed').mean()
            contrasts.append({'encoding':enc,'optimizer':opt,'shot_training_exact_auc_delta_mean':delta.mean(),
                              'shot_training_exact_auc_delta_sd':delta.std()})
    pd.DataFrame(contrasts).to_csv(out/'quantum_sampling_contrasts.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--supplement',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(42,62)))
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--mc-repeats',type=int,default=64)
    p.add_argument('--only',choices=['all','classical','quantum'],default='all');args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    for d in ['fits','ranks']:(out/d).mkdir(exist_ok=True)
    root=args.supplement;raw=dict(np.load(root/'numeric_data/host_graph_content.npz'))
    n=len(raw['content_svd32']);full=set(map(tuple,raw['full_edges'].tolist()));F=np.zeros((n,n),bool);F[tuple(raw['full_edges'].T)]=True
    assert sum(np.isin(raw['semantic_group'],DIR_GROUPS))==55
    assert np.isin(raw['semantic_group'][raw['full_edges'][:,0]],DIR_GROUPS).sum()==1959
    meta={'seeds':args.seeds,'epochs':args.epochs,'mc_repeats':args.mc_repeats,'implementation':'NumPy float64 analytic-gradient GAE/VGAE; fresh NumPy initialization',
       'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__},
       'gradient_checks':gradient_check(),'directory_group_ids':DIR_GROUPS,'test_based_model_selection':False,
       'optimizer':{'name':'Adam','lr':.01,'betas':[.9,.999],'eps':1e-8,'coupled_weight_decay':1e-4},
       'latent_dim':24,'variants':VARIANTS,'checkpoint':'lowest training objective; posterior mean at evaluation',
       'test_protocol':'fixed published test pairs; average-tie full-graph filtered all-target ranking',
       'sampling':'source replacement may repeat train negatives; drop removes both aligned loss pairs, retains train adjacency',
       'subgroups':'descriptive semantic-source strata, no directory-removal training',
       'quantum':'saved parameters only; exact parameter-shift BCE gradients; fresh conditional binary parity samples; no VQC training',
       'statistics':'starts averaged within quantum split; means/SD across repeated holdouts of one graph; descriptive paired differences',
       'inputs':{'numeric_graph':sha(root/'numeric_data/host_graph_content.npz'),
                 **{f'pairs_seed{s}':sha(root/'pairs'/f'pairs_seed{s}.npz') for s in args.seeds}}}
    t0=time.perf_counter();save_json(out/'configuration.json',meta)
    if args.only in ['all','classical']:classical(args,root,out,raw,full,F)
    if args.only in ['all','quantum']:quantum(args,root,out)
    meta['elapsed_seconds']=time.perf_counter()-t0;save_json(out/'configuration.json',meta)
    print(f'Completed in {meta["elapsed_seconds"]:.1f} seconds',flush=True)

if __name__=='__main__':main()
