"""Rebuild train-only directed graph/PCA features without NetworkX.

The numerical input is node-content SVD plus full directed edges and explicit
pair splits. Original addresses and HTML are not required by this module.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler


def pagerank(A, alpha=.85, tol=1e-6, max_iter=100):
    n=A.shape[0]; degree=np.asarray(A.sum(axis=1)).ravel()
    P=sparse.diags(1/np.maximum(degree,1))@A
    x=np.full(n,1/n); dangling=degree==0
    for _ in range(max_iter):
        old=x
        x=alpha*(np.asarray(old@P).ravel()+old[dangling].sum()/n)+(1-alpha)/n
        if np.abs(x-old).sum()<n*tol:return x
    raise RuntimeError('PageRank failed to converge')


def graph_features(n, edges, seed, dim=32):
    A=sparse.csr_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(n,n))
    Zo=TruncatedSVD(dim,random_state=seed).fit_transform(A)
    Zi=TruncatedSVD(dim,random_state=seed+1).fit_transform(A.T)
    din=np.asarray(A.sum(axis=0)).ravel();dout=np.asarray(A.sum(axis=1)).ravel()
    D=StandardScaler().fit_transform(np.column_stack([din,dout,np.log1p(din),np.log1p(dout),pagerank(A)]))
    return np.hstack([Zo,Zi,D])


def pair_features(Z, pairs):
    a=Z[pairs[:,0]];b=Z[pairs[:,1]]
    return np.hstack([a,b,a*b,np.abs(a-b)])


def project(content, pairs, seed, qdim=8):
    Z=np.hstack([graph_features(len(content),pairs['train_pos'],seed),content])
    tr=np.vstack([pairs['train_pos'],pairs['train_neg']]);te=np.vstack([pairs['test_pos'],pairs['test_neg']])
    scaler=StandardScaler();Xs=scaler.fit_transform(pair_features(Z,tr));Xt=scaler.transform(pair_features(Z,te))
    pca=PCA(n_components=qdim,random_state=seed)
    Atr=pca.fit_transform(Xs);Ate=pca.transform(Xt)
    scale=np.maximum(np.max(np.abs(Atr),axis=0),1e-9)
    out={**pairs,'X_train':(np.clip(Atr/scale,-1,1)*np.pi).astype(np.float32),
         'X_test':(np.clip(Ate/scale,-1,1)*np.pi).astype(np.float32),
         'y_train':np.r_[np.ones(len(pairs['train_pos'])),np.zeros(len(pairs['train_neg']))].astype(np.int8),
         'y_test':np.r_[np.ones(len(pairs['test_pos'])),np.zeros(len(pairs['test_neg']))].astype(np.int8),
         'seed':np.array([seed]),'regime':np.array(['source_degree_matched']),
         'pca_explained_variance_ratio':pca.explained_variance_ratio_,'angle_scale':scale,
         'pair_scaler_mean':scaler.mean_,'pair_scaler_scale':scaler.scale_,
         'pca_components':pca.components_,'pca_mean':pca.mean_}
    return out


if __name__=='__main__':
    import argparse
    from pathlib import Path
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--numeric-data',type=Path,required=True)
    a.add_argument('--splits',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--seeds',type=int,nargs='+',default=list(range(42,62)))
    args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    z=np.load(args.numeric_data,allow_pickle=False)
    for seed in args.seeds:
        p=np.load(args.splits/f'pairs_seed{seed}.npz',allow_pickle=False)
        out=project(z['content_svd32'],dict(p),seed)
        np.savez_compressed(args.out/f'qml_dataset_source_degree_matched_seed{seed}.npz',**out)
        print(f'Projected seed {seed}',flush=True)
