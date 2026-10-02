"""Directed GAE/VGAE with explicit pair-indexed loss and analytic NumPy gradients.

Same layer/decoder/objective definitions as the manuscript's original PyTorch
models. New NumPy initialization draws and float64 arithmetic: these are new
controlled experiments, not bitwise reproduction of archived PyTorch fits.
"""
import numpy as np
from scipy.special import expit

class Autoencoder:
    def __init__(self, din, hidden=48, latent=24, variational=False, messages=True, seed=42):
        self.rng=np.random.default_rng(seed);self.var=variational;self.msg=messages
        self.hidden=hidden;self.latent=latent;self.P={}
        def layer(name,di,do):
            b=1/np.sqrt(di)
            for k in ['s','o','i'] if messages else ['s']:
                self.P[name+k]=self.rng.uniform(-b,b,(di,do))
                if k=='s': self.P[name+'b']=self.rng.uniform(-b,b,do)
        layer('h',din,hidden)
        for name in ['m','v'] if variational else ['z']: layer(name,hidden,latent)
        b=1/np.sqrt(latent)
        self.P['src']=self.rng.uniform(-b,b,(latent,latent))
        self.P['dst']=self.rng.uniform(-b,b,(latent,latent))
        self.P['bias']=np.array(0.)

    def forward(self,X,Ao,Ai,eps=None):
        cache={}
        def layer(name,H):
            Q=[H,Ao@H,Ai@H] if self.msg else [H]
            cache[name]=(H,Q)
            return sum(q@self.P[name+k] for q,k in zip(Q,['s','o','i']))+self.P[name+'b']
        pre=layer('h',X);H=np.maximum(pre,0);cache['relu']=pre>0
        if self.var:
            mu=layer('m',H);raw=layer('v',H);lv=np.clip(raw,-8,8)
            sd=np.exp(lv*.5);z=mu if eps is None else mu+sd*eps
            cache.update(mu=mu,lv=lv,raw=raw,sd=sd,eps=eps)
        else:z=layer('z',H)
        return z,cache

    def objective(self,X,Ao,Ai,pairs,y,kl_weight=.001,eps=None):
        z,C=self.forward(X,Ao,Ai,eps)
        U=z@self.P['src'];V=z@self.P['dst'];u,v=pairs.T
        logits=np.sum(U[u]*V[v],axis=1)/np.sqrt(self.latent)+self.P['bias']
        bce=np.mean(np.logaddexp(0,logits)-y*logits)
        kl=0. if not self.var else -.5*np.mean(1+C['lv']-C['mu']**2-np.exp(C['lv']))
        dl=(expit(logits)-y)/len(y);dU=np.zeros_like(U);dV=np.zeros_like(V)
        np.add.at(dU,u,dl[:,None]*V[v]/np.sqrt(self.latent))
        np.add.at(dV,v,dl[:,None]*U[u]/np.sqrt(self.latent))
        grads={'src':z.T@dU,'dst':z.T@dV,'bias':dl.sum()}
        dz=dU@self.P['src'].T+dV@self.P['dst'].T
        def backward(name,d):
            _,Q=C[name];grads[name+'b']=d.sum(axis=0)
            dh=np.zeros((len(X),Q[0].shape[1]))
            for q,k in zip(Q,['s','o','i']):
                grads[name+k]=q.T@d
                tmp=d@self.P[name+k].T
                dh+=tmp if k=='s' else (Ao.T if k=='o' else Ai.T)@tmp
            return dh
        if self.var:
            n=C['mu'].size
            dmu=dz+kl_weight*C['mu']/n
            dlv=kl_weight*.5*(np.exp(C['lv'])-1)/n
            if eps is not None:dlv+=dz*.5*C['sd']*eps
            dlv*= (C['raw']>-8)&(C['raw']<8)
            dh=backward('m',dmu)+backward('v',dlv)
        else:dh=backward('z',dz)
        backward('h',dh*C['relu'])
        return float(bce+kl_weight*kl),grads,float(bce),float(kl)

    def train(self,X,Ao,Ai,pairs,y,epochs=80,kl_weight=.001,lr=.01,wd=1e-4):
        m={k:np.zeros_like(v) for k,v in self.P.items()};v={k:q.copy() for k,q in m.items()}
        best=np.inf;state=None;history=[]
        # Original checkpoint rule: training objective before update, state after
        # update. No held-out labels or metrics select weights or epochs.
        for t in range(1,epochs+1):
            eps=self.rng.normal(size=(len(X),self.latent)) if self.var else None
            loss,g,bce,kl=self.objective(X,Ao,Ai,pairs,y,kl_weight,eps)
            if not np.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            for k in self.P:
                gg=g[k]+wd*self.P[k]
                m[k]=.9*m[k]+.1*gg;v[k]=.999*v[k]+.001*gg**2
                self.P[k]-=lr*(m[k]/(1-.9**t))/(np.sqrt(v[k]/(1-.999**t))+1e-8)
            history.append([t,loss,bce,kl])
            if loss<best-1e-6:best=loss;state={k:q.copy() for k,q in self.P.items()};best_epoch=t
        self.P=state
        z,_=self.forward(X,Ao,Ai)
        return z,np.array(history),best_epoch

    def factors(self,z):return z@self.P['src'],z@self.P['dst']
    def n_parameters(self):return sum(v.size for v in self.P.values())


def gradient_check():
    rng=np.random.default_rng(891);X=rng.normal(size=(7,5));Ao=rng.uniform(size=(7,7));Ao/=Ao.sum(1)[:,None];Ai=Ao.T.copy()
    pairs=np.array([[0,1],[0,2],[3,1],[4,5]]);y=np.array([1,0,1,0]);report=[]
    for var in [False,True]:
        for messages in [False,True]:
            model=Autoencoder(5,4,3,var,messages,27)
            eps=rng.normal(size=(7,3)) if var else None
            val,grad,_,_=model.objective(X,Ao,Ai,pairs,y,.001,eps);err=0.
            for key,a in model.P.items():
                for flat in range(a.size):
                    ix=np.unravel_index(flat,a.shape);old=a[ix];h=1e-6
                    a[ix]=old+h;p=model.objective(X,Ao,Ai,pairs,y,.001,eps)[0]
                    a[ix]=old-h;n=model.objective(X,Ao,Ai,pairs,y,.001,eps)[0];a[ix]=old
                    err=max(err,abs((p-n)/(2*h)-np.asarray(grad[key])[ix]))
            assert err<2e-7,(var,messages,err)
            report.append({'variational':var,'messages':messages,'max_abs_error':err})
    return report
