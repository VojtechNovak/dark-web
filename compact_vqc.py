"""Batched statevector simulation of explicitly defined compact VQC circuits.

Qubits use little-endian indexing, as in Qiskit. No hardware or gate noise is
modeled. Binomial sampling is exactly the marginal distribution of the binary
all-bit parity measurement with a fixed number of independent shots.
"""
from __future__ import annotations
import numpy as np


class Circuit:
    def __init__(self, qubits=4, layers=2):
        self.qubits=qubits; self.layers=layers
        self.n_parameters=2*qubits*layers
        self.basis=np.arange(2**qubits)
        self.bits=(self.basis[:,None]>>np.arange(qubits))&1
        self.odd=self.bits.sum(axis=1)%2==1
        self.zeros=[self.basis[(self.basis&(1<<q))==0] for q in range(qubits)]
        self.ones=[i|(1<<q) for q,i in enumerate(self.zeros)]
        self.cnot=[self.basis^(((self.basis>>q)&1)<<(q+1)) for q in range(qubits-1)]

    def ry(self,state,q,angle):
        angle=np.asarray(angle)
        if angle.ndim: angle=angle[:,None]
        c=np.cos(angle/2);s=np.sin(angle/2)
        a=state[:,self.zeros[q]].copy();b=state[:,self.ones[q]].copy()
        state[:,self.zeros[q]]=c*a-s*b
        state[:,self.ones[q]]=s*a+c*b

    def encode(self,X,encoding):
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[1]!=self.qubits:raise ValueError('Wrong feature dimension')
        if encoding=='ry':
            state=np.zeros((len(X),2**self.qubits),complex);state[:,0]=1
            for q in range(self.qubits):self.ry(state,q,X[:,q])
        elif encoding=='zz':
            phase=2*X@self.bits.T
            for q in range(self.qubits-1):
                phi=2*(np.pi-X[:,q])*(np.pi-X[:,q+1])
                phase+=phi[:,None]*(self.bits[:,q]^self.bits[:,q+1])[None,:]
            state=np.exp(1j*phase)/np.sqrt(2**self.qubits)
        else:raise ValueError(encoding)
        return state

    def probabilities(self,encoded,theta):
        theta=np.asarray(theta,float)
        if theta.shape!=(self.n_parameters,):raise ValueError('Wrong parameter count')
        state=np.asarray(encoded,complex).copy();n=self.qubits
        for layer in range(self.layers):
            offset=2*n*layer
            # RZ is followed by RY, so no terminal Z rotations are invisible.
            phase=(self.bits-.5)@theta[offset:offset+n]
            state*=np.exp(1j*phase)[None,:]
            for q in range(n):self.ry(state,q,theta[offset+n+q])
            if layer<self.layers-1:
                for perm in self.cnot:state=state[:,perm]
        p=np.sum(np.abs(state[:,self.odd])**2,axis=1)
        return np.clip(p,0,1)

    def gate_counts(self,encoding):
        n=self.qubits
        return {'one_qubit':(n if encoding=='ry' else 3*n-1)+self.n_parameters,
                'cnot':(0 if encoding=='ry' else 2*(n-1))+(self.layers-1)*(n-1)}


def validate():
    """Compare with explicit dense gate matrices and audit all input/angle effects."""
    rng=np.random.default_rng(20261001);c=Circuit();n=c.qubits
    H=np.array([[1,1],[1,-1]])/np.sqrt(2)
    def local(g,q):
        out=np.ones((1,1),complex)
        for bit in reversed(range(n)):out=np.kron(out,g if bit==q else np.eye(2))
        return out
    def cx(control,target):
        out=np.zeros((2**n,2**n),complex)
        for b in range(2**n):out[b^(((b>>control)&1)<<target),b]=1
        return out
    def dense(x,theta,encoding):
        s=np.zeros(2**n,complex);s[0]=1
        def ry(t):return np.array([[np.cos(t/2),-np.sin(t/2)],[np.sin(t/2),np.cos(t/2)]])
        if encoding=='ry':
            for q in range(n):s=local(ry(x[q]),q)@s
        else:
            for q in range(n):s=local(H,q)@s;s=local(np.diag([1,np.exp(2j*x[q])]),q)@s
            for q in range(n-1):
                U=cx(q,q+1);s=U@s
                s=local(np.diag([1,np.exp(2j*(np.pi-x[q])*(np.pi-x[q+1]))]),q+1)@s
                s=U@s
        for layer in range(c.layers):
            off=2*n*layer
            for q in range(n):s=local(np.diag(np.exp(np.array([-1j,1j])*theta[off+q]/2)),q)@s
            for q in range(n):s=local(ry(theta[off+n+q]),q)@s
            if layer<c.layers-1:
                for q in range(n-1):s=cx(q,q+1)@s
        return np.sum(np.abs(s[c.odd])**2)
    report=[]
    for encoding in ['ry','zz']:
        maxerr=0.;dx=np.zeros(n);dt=np.zeros(c.n_parameters)
        for _ in range(12):
            x=rng.uniform(-2.5,2.5,(1,n));t=rng.uniform(-3,3,c.n_parameters)
            p=c.probabilities(c.encode(x,encoding),t)[0]
            maxerr=max(maxerr,abs(p-dense(x[0],t,encoding)))
            for q in range(n):
                y=x.copy();y[0,q]+=.31
                dx[q]=max(dx[q],abs(c.probabilities(c.encode(y,encoding),t)[0]-p))
            for q in range(c.n_parameters):
                u=t.copy();u[q]+=.23
                dt[q]=max(dt[q],abs(c.probabilities(c.encode(x,encoding),u)[0]-p))
        assert maxerr<1e-12
        assert (dx>1e-8).all() and (dt>1e-8).all()
        X=rng.uniform(-3,3,(80,n));E=c.encode(X,encoding);t=rng.uniform(-3,3,c.n_parameters)
        jac=np.column_stack([(c.probabilities(E,t+np.eye(c.n_parameters)[q]*1e-5)-c.probabilities(E,t-np.eye(c.n_parameters)[q]*1e-5))/2e-5 for q in range(c.n_parameters)])
        rank=int(np.linalg.matrix_rank(jac,tol=1e-7))
        report.append({'encoding':encoding,'max_dense_error':maxerr,'active_inputs':int(sum(dx>1e-8)),
                       'active_parameters':int(sum(dt>1e-8)),'sampled_jacobian_rank':rank,**c.gate_counts(encoding)})
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import zz_feature_map
        from qiskit.quantum_info import Statevector
    except ImportError:
        sdk='Qiskit unavailable; independent dense-gate checks passed.'
    else:
        for encoding in ['ry','zz']:
            for _ in range(5):
                x=rng.uniform(-2,2,n);t=rng.uniform(-3,3,c.n_parameters)
                qc=QuantumCircuit(n)
                if encoding=='ry':
                    for q in range(n):qc.ry(float(x[q]),q)
                else:qc=zz_feature_map(n,reps=1,entanglement='linear').assign_parameters(x)
                for layer in range(c.layers):
                    off=2*n*layer
                    for q in range(n):qc.rz(float(t[off+q]),q)
                    for q in range(n):qc.ry(float(t[off+n+q]),q)
                    if layer<c.layers-1:
                        for q in range(n-1):qc.cx(q,q+1)
                p=Statevector.from_instruction(qc).probabilities()[c.odd].sum()
                assert abs(p-c.probabilities(c.encode(x[None,:],encoding),t)[0])<1e-12
        sdk='Direct Qiskit Statevector checks passed.'
    return {'circuits':report,'sdk_check':sdk}


if __name__=='__main__':
    import json
    print(json.dumps(validate(),indent=2))
