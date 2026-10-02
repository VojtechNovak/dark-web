"""Function-evaluation-budgeted optimizers for VQC experiments.

Each optimizer is a callable compatible with Qiskit Machine Learning VQC's
``optimizer=`` argument. The scientific contract is that ``max_fev`` counts
calls to the *same batch loss function*. This makes COBYLA, SPSA, iSOMA and jSO
comparable by objective evaluations rather than iteration counts.

The implementations deliberately include x0 in population methods, use local
RNGs, and check the budget before every objective call.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


def _optimizer_result():
    try:
        from qiskit_machine_learning.optimizers import OptimizerResult
        return OptimizerResult()
    except Exception:
        return SimpleNamespace()


def _bounds_arrays(x0: np.ndarray, bounds: Optional[Sequence[Tuple[float | None, float | None]]]):
    d = len(x0)
    if bounds is None:
        return np.full(d, -np.pi), np.full(d, np.pi)
    lo = np.asarray([(-np.pi if b[0] is None else b[0]) for b in bounds], float)
    hi = np.asarray([( np.pi if b[1] is None else b[1]) for b in bounds], float)
    return lo, hi


class _Tracker:
    def __init__(self, fun: Callable[[np.ndarray], float], max_fev: int):
        self.fun = fun
        self.max_fev = int(max_fev)
        self.nfev = 0
        self.best_fun = math.inf
        self.best_x = None
        self.history = []

    def can_eval(self, n: int = 1) -> bool:
        return self.nfev + n <= self.max_fev

    def __call__(self, x: np.ndarray) -> float:
        if not self.can_eval():
            raise RuntimeError('function-evaluation budget exhausted')
        value = float(self.fun(np.asarray(x, float)))
        self.nfev += 1
        if value < self.best_fun:
            self.best_fun = value
            self.best_x = np.asarray(x, float).copy()
        self.history.append((self.nfev, self.best_fun, value))
        return value


def _result_from_tracker(tr: _Tracker, x: np.ndarray, fun: float, nit: int):
    r = _optimizer_result()
    r.x = np.asarray(x, float)
    r.fun = float(fun)
    r.nfev = int(tr.nfev)
    r.njev = 0
    r.nit = int(nit)
    return r


class BudgetedCOBYLA:
    def __init__(self, max_fev: int, rhobeg: float = 0.5, tol: float = 1e-4):
        self.max_fev = int(max_fev)
        self.rhobeg = float(rhobeg)
        self.tol = float(tol)
        self.history = []

    def __call__(self, fun, x0, jac=None, bounds=None):
        x0 = np.asarray(x0, float)
        lo, hi = _bounds_arrays(x0, bounds)
        tr = _Tracker(fun, self.max_fev)
        # SciPy COBYLA's maxiter is a function-evaluation cap in current PRIMA-based SciPy.
        res = minimize(tr, np.clip(x0, lo, hi), method='COBYLA', bounds=list(zip(lo, hi)),
                       options={'maxiter': self.max_fev, 'rhobeg': self.rhobeg, 'tol': self.tol, 'disp': False})
        self.history = tr.history
        xbest = tr.best_x if tr.best_x is not None else np.asarray(res.x, float)
        fbest = tr.best_fun if np.isfinite(tr.best_fun) else float(res.fun)
        return _result_from_tracker(tr, xbest, fbest, getattr(res, 'nit', tr.nfev))


class BudgetedSPSA:
    """Simple first-order SPSA with an exact objective-evaluation budget.

    Each complete step consumes 3 loss calls: theta+c*delta, theta-c*delta, and
    the proposed new theta. This is intentionally explicit for resource accounting.
    """
    def __init__(self, max_fev: int, seed: int = 0, a: float = 0.15, c: float = 0.10,
                 alpha: float = 0.602, gamma: float = 0.101, A: float = 10.0):
        self.max_fev = int(max_fev); self.seed = int(seed)
        self.a = float(a); self.c = float(c); self.alpha = float(alpha); self.gamma = float(gamma); self.A = float(A)
        self.history = []

    def __call__(self, fun, x0, jac=None, bounds=None):
        rng = np.random.default_rng(self.seed)
        x = np.asarray(x0, float).copy()
        lo, hi = _bounds_arrays(x, bounds); x = np.clip(x, lo, hi)
        tr = _Tracker(fun, self.max_fev)
        fcur = tr(x) if tr.can_eval() else math.inf
        k = 0
        while tr.can_eval(3):
            ak = self.a / ((self.A + k + 1) ** self.alpha)
            ck = self.c / ((k + 1) ** self.gamma)
            delta = rng.choice([-1.0, 1.0], size=len(x))
            xp = np.clip(x + ck * delta, lo, hi)
            xm = np.clip(x - ck * delta, lo, hi)
            fp = tr(xp); fm = tr(xm)
            g = ((fp - fm) / (2.0 * ck)) * delta  # 1/delta == delta for Rademacher perturbations
            xnew = np.clip(x - ak * g, lo, hi)
            fnew = tr(xnew)
            x, fcur = xnew, fnew
            k += 1
        self.history = tr.history
        xbest = tr.best_x if tr.best_x is not None else x
        return _result_from_tracker(tr, xbest, tr.best_fun, k)


class BudgetedISOMA:
    """Budget-safe iSOMA/SOMA-style population optimizer.

    Fixes three issues common in ad-hoc VQC wrappers: x0 is retained, randomness
    is local/reproducible, and no migration is allowed to overshoot max_fev.
    """
    def __init__(self, max_fev: int, seed: int = 0, pop_size: int = 24, n_jump: int = 10,
                 step: float = 0.30, migrants: int = 5, leader_pool: int = 12):
        self.max_fev = int(max_fev); self.seed = int(seed); self.pop_size = int(pop_size)
        self.n_jump = int(n_jump); self.step = float(step); self.migrants = int(migrants); self.leader_pool = int(leader_pool)
        self.history = []

    def __call__(self, fun, x0, jac=None, bounds=None):
        rng = np.random.default_rng(self.seed)
        x0 = np.asarray(x0, float); d = len(x0)
        lo, hi = _bounds_arrays(x0, bounds)
        tr = _Tracker(fun, self.max_fev)
        ps = max(2, min(self.pop_size, self.max_fev))
        pop = rng.uniform(lo, hi, size=(ps, d)); pop[0] = np.clip(x0, lo, hi)
        fit = np.full(ps, np.inf)
        for i in range(ps):
            if not tr.can_eval(): break
            fit[i] = tr(pop[i])
        migration = 0
        while tr.can_eval():
            migration += 1
            m = min(self.migrants, ps)
            migrant_idx = rng.choice(ps, size=m, replace=False)
            migrant_idx = migrant_idx[np.argsort(fit[migrant_idx])]
            any_eval = False
            for mi in migrant_idx:
                if not tr.can_eval(): break
                k = min(self.leader_pool, ps)
                leaders = rng.choice(ps, size=k, replace=False)
                leaders = leaders[np.argsort(fit[leaders])]
                li = leaders[0] if leaders[0] != mi else (leaders[1] if len(leaders) > 1 else leaders[0])
                migrant = pop[mi].copy(); leader = pop[li].copy()
                for move in range(1, self.n_jump + 1):
                    if not tr.can_eval(): break
                    nstep = (self.n_jump - move + 1) * self.step
                    prt = 0.1 + 0.9 * (tr.nfev / max(1, self.max_fev))
                    mask = (rng.random(d) < prt).astype(float)
                    if not mask.any(): mask[rng.integers(d)] = 1.0
                    child = np.clip(migrant + (leader - migrant) * nstep * mask, lo, hi)
                    fc = tr(child); any_eval = True
                    if fc <= fit[mi]:
                        pop[mi], fit[mi] = child, fc
                        break
            if not any_eval:
                break
        self.history = tr.history
        return _result_from_tracker(tr, tr.best_x, tr.best_fun, migration)


class BudgetedJSO:
    """Budget-safe jSO/L-SHADE-style differential evolution optimizer."""
    def __init__(self, max_fev: int, seed: int = 0, pop_size: int | None = None, memory_size: int = 5):
        self.max_fev = int(max_fev); self.seed = int(seed); self.pop_size = pop_size; self.memory_size = int(memory_size)
        self.history = []

    def __call__(self, fun, x0, jac=None, bounds=None):
        rng = np.random.default_rng(self.seed)
        x0 = np.asarray(x0, float); d = len(x0); lo, hi = _bounds_arrays(x0, bounds)
        tr = _Tracker(fun, self.max_fev)
        if self.pop_size is None:
            ps0 = max(8, int(10 * np.sqrt(max(1, d))))
        else:
            ps0 = int(self.pop_size)
        ps0 = max(4, min(ps0, self.max_fev))
        psmin = min(4, ps0); H = self.memory_size
        pop = rng.uniform(lo, hi, size=(ps0, d)); pop[0] = np.clip(x0, lo, hi)
        fit = np.full(ps0, np.inf)
        for i in range(ps0):
            if not tr.can_eval(): break
            fit[i] = tr(pop[i])
        M_F = np.full(H, 0.5); M_CR = np.full(H, 0.8); k_mem = 0
        archive = []
        generations = 0
        while tr.can_eval() and len(pop) >= 4:
            generations += 1
            progress = tr.nfev / max(1, self.max_fev)
            target_ps = int(round((psmin - ps0) * progress + ps0))
            target_ps = max(psmin, min(len(pop), target_ps))
            if len(pop) > target_ps:
                keep = np.argsort(fit)[:target_ps]; pop, fit = pop[keep], fit[keep]
            ps = len(pop)
            p = 0.25 - 0.125 * progress; pbest_n = max(2, int(round(p * ps)))
            mem_idx = rng.integers(0, H, size=ps)
            F = M_F[mem_idx] + 0.1 * rng.standard_cauchy(ps)
            bad = F <= 0
            while np.any(bad):
                F[bad] = M_F[mem_idx[bad]] + 0.1 * rng.standard_cauchy(np.sum(bad)); bad = F <= 0
            F = np.minimum(F, 1.0)
            CR = np.clip(rng.normal(M_CR[mem_idx], 0.1), 0.0, 1.0)
            order = np.argsort(fit); combined = np.vstack([pop] + ([np.asarray(archive)] if archive else []))
            SF=[]; SCR=[]; DIF=[]
            for i in range(ps):
                if not tr.can_eval(): break
                pbest = pop[rng.choice(order[:pbest_n])]
                r1 = rng.integers(ps)
                while r1 == i: r1 = rng.integers(ps)
                r2 = rng.integers(len(combined))
                # if r2 refers to population, avoid i/r1; archive indices need no such restriction
                while r2 < ps and (r2 == i or r2 == r1): r2 = rng.integers(len(combined))
                mutant = pop[i] + F[i] * (pbest - pop[i]) + F[i] * (pop[r1] - combined[r2])
                mask = rng.random(d) < CR[i]; mask[rng.integers(d)] = True
                trial = np.clip(np.where(mask, mutant, pop[i]), lo, hi)
                ft = tr(trial)
                if ft <= fit[i]:
                    if ft < fit[i]:
                        archive.append(pop[i].copy()); SF.append(F[i]); SCR.append(CR[i]); DIF.append(fit[i]-ft)
                    pop[i], fit[i] = trial, ft
            if len(archive) > len(pop):
                keep = rng.choice(len(archive), size=len(pop), replace=False); archive = [archive[j] for j in keep]
            if DIF and np.sum(DIF) > 0:
                w = np.asarray(DIF) / np.sum(DIF); sf=np.asarray(SF); scr=np.asarray(SCR)
                den = np.sum(w*sf)
                mean_f = np.sum(w*sf*sf)/den if den > 0 else M_F[k_mem]
                mean_cr = np.sum(w*scr)
                M_F[k_mem] = 0.5*M_F[k_mem] + 0.5*mean_f
                M_CR[k_mem] = 0.5*M_CR[k_mem] + 0.5*mean_cr
                k_mem = (k_mem + 1) % H
        self.history = tr.history
        return _result_from_tracker(tr, tr.best_x, tr.best_fun, generations)


def make_optimizer(name: str, max_fev: int, seed: int):
    name = name.lower()
    if name == 'cobyla': return BudgetedCOBYLA(max_fev=max_fev)
    if name == 'spsa': return BudgetedSPSA(max_fev=max_fev, seed=seed)
    if name in {'isoma','soma'}: return BudgetedISOMA(max_fev=max_fev, seed=seed)
    if name == 'jso': return BudgetedJSO(max_fev=max_fev, seed=seed)
    raise ValueError(f'Unknown optimizer {name}')
