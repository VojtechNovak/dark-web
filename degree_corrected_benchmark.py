#!/usr/bin/env python3
"""Bias-aware held-out link prediction for directed dark-web hyperlink graphs.

Designed as an extension to darkweb_ml_study.py. Main additions:
  * regime-specific training negatives (train/test distributions are matched),
  * directed Aiyappa-style degree-corrected negatives,
  * source-conditioned target-in-degree-matched negatives,
  * optional degree+semantic-hard negatives,
  * explicit degree-only diagnostics,
  * classical models on exactly the same PCA-reduced input exported for QML,
  * optional filtered source->target ranking metrics.

The primary recommended regime is ``source_degree_matched``. It fixes the source
node for every positive and selects a non-edge target with nearly the same
training in-degree as the positive target. This strongly suppresses the trivial
out-degree/in-degree shortcut while remaining natural for directed recommendation.

No network access is used; HTML is not parsed here. If --host-features points to
host_features_full.csv from darkweb_ml_study.py, content views are evaluated.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

Pair = Tuple[str, str]


@dataclass
class SplitData:
    train_graph: nx.DiGraph
    train_pos: List[Pair]
    test_pos: List[Pair]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nodes', required=True)
    p.add_argument('--edges', required=True)
    p.add_argument('--host-features', default=None,
                   help='host_features_full.csv produced by darkweb_ml_study.py; enables content models')
    p.add_argument('--out', default='degree_corrected_output')
    p.add_argument('--status', type=int, default=200)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-size', type=float, default=0.20)
    p.add_argument('--graph-dim', type=int, default=32)
    p.add_argument('--text-dim', type=int, default=32)
    p.add_argument('--qdim', type=int, default=8)
    p.add_argument('--max-features', type=int, default=6000)
    p.add_argument('--export-qml', action='store_true')
    p.add_argument('--ranking', action='store_true', help='Also compute filtered all-candidate MRR/Hits on primary regime')
    p.add_argument('--primary-regime', default='source_degree_matched',
                   choices=['degree_corrected', 'source_degree_matched', 'source_degree_semantic_hard'])
    p.add_argument('--regimes', nargs='*', default=[
        'random', 'topology_hard', 'semantic_hard', 'degree_corrected',
        'source_degree_matched', 'source_degree_semantic_hard'
    ])
    return p.parse_args()


def safe_host(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ''
    try:
        return (urlparse(url).hostname or '').lower().strip('.')
    except Exception:
        return ''


def build_host_graph(nodes: pd.DataFrame, edges: pd.DataFrame, status: int = 200) -> nx.DiGraph:
    nodes = nodes.copy()
    nodes['host'] = nodes['full_link'].map(safe_host)
    ok = nodes.loc[nodes['status'] == status].copy()
    ids = set(ok['id'])
    e = edges.loc[edges['source_id'].isin(ids) & edges['target_id'].isin(ids)].copy()
    G = nx.DiGraph()
    G.add_nodes_from(ids)
    G.add_edges_from(e[['source_id', 'target_id']].itertuples(index=False, name=None))
    core = max(nx.weakly_connected_components(G), key=len)
    ok = ok.loc[ok['id'].isin(core)].copy()
    e = e.loc[e['source_id'].isin(core) & e['target_id'].isin(core)].copy()
    id2host = ok.set_index('id')['host'].to_dict()
    H = nx.DiGraph()
    H.add_nodes_from(sorted(h for h in ok['host'].dropna().unique() if h))
    for s, t in e[['source_id', 'target_id']].itertuples(index=False, name=None):
        hs, ht = id2host.get(s, ''), id2host.get(t, '')
        if hs and ht and hs != ht:
            H.add_edge(hs, ht)
    return H


def make_split(H: nx.DiGraph, test_size: float, seed: int) -> SplitData:
    """Hold out whole undirected dyads while preserving one spanning tree."""
    rng = np.random.default_rng(seed)
    full_edge_list = sorted(H.edges())
    full_edges = set(full_edge_list)
    U = H.to_undirected()
    tree = nx.minimum_spanning_tree(U)
    protected = {tuple(sorted(e)) for e in tree.edges()}
    dyads: Dict[Tuple[str, str], List[Pair]] = defaultdict(list)
    for u, v in full_edge_list:
        dyads[tuple(sorted((u, v)))].append((u, v))
    candidates = [d for d in dyads if d not in protected]
    rng.shuffle(candidates)
    n_test = max(1, int(round(test_size * len(candidates))))
    test_dyads = set(candidates[:n_test])
    test_pos = [e for d in test_dyads for e in dyads[d]]
    train_pos = [e for e in full_edge_list if tuple(sorted(e)) not in test_dyads]
    Gtr = nx.DiGraph()
    Gtr.add_nodes_from(H.nodes())
    Gtr.add_edges_from(train_pos)
    return SplitData(Gtr, train_pos, test_pos)


def _valid_nonedge(u: str, v: str, full_edges: set[Pair], used: set[Pair]) -> bool:
    return u != v and (u, v) not in full_edges and (u, v) not in used


def random_negatives(H: nx.DiGraph, n: int, seed: int, used: Iterable[Pair] = ()) -> List[Pair]:
    rng = np.random.default_rng(seed)
    nodes = list(H.nodes())
    full = set(H.edges())
    used = set(used)
    out: List[Pair] = []
    for _ in range(max(10000, n * 300)):
        if len(out) >= n:
            break
        u = nodes[int(rng.integers(len(nodes)))]
        v = nodes[int(rng.integers(len(nodes)))]
        if _valid_nonedge(u, v, full, used):
            used.add((u, v)); out.append((u, v))
    if len(out) != n:
        raise RuntimeError(f'Random sampler produced {len(out)}/{n} negatives')
    return out


def topology_hard_negatives(H: nx.DiGraph, Gtr: nx.DiGraph, n: int, seed: int) -> List[Pair]:
    rng = np.random.default_rng(seed)
    full = set(H.edges())
    U = Gtr.to_undirected()
    pool: List[Pair] = []
    for u in U.nodes():
        two = set()
        for nbr in U.neighbors(u):
            two.update(U.neighbors(nbr))
        two.discard(u)
        for v in sorted(two):
            if (u, v) not in full:
                pool.append((u, v))
    if len(pool) < n:
        comms = nx.community.louvain_communities(U, seed=seed) if U.number_of_edges() else [{x} for x in U.nodes()]
        cmap = {x: i for i, c in enumerate(comms) for x in c}
        nodes = list(U.nodes())
        for u in nodes:
            for v in nodes:
                if u != v and cmap[u] == cmap[v] and (u, v) not in full:
                    pool.append((u, v))
    pool = list(dict.fromkeys(pool))
    rng.shuffle(pool)
    out = pool[:n]
    if len(out) < n:
        out += random_negatives(H, n - len(out), seed + 991, used=out)
    return out


def directed_degree_corrected_negatives(H: nx.DiGraph, Gtr: nx.DiGraph, n: int, seed: int,
                                        used: Iterable[Pair] = ()) -> List[Pair]:
    """Directed extension of Aiyappa et al.: sample source ∝ out-degree, target ∝ in-degree."""
    rng = np.random.default_rng(seed)
    nodes = list(H.nodes())
    full = set(H.edges())
    used = set(used)
    sw = np.asarray([Gtr.out_degree(u) for u in nodes], dtype=float)
    tw = np.asarray([Gtr.in_degree(v) for v in nodes], dtype=float)
    if sw.sum() <= 0: sw[:] = 1.0
    if tw.sum() <= 0: tw[:] = 1.0
    sw /= sw.sum(); tw /= tw.sum()
    out: List[Pair] = []
    for _ in range(max(20000, n * 1000)):
        if len(out) >= n:
            break
        u = nodes[int(rng.choice(len(nodes), p=sw))]
        v = nodes[int(rng.choice(len(nodes), p=tw))]
        if _valid_nonedge(u, v, full, used):
            used.add((u, v)); out.append((u, v))
    if len(out) < n:
        out += random_negatives(H, n - len(out), seed + 1777, used=used)
    return out


def source_degree_matched_negatives(H: nx.DiGraph, Gtr: nx.DiGraph, positives: Sequence[Pair], seed: int,
                                    Z_content: np.ndarray | None = None,
                                    node_index: Dict[str, int] | None = None,
                                    semantic_hard: bool = False) -> List[Pair]:
    """For each (u,v)+ choose (u,w)-, minimizing target in-degree mismatch.

    When semantic_hard=True, first keep targets in the best available degree-distance
    tier, then pick the candidate with the largest cosine similarity to source u.
    """
    rng = np.random.default_rng(seed)
    nodes = list(H.nodes())
    full = set(H.edges())
    used: set[Pair] = set()
    out: List[Pair | None] = [None] * len(positives)

    Zn = None
    if semantic_hard:
        if Z_content is None or node_index is None:
            raise ValueError('semantic_hard matching requires content embeddings')
        Z = np.asarray(Z_content, float)
        Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

    # Hardest positives first: high target in-degree tends to have fewer matched non-neighbors.
    order = sorted(range(len(positives)), key=lambda i: Gtr.in_degree(positives[i][1]), reverse=True)
    for i in order:
        u, v = positives[i]
        target_deg = Gtr.in_degree(v)
        candidates = [w for w in nodes if _valid_nonedge(u, w, full, used)]
        if not candidates:
            continue
        d = np.asarray([abs(Gtr.in_degree(w) - target_deg) for w in candidates])
        best_dist = d.min()
        # Permit the nearest degree tier only. This makes the diagnostic interpretable.
        cand = [w for w, dd in zip(candidates, d) if dd == best_dist]
        if semantic_hard and Zn is not None:
            ui = node_index[u]
            sims = np.asarray([Zn[ui] @ Zn[node_index[w]] for w in cand])
            best = np.flatnonzero(sims >= np.max(sims) - 1e-12)
            w = cand[int(rng.choice(best))]
        else:
            w = cand[int(rng.integers(len(cand)))]
        used.add((u, w)); out[i] = (u, w)

    # Very dense sources can exhaust candidates. Fill any holes with directed degree correction.
    missing = [i for i, x in enumerate(out) if x is None]
    if missing:
        fill = directed_degree_corrected_negatives(H, Gtr, len(missing), seed + 31337,
                                                   used=[x for x in out if x is not None])
        for i, e in zip(missing, fill):
            out[i] = e
    return [x for x in out if x is not None]


def semantic_hard_negatives(H: nx.DiGraph, Z: np.ndarray, node_order: Sequence[str], n: int, seed: int,
                            used: Iterable[Pair] = ()) -> List[Pair]:
    rng = np.random.default_rng(seed)
    full = set(H.edges())
    used = set(used)
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    sim = Zn @ Zn.T
    cand = []
    m = len(node_order)
    for i in range(m):
        for j in range(m):
            if i == j: continue
            e = (node_order[i], node_order[j])
            if e in full or e in used: continue
            cand.append((float(sim[i, j]), rng.random(), e))
    cand.sort(reverse=True)
    return [e for _, _, e in cand[:n]]


def sample_negatives(regime: str, H: nx.DiGraph, Gtr: nx.DiGraph, positives: Sequence[Pair], seed: int,
                     Z_content: np.ndarray | None, node_order: Sequence[str]) -> List[Pair]:
    idx = {n: i for i, n in enumerate(node_order)}
    n = len(positives)
    if regime == 'random':
        return random_negatives(H, n, seed)
    if regime == 'topology_hard':
        return topology_hard_negatives(H, Gtr, n, seed)
    if regime == 'degree_corrected':
        return directed_degree_corrected_negatives(H, Gtr, n, seed)
    if regime == 'source_degree_matched':
        return source_degree_matched_negatives(H, Gtr, positives, seed)
    if regime == 'semantic_hard':
        if Z_content is None: raise ValueError('semantic_hard requires --host-features')
        return semantic_hard_negatives(H, Z_content, node_order, n, seed)
    if regime == 'source_degree_semantic_hard':
        if Z_content is None: raise ValueError('source_degree_semantic_hard requires --host-features')
        return source_degree_matched_negatives(H, Gtr, positives, seed, Z_content, idx, semantic_hard=True)
    raise ValueError(f'Unknown regime: {regime}')


def directed_graph_embedding(G: nx.DiGraph, node_order: Sequence[str], dim: int, seed: int) -> np.ndarray:
    idx = {n: i for i, n in enumerate(node_order)}
    rows, cols = zip(*[(idx[u], idx[v]) for u, v in G.edges()]) if G.number_of_edges() else ([], [])
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(node_order), len(node_order)))
    k = max(1, min(dim, len(node_order) - 2))
    Zo = TruncatedSVD(k, random_state=seed).fit_transform(A)
    Zi = TruncatedSVD(k, random_state=seed + 1).fit_transform(A.T)
    pr = nx.pagerank(G) if G.number_of_edges() else {n: 0.0 for n in node_order}
    D = np.asarray([[G.in_degree(n), G.out_degree(n), math.log1p(G.in_degree(n)),
                     math.log1p(G.out_degree(n)), pr.get(n, 0.0)] for n in node_order], float)
    D = StandardScaler().fit_transform(D)
    return np.hstack([Zo, Zi, D])


def pair_features(Z: np.ndarray, pairs: Sequence[Pair], idx: Dict[str, int]) -> np.ndarray:
    a = np.vstack([Z[idx[u]] for u, _ in pairs])
    b = np.vstack([Z[idx[v]] for _, v in pairs])
    return np.hstack([a, b, a * b, np.abs(a - b)])


def degree_pair_features(Gtr: nx.DiGraph, pairs: Sequence[Pair]) -> np.ndarray:
    return np.asarray([[math.log1p(Gtr.out_degree(u)), math.log1p(Gtr.in_degree(v)),
                        math.log1p(Gtr.out_degree(u) * Gtr.in_degree(v))] for u, v in pairs], float)


def heuristic_scores(Gtr: nx.DiGraph, pairs: Sequence[Pair]) -> Dict[str, np.ndarray]:
    U = Gtr.to_undirected()
    cn, jac, aa, pa = [], [], [], []
    neigh = {u: set(U.neighbors(u)) for u in U.nodes()}
    for u, v in pairs:
        nu, nv = neigh[u], neigh[v]
        inter = nu & nv; union = nu | nv
        cn.append(len(inter))
        jac.append(len(inter) / len(union) if union else 0.0)
        aa.append(sum(1.0 / math.log(U.degree(w)) for w in inter if U.degree(w) > 1))
        pa.append(Gtr.out_degree(u) * Gtr.in_degree(v))
    return {
        'common_neighbors': np.asarray(cn, float),
        'jaccard': np.asarray(jac, float),
        'adamic_adar': np.asarray(aa, float),
        'preferential_attachment_directed': np.asarray(pa, float),
    }


def fit_text(host_features: str, node_order: Sequence[str], dim: int, max_features: int, seed: int) -> np.ndarray:
    df = pd.read_csv(host_features, usecols=['host', 'content_text'])
    mp = df.set_index('host')['content_text'].fillna('').to_dict()
    texts = [mp.get(h, '') for h in node_order]
    vec = TfidfVectorizer(lowercase=True, stop_words=sorted(ENGLISH_STOP_WORDS), max_features=max_features,
                          min_df=2, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(texts)
    k = max(1, min(dim, X.shape[0] - 1, X.shape[1] - 1))
    return TruncatedSVD(k, random_state=seed).fit_transform(X)


def safe_metrics(y: np.ndarray, score: np.ndarray) -> Tuple[float, float]:
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def degree_diagnostics(Gtr: nx.DiGraph, pos: Sequence[Pair], neg: Sequence[Pair]) -> Dict[str, float]:
    def vals(pairs):
        so = np.asarray([math.log1p(Gtr.out_degree(u)) for u, _ in pairs])
        ti = np.asarray([math.log1p(Gtr.in_degree(v)) for _, v in pairs])
        pp = np.asarray([math.log1p(Gtr.out_degree(u) * Gtr.in_degree(v)) for u, v in pairs])
        return so, ti, pp
    pso, pti, ppa = vals(pos); nso, nti, npa = vals(neg)
    return {
        'w1_log_source_out': float(wasserstein_distance(pso, nso)),
        'w1_log_target_in': float(wasserstein_distance(pti, nti)),
        'w1_log_pa': float(wasserstein_distance(ppa, npa)),
        'mean_abs_target_indegree_gap': float(np.mean(np.abs(np.expm1(pti) - np.expm1(nti)))) if len(pos)==len(neg) else np.nan,
    }


def qml_projection(Xtr: np.ndarray, Xte: np.ndarray, qdim: int, seed: int):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    q = max(2, min(qdim, Xtr_s.shape[1], Xtr_s.shape[0] - 1))
    pca = PCA(n_components=q, random_state=seed)
    Atr = pca.fit_transform(Xtr_s); Ate = pca.transform(Xte_s)
    scale = np.maximum(np.max(np.abs(Atr), axis=0), 1e-9)
    Atr = np.clip(Atr / scale, -1, 1) * np.pi
    Ate = np.clip(Ate / scale, -1, 1) * np.pi
    return Atr, Ate, pca.explained_variance_ratio_, scale


def evaluate_one_regime(H, split, node_order, Zg, Zc, regime, seed, qdim, export_path=None):
    idx = {n: i for i, n in enumerate(node_order)}
    # Match training and test negative mechanisms. Use decorrelated seeds.
    train_neg = sample_negatives(regime, H, split.train_graph, split.train_pos, seed + 10000, Zc, node_order)
    test_neg = sample_negatives(regime, H, split.train_graph, split.test_pos, seed + 20000, Zc, node_order)
    train_pairs = split.train_pos + train_neg
    test_pairs = split.test_pos + test_neg
    ytr = np.r_[np.ones(len(split.train_pos)), np.zeros(len(train_neg))]
    yte = np.r_[np.ones(len(split.test_pos)), np.zeros(len(test_neg))]

    rows = []
    for model, score in heuristic_scores(split.train_graph, test_pairs).items():
        auc, ap = safe_metrics(yte, score)
        rows.append((model, auc, ap))

    # Explicit degree-only learned null.
    Xtr_d = degree_pair_features(split.train_graph, train_pairs)
    Xte_d = degree_pair_features(split.train_graph, test_pairs)
    degree_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed))
    degree_lr.fit(Xtr_d, ytr)
    rows.append(('degree_logreg', *safe_metrics(yte, degree_lr.predict_proba(Xte_d)[:, 1])))

    reps = [('graph_logreg', Zg)]
    if Zc is not None:
        reps += [('content_logreg', Zc), ('multimodal_logreg', np.hstack([Zg, Zc]))]
    for name, Z in reps:
        Xtr = pair_features(Z, train_pairs, idx); Xte = pair_features(Z, test_pairs, idx)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight='balanced', random_state=seed))
        clf.fit(Xtr, ytr)
        rows.append((name, *safe_metrics(yte, clf.predict_proba(Xte)[:, 1])))

    # Matched classical controls on exactly the quantum input.
    Zm = np.hstack([Zg, Zc]) if Zc is not None else Zg
    Xtr_full = pair_features(Zm, train_pairs, idx); Xte_full = pair_features(Zm, test_pairs, idx)
    Xtr_q, Xte_q, evr, qscale = qml_projection(Xtr_full, Xte_full, qdim, seed)
    matched = {
        'qml_input_logreg': LogisticRegression(max_iter=3000, class_weight='balanced', random_state=seed),
        'qml_input_rbf_svc': SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, class_weight='balanced', random_state=seed),
        'qml_input_mlp': MLPClassifier(hidden_layer_sizes=(16,), activation='tanh', alpha=1e-3,
                                       max_iter=1500, early_stopping=True, random_state=seed),
    }
    for name, clf in matched.items():
        clf.fit(Xtr_q, ytr)
        if hasattr(clf, 'predict_proba'):
            score = clf.predict_proba(Xte_q)[:, 1]
        else:
            score = clf.decision_function(Xte_q)
        rows.append((name, *safe_metrics(yte, score)))

    if export_path is not None:
        np.savez_compressed(export_path,
            X_train=Xtr_q.astype(np.float32), y_train=ytr.astype(np.int8),
            X_test=Xte_q.astype(np.float32), y_test=yte.astype(np.int8),
            train_pos=np.asarray(split.train_pos, dtype=str), train_neg=np.asarray(train_neg, dtype=str),
            test_pos=np.asarray(split.test_pos, dtype=str), test_neg=np.asarray(test_neg, dtype=str),
            pca_explained_variance_ratio=evr, angle_scale=qscale,
            seed=np.asarray([seed]), regime=np.asarray([regime]),
        )
    return rows, degree_diagnostics(split.train_graph, split.test_pos, test_neg), (train_pairs, ytr, test_pairs, yte, Xtr_q, Xte_q)


def filtered_ranking(H: nx.DiGraph, Gtr: nx.DiGraph, test_pos: Sequence[Pair], score_fun) -> Dict[str, float]:
    """Filtered all-target ranking: all observed full-graph edges except the query are filtered."""
    nodes = list(H.nodes()); full = set(H.edges())
    ranks = []
    for u, v in test_pos:
        candidates = [w for w in nodes if w != u and ((u, w) not in full or w == v)]
        pairs = [(u, w) for w in candidates]
        scores = np.asarray(score_fun(pairs), float)
        target = candidates.index(v)
        # Average rank across ties, descending. 1 is best.
        greater = np.sum(scores > scores[target])
        equal = np.sum(np.isclose(scores, scores[target], rtol=1e-12, atol=1e-12))
        rank = 1.0 + greater + 0.5 * max(0, equal - 1)
        ranks.append(rank)
    r = np.asarray(ranks)
    return {
        'mrr': float(np.mean(1.0 / r)),
        'hits1': float(np.mean(r <= 1)),
        'hits3': float(np.mean(r <= 3)),
        'hits10': float(np.mean(r <= 10)),
        'median_rank': float(np.median(r)),
    }


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(a.nodes); edges = pd.read_csv(a.edges)
    H = build_host_graph(nodes, edges, a.status)
    node_order = sorted(H.nodes())
    Zc = fit_text(a.host_features, node_order, a.text_dim, a.max_features, a.seed) if a.host_features else None

    rows = []; diags = []; rank_rows = []
    for r in range(a.repeats):
        seed = a.seed + r
        split = make_split(H, a.test_size, seed)
        Zg = directed_graph_embedding(split.train_graph, node_order, a.graph_dim, seed)
        for regime in a.regimes:
            if 'semantic' in regime and Zc is None:
                continue
            export_path = None
            if a.export_qml and regime == a.primary_regime:
                export_path = out / f'qml_dataset_{regime}_seed{seed}.npz'
            result, diag, payload = evaluate_one_regime(H, split, node_order, Zg, Zc, regime, seed, a.qdim, export_path)
            for model, auc, ap in result:
                rows.append({'repeat': r, 'seed': seed, 'negative_sampling': regime,
                             'model': model, 'roc_auc': auc, 'average_precision': ap})
            diags.append({'repeat': r, 'seed': seed, 'negative_sampling': regime, **diag})

            if a.ranking and regime == a.primary_regime:
                idx = {n: i for i, n in enumerate(node_order)}
                train_pairs, ytr, _, _, _, _ = payload
                Zm = np.hstack([Zg, Zc]) if Zc is not None else Zg
                Xtr_full = pair_features(Zm, train_pairs, idx)
                # Fit three interpretable ranking models on primary regime.
                lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight='balanced', random_state=seed))
                lr.fit(Xtr_full, ytr)
                def lr_score(pairs): return lr.predict_proba(pair_features(Zm, pairs, idx))[:, 1]
                def pa_score(pairs): return [split.train_graph.out_degree(u) * split.train_graph.in_degree(v) for u, v in pairs]
                for model, fun in [('preferential_attachment_directed', pa_score), ('multimodal_logreg', lr_score)]:
                    m = filtered_ranking(H, split.train_graph, split.test_pos, fun)
                    rank_rows.append({'repeat': r, 'seed': seed, 'model': model, **m})

    allres = pd.DataFrame(rows)
    allres.to_csv(out / 'degree_corrected_results_all.csv', index=False)
    summary = allres.groupby(['negative_sampling', 'model'], as_index=False).agg(
        roc_auc_mean=('roc_auc', 'mean'), roc_auc_std=('roc_auc', 'std'),
        ap_mean=('average_precision', 'mean'), ap_std=('average_precision', 'std'), n=('roc_auc', 'size'))
    summary.to_csv(out / 'degree_corrected_results_summary.csv', index=False)
    pd.DataFrame(diags).to_csv(out / 'degree_balance_diagnostics.csv', index=False)
    if rank_rows:
        rr = pd.DataFrame(rank_rows); rr.to_csv(out / 'filtered_ranking_all.csv', index=False)
        rr.groupby('model', as_index=False).agg(
            mrr_mean=('mrr','mean'), mrr_std=('mrr','std'),
            hits1_mean=('hits1','mean'), hits3_mean=('hits3','mean'), hits10_mean=('hits10','mean'),
            median_rank_mean=('median_rank','mean')).to_csv(out / 'filtered_ranking_summary.csv', index=False)

    metadata = {
        'host_nodes': H.number_of_nodes(), 'host_edges': H.number_of_edges(),
        'density': nx.density(H), 'reciprocity': float(nx.reciprocity(H) or 0.0),
        'zero_out_degree_fraction': float(np.mean([H.out_degree(n) == 0 for n in H])),
        'zero_in_degree_fraction': float(np.mean([H.in_degree(n) == 0 for n in H])),
        'primary_regime': a.primary_regime, 'repeats': a.repeats,
        'note': 'source_degree_matched is the recommended primary pair-classification regime; filtered ranking avoids sampled test negatives altogether.'
    }
    (out / 'benchmark_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
    print('\nMetadata:', json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
