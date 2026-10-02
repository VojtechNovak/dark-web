#!/usr/bin/env python3
"""
Dark-web hyperlink/HTML study pipeline for the currently available dataset.

Primary goals
-------------
1) Build the HTTP-200 largest weakly connected page component.
2) Collapse pages to onion hosts and keep cross-host directed hyperlinks.
3) Extract safe, static HTML/content features without executing scripts.
4) Quantify text-noise / boilerplate effects (including a configurable
   domain-stopword ablation such as 'porn').
5) Compare graph communities with content clusters.
6) Run self-supervised held-out link prediction with leakage-aware graph splits:
      - classical topology heuristics
      - graph representation + logistic regression
      - content representation + logistic regression
      - graph + content representation + logistic regression
   under both random and hard negative sampling.
7) Rank structural/content outliers (NOT malware labels).
8) Export a low-dimensional, fixed-split dataset ready for later VQC/QKernel work.

The script never executes HTML/JavaScript and never accesses Tor/the Internet.

Example
-------
python darkweb_ml_study.py \
  --nodes nodes.csv \
  --edges edges.csv \
  --html vybrane_html.zip \
  --out study_output \
  --domain-stopwords porn porno xxx adult \
  --repeats 5

Dependencies
------------
pandas numpy scipy networkx scikit-learn beautifulsoup4 matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
try:
    from lxml import html as lxml_html
except Exception:
    lxml_html = None
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

TOKEN_RE = re.compile(r"(?u)\b[\w-]{2,}\b")
BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{120,}={0,2}(?![A-Za-z0-9+/])")
DOWNLOAD_EXT_RE = re.compile(
    r"\.(?:exe|dll|msi|apk|zip|rar|7z|iso|dmg|pkg|deb|rpm|jar|scr|bat|cmd|ps1)(?:$|[?#])",
    flags=re.I,
)


@dataclass
class SplitData:
    train_graph: nx.DiGraph
    train_pos: List[Tuple[str, str]]
    test_pos: List[Tuple[str, str]]
    train_neg: List[Tuple[str, str]]
    test_neg_random: List[Tuple[str, str]]
    test_neg_hard: List[Tuple[str, str]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", required=True)
    p.add_argument("--edges", required=True)
    p.add_argument("--html", required=True, help="ZIP archive containing <html_id>.html")
    p.add_argument("--out", default="study_output")
    p.add_argument("--status", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-dim", type=int, default=32)
    p.add_argument("--graph-dim", type=int, default=32)
    p.add_argument("--quantum-dim", type=int, default=8)
    p.add_argument("--max-features", type=int, default=6000)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--max-df", type=float, default=0.50)
    p.add_argument(
        "--domain-stopwords",
        nargs="*",
        default=["porn"],
        help="Domain-specific terms to remove only in the ablation view. Default: porn",
    )
    return p.parse_args()


def safe_host(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    try:
        h = (urlparse(url).hostname or "").lower().strip(".")
        return h
    except Exception:
        return ""


def build_page_core(nodes: pd.DataFrame, edges: pd.DataFrame, status: int):
    if not {"id", "full_link", "status", "html_id"}.issubset(nodes.columns):
        raise ValueError("nodes.csv must contain: id, full_link, status, html_id")
    if not {"source_id", "target_id"}.issubset(edges.columns):
        raise ValueError("edges.csv must contain: source_id, target_id")

    nodes = nodes.copy()
    nodes["host"] = nodes["full_link"].map(safe_host)

    status_nodes = nodes.loc[nodes["status"] == status].copy()
    ids = set(status_nodes["id"].tolist())
    e = edges.loc[edges["source_id"].isin(ids) & edges["target_id"].isin(ids)].copy()

    G = nx.DiGraph()
    G.add_nodes_from(ids)
    G.add_edges_from(e[["source_id", "target_id"]].itertuples(index=False, name=None))

    wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    if not wccs:
        raise RuntimeError("No weakly connected component found after status filtering.")
    core_ids = set(wccs[0])
    core_nodes = status_nodes.loc[status_nodes["id"].isin(core_ids)].copy()
    core_edges = e.loc[e["source_id"].isin(core_ids) & e["target_id"].isin(core_ids)].copy()

    Gc = nx.DiGraph()
    Gc.add_nodes_from(core_ids)
    Gc.add_edges_from(core_edges[["source_id", "target_id"]].itertuples(index=False, name=None))
    return nodes, status_nodes, core_nodes, core_edges, G, Gc, [len(c) for c in wccs]


def build_host_graph(core_nodes: pd.DataFrame, core_edges: pd.DataFrame):
    id2host = core_nodes.set_index("id")["host"].to_dict()
    hosts = sorted(h for h in core_nodes["host"].dropna().unique().tolist() if h)

    pair_counts = Counter()
    same_host = 0
    unknown = 0
    for s, t in core_edges[["source_id", "target_id"]].itertuples(index=False, name=None):
        hs, ht = id2host.get(s, ""), id2host.get(t, "")
        if not hs or not ht:
            unknown += 1
            continue
        if hs == ht:
            same_host += 1
            continue
        pair_counts[(hs, ht)] += 1

    H = nx.DiGraph()
    H.add_nodes_from(hosts)
    for (u, v), w in pair_counts.items():
        H.add_edge(u, v, weight=int(w))

    host_edges = pd.DataFrame(
        [(u, v, d["weight"]) for u, v, d in H.edges(data=True)],
        columns=["source_host", "target_host", "weight"],
    )
    diagnostics = {
        "unique_hosts": H.number_of_nodes(),
        "cross_host_directed_edges": H.number_of_edges(),
        "same_host_page_edges": same_host,
        "cross_host_page_edges": int(sum(pair_counts.values())),
        "unknown_host_page_edges": unknown,
    }
    return H, host_edges, diagnostics


def _remove_tags(soup: BeautifulSoup, tags: Iterable[str]):
    for tag in soup.find_all(list(tags)):
        tag.decompose()


def parse_html_bytes(raw: bytes) -> Dict[str, object]:
    """Parse one HTML snapshot statically. lxml is used when available for speed."""
    text = raw.decode("utf-8", errors="replace")
    low = text.lower()

    if lxml_html is not None:
        try:
            parser = lxml_html.HTMLParser(recover=True, encoding="utf-8")
            tree = lxml_html.fromstring(raw, parser=parser)

            title_nodes = tree.xpath("//title//text()")
            title = " ".join(x.strip() for x in title_nodes if str(x).strip())
            hrefs = [str(x) for x in tree.xpath("//a/@href")]
            count = lambda xp: len(tree.xpath(xp))

            visible_nodes = tree.xpath(
                "//text()[not(ancestor::script) and not(ancestor::style) and "
                "not(ancestor::noscript) and not(ancestor::svg) and "
                "not(ancestor::canvas) and not(ancestor::template)]"
            )
            content_nodes = tree.xpath(
                "//text()[not(ancestor::script) and not(ancestor::style) and "
                "not(ancestor::noscript) and not(ancestor::svg) and "
                "not(ancestor::canvas) and not(ancestor::template) and "
                "not(ancestor::nav) and not(ancestor::header) and "
                "not(ancestor::footer) and not(ancestor::aside)]"
            )
            visible = " ".join(str(x).strip() for x in visible_nodes if str(x).strip())
            content = " ".join(str(x).strip() for x in content_nodes if str(x).strip())

            n_links = len(hrefs)
            n_forms = count("//form")
            n_inputs = count("//input")
            n_images = count("//img")
            n_scripts = count("//script")
            n_iframes = count("//iframe")
        except Exception:
            tree = None
    else:
        tree = None

    if tree is None:
        # Conservative fallback for unusual malformed pages.
        soup = BeautifulSoup(text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        hrefs = [a.get("href") or "" for a in soup.find_all("a")]
        n_links = len(hrefs)
        n_forms = len(soup.find_all("form"))
        n_inputs = len(soup.find_all("input"))
        n_images = len(soup.find_all("img"))
        n_scripts = len(soup.find_all("script"))
        n_iframes = len(soup.find_all("iframe"))
        _remove_tags(soup, ["script", "style", "noscript", "svg", "canvas", "template"])
        visible = " ".join(soup.stripped_strings)
        _remove_tags(soup, ["nav", "header", "footer", "aside"])
        content = " ".join(soup.stripped_strings)

    download_links = sum(bool(DOWNLOAD_EXT_RE.search(href)) for href in hrefs)
    tok = [x.lower() for x in TOKEN_RE.findall(visible)]

    return {
        "title": title,
        "html_bytes": len(raw),
        "n_links": n_links,
        "n_forms": n_forms,
        "n_inputs": n_inputs,
        "n_images": n_images,
        "n_scripts": n_scripts,
        "n_iframes": n_iframes,
        "n_download_links": int(download_links),
        "eval_hits": low.count("eval("),
        "atob_hits": low.count("atob("),
        "fromcharcode_hits": low.count("fromcharcode"),
        "document_write_hits": low.count("document.write"),
        "base64_chunks": len(BASE64_RE.findall(text)),
        "visible_text": visible,
        "content_text": content,
        "text_chars": len(visible),
        "text_words": len(tok),
        "porn_count": tok.count("porn"),
        "porn_present": int("porn" in tok),
    }


def extract_html_features(zip_path: str, core_nodes: pd.DataFrame) -> pd.DataFrame:
    wanted = core_nodes[["id", "html_id", "host"]].dropna(subset=["html_id"]).copy()
    wanted["html_id"] = wanted["html_id"].astype(str)
    records = []

    with zipfile.ZipFile(zip_path) as zf:
        lookup = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith(".html")}
        for row in wanted.itertuples(index=False):
            zname = lookup.get(row.html_id)
            if zname is None:
                records.append({"id": row.id, "host": row.host, "html_id": row.html_id, "html_present": False})
                continue
            try:
                d = parse_html_bytes(zf.read(zname))
                d.update({"id": row.id, "host": row.host, "html_id": row.html_id, "html_present": True})
                records.append(d)
            except Exception as exc:
                records.append(
                    {
                        "id": row.id,
                        "host": row.host,
                        "html_id": row.html_id,
                        "html_present": False,
                        "parse_error": repr(exc),
                    }
                )
    return pd.DataFrame(records)


def aggregate_host_content(html_df: pd.DataFrame, core_nodes: pd.DataFrame) -> pd.DataFrame:
    # Include hosts even if parsing failed.
    counts = core_nodes.groupby("host").size().rename("page_count").reset_index()
    numeric_cols = [
        "html_bytes", "n_links", "n_forms", "n_inputs", "n_images", "n_scripts", "n_iframes",
        "n_download_links", "eval_hits", "atob_hits", "fromcharcode_hits", "document_write_hits",
        "base64_chunks", "text_chars", "text_words", "porn_count", "porn_present",
    ]
    for c in numeric_cols:
        if c not in html_df.columns:
            html_df[c] = 0
        html_df[c] = pd.to_numeric(html_df[c], errors="coerce").fillna(0)

    agg_num = html_df.groupby("host")[numeric_cols].sum().reset_index()
    agg_num = agg_num.rename(columns={"porn_present": "pages_with_porn"})
    html_count = html_df.groupby("host")["html_present"].sum().rename("html_count").reset_index()

    text_agg = (
        html_df.groupby("host")
        .agg(
            visible_text=("visible_text", lambda s: "\n".join(x for x in s.dropna().astype(str) if x)),
            content_text=("content_text", lambda s: "\n".join(x for x in s.dropna().astype(str) if x)),
        )
        .reset_index()
    )

    out = counts.merge(agg_num, on="host", how="left").merge(html_count, on="host", how="left").merge(text_agg, on="host", how="left")
    out = out.fillna({"visible_text": "", "content_text": "", "html_count": 0})
    return out


def graph_metrics(H: nx.DiGraph) -> pd.DataFrame:
    U = H.to_undirected()
    if U.number_of_edges() > 0:
        communities = nx.community.louvain_communities(U, seed=42, weight="weight")
    else:
        communities = [{n} for n in U.nodes()]
    cmap = {n: i for i, c in enumerate(communities) for n in c}

    pagerank = nx.pagerank(H, weight="weight") if H.number_of_edges() else {n: 0.0 for n in H.nodes()}
    btw = nx.betweenness_centrality(U, weight=None, normalized=True) if U.number_of_nodes() <= 3000 else {n: np.nan for n in U.nodes()}
    clustering = nx.clustering(U)
    core = nx.core_number(U) if U.number_of_edges() else {n: 0 for n in U.nodes()}

    rows = []
    for n in H.nodes():
        rows.append(
            {
                "host": n,
                "in_degree": H.in_degree(n),
                "out_degree": H.out_degree(n),
                "total_degree": H.in_degree(n) + H.out_degree(n),
                "pagerank": pagerank.get(n, 0.0),
                "betweenness": btw.get(n, 0.0),
                "clustering_undirected": clustering.get(n, 0.0),
                "core_number": core.get(n, 0),
                "community_louvain": cmap.get(n, -1),
            }
        )
    return pd.DataFrame(rows)


def fit_text_view(
    texts: Sequence[str],
    dim: int,
    max_features: int,
    min_df: int,
    max_df: float,
    stopwords=None,
    seed: int = 42,
):
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[\w-]{2,}\b",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
        stop_words=stopwords,
        norm="l2",
    )
    X = vec.fit_transform(texts)
    max_dim = max(1, min(dim, X.shape[0] - 1, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=max_dim, random_state=seed)
    Z = svd.fit_transform(X)
    return vec, X, Z, svd


def top_terms(vec: TfidfVectorizer, X, n=40):
    mean = np.asarray(X.mean(axis=0)).ravel()
    names = np.asarray(vec.get_feature_names_out())
    idx = np.argsort(mean)[::-1][:n]
    return pd.DataFrame({"term": names[idx], "mean_tfidf": mean[idx]})


def content_graph_agreement(host_df: pd.DataFrame, Z: np.ndarray, seed: int = 42):
    graph_labels = host_df["community_louvain"].to_numpy()
    k_graph = max(2, len(np.unique(graph_labels)))

    rows = []
    best = None
    max_k = min(12, max(2, len(host_df) - 1))
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        pred = km.fit_predict(Z)
        sil = silhouette_score(Z, pred) if len(np.unique(pred)) > 1 else np.nan
        row = {"k": k, "silhouette": sil}
        if k == k_graph:
            row.update(
                {
                    "nmi_vs_louvain": normalized_mutual_info_score(graph_labels, pred),
                    "ami_vs_louvain": adjusted_mutual_info_score(graph_labels, pred),
                    "ari_vs_louvain": adjusted_rand_score(graph_labels, pred),
                }
            )
        rows.append(row)
        if best is None or (not np.isnan(sil) and sil > best[0]):
            best = (sil, k, pred)
    return pd.DataFrame(rows), best


def make_link_split(H: nx.DiGraph, test_size: float, seed: int) -> SplitData:
    rng = np.random.default_rng(seed)
    nodes = list(H.nodes())
    full_edges = set(H.edges())
    if len(full_edges) < 10:
        raise RuntimeError("Host graph is too small for link prediction.")

    U = H.to_undirected()
    # Protect one spanning tree so the train graph retains weak connectivity.
    tree = nx.minimum_spanning_tree(U)
    protected_dyads = {tuple(sorted(e)) for e in tree.edges()}

    dyad_to_directed = defaultdict(list)
    for u, v in full_edges:
        dyad_to_directed[tuple(sorted((u, v)))].append((u, v))

    candidate_dyads = [d for d in dyad_to_directed if d not in protected_dyads]
    rng.shuffle(candidate_dyads)
    n_test_dyads = max(1, int(round(test_size * len(candidate_dyads))))
    test_dyads = set(candidate_dyads[:n_test_dyads])
    test_pos = [e for d in test_dyads for e in dyad_to_directed[d]]
    train_pos = [e for e in full_edges if tuple(sorted(e)) not in test_dyads]

    Gtr = nx.DiGraph()
    Gtr.add_nodes_from(nodes)
    Gtr.add_edges_from(train_pos)

    def sample_random_neg(n: int, forbidden_extra=None):
        out = []
        used = set(forbidden_extra or [])
        max_tries = max(10000, n * 100)
        tries = 0
        while len(out) < n and tries < max_tries:
            tries += 1
            u = nodes[int(rng.integers(0, len(nodes)))]
            v = nodes[int(rng.integers(0, len(nodes)))]
            if u == v or (u, v) in full_edges or (u, v) in used:
                continue
            used.add((u, v))
            out.append((u, v))
        if len(out) < n:
            raise RuntimeError(f"Could only sample {len(out)}/{n} random negatives")
        return out

    # Hard negatives: absent edges whose endpoints are close/same-community in train topology.
    Utr = Gtr.to_undirected()
    comms = nx.community.louvain_communities(Utr, seed=seed) if Utr.number_of_edges() else [{n} for n in nodes]
    cmap = {n: i for i, c in enumerate(comms) for n in c}
    hard_pool = []
    for u in nodes:
        # two-hop local candidates are intentionally difficult
        two_hop = set()
        for nbr in Utr.neighbors(u):
            two_hop.update(Utr.neighbors(nbr))
        two_hop.discard(u)
        for v in two_hop:
            if (u, v) not in full_edges:
                hard_pool.append((u, v))
    # Add same-community absent pairs if two-hop pool is too small.
    if len(hard_pool) < len(test_pos) * 2:
        for u in nodes:
            for v in nodes:
                if u != v and cmap.get(u) == cmap.get(v) and (u, v) not in full_edges:
                    hard_pool.append((u, v))
    hard_pool = list(dict.fromkeys(hard_pool))
    rng.shuffle(hard_pool)
    test_neg_hard = hard_pool[: len(test_pos)]
    if len(test_neg_hard) < len(test_pos):
        supplement = sample_random_neg(len(test_pos) - len(test_neg_hard), forbidden_extra=test_neg_hard)
        test_neg_hard += supplement

    test_neg_random = sample_random_neg(len(test_pos), forbidden_extra=test_neg_hard)

    # Mixed train negatives: half hard, half random.
    n_train = len(train_pos)
    nh = min(len(hard_pool) - len(test_neg_hard), n_train // 2)
    hard_for_train = hard_pool[len(test_neg_hard): len(test_neg_hard) + max(0, nh)]
    train_neg = list(hard_for_train)
    train_neg += sample_random_neg(n_train - len(train_neg), forbidden_extra=train_neg + test_neg_hard + test_neg_random)

    return SplitData(Gtr, train_pos, test_pos, train_neg, test_neg_random, test_neg_hard)


def directed_graph_embedding(G: nx.DiGraph, node_order: Sequence[str], dim: int, seed: int):
    idx = {n: i for i, n in enumerate(node_order)}
    rows, cols = [], []
    for u, v in G.edges():
        rows.append(idx[u])
        cols.append(idx[v])
    data = np.ones(len(rows), dtype=float)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(len(node_order), len(node_order)))
    max_dim = max(1, min(dim, len(node_order) - 2))
    svd_out = TruncatedSVD(n_components=max_dim, random_state=seed)
    svd_in = TruncatedSVD(n_components=max_dim, random_state=seed + 1)
    Zout = svd_out.fit_transform(A)
    Zin = svd_in.fit_transform(A.T)

    # Add simple train-only structural features.
    pr = nx.pagerank(G) if G.number_of_edges() else {n: 0.0 for n in node_order}
    deg = np.array(
        [[G.in_degree(n), G.out_degree(n), math.log1p(G.in_degree(n)), math.log1p(G.out_degree(n)), pr.get(n, 0.0)] for n in node_order],
        dtype=float,
    )
    deg = StandardScaler().fit_transform(deg)
    return np.hstack([Zout, Zin, deg])


def pair_features(Z: np.ndarray, pairs: Sequence[Tuple[str, str]], index: Dict[str, int]) -> np.ndarray:
    a = np.vstack([Z[index[u]] for u, _ in pairs])
    b = np.vstack([Z[index[v]] for _, v in pairs])
    return np.hstack([a, b, a * b, np.abs(a - b)])


def heuristic_scores(Gtr: nx.DiGraph, pairs: Sequence[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    U = Gtr.to_undirected()
    cn, jac, aa, pref = [], [], [], []
    for u, v in pairs:
        nu, nv = set(U.neighbors(u)), set(U.neighbors(v))
        inter = nu & nv
        union = nu | nv
        cn.append(float(len(inter)))
        jac.append(float(len(inter) / len(union)) if union else 0.0)
        aa.append(sum(1.0 / math.log(U.degree(w)) for w in inter if U.degree(w) > 1))
        pref.append(float(Gtr.out_degree(u) * Gtr.in_degree(v)))
    return {
        "common_neighbors": np.asarray(cn),
        "jaccard": np.asarray(jac),
        "adamic_adar": np.asarray(aa),
        "preferential_attachment_directed": np.asarray(pref),
    }


def safe_auc_ap(y: np.ndarray, score: np.ndarray):
    try:
        auc = roc_auc_score(y, score)
    except Exception:
        auc = np.nan
    try:
        ap = average_precision_score(y, score)
    except Exception:
        ap = np.nan
    return auc, ap


def semantic_hard_negatives(
    H: nx.DiGraph,
    host_order: Sequence[str],
    Z_content: np.ndarray,
    n: int,
    seed: int,
    excluded: Sequence[Tuple[str, str]] = (),
) -> List[Tuple[str, str]]:
    """High-content-similarity host pairs with no observed edge in either direction."""
    rng = np.random.default_rng(seed)
    full_edges = set(H.edges())
    excluded = set(excluded)
    Z = np.asarray(Z_content, dtype=float)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / np.maximum(norms, 1e-12)
    sim = Zn @ Zn.T

    candidates = []
    m = len(host_order)
    for i in range(m):
        for j in range(i + 1, m):
            u, v = host_order[i], host_order[j]
            if (u, v) in full_edges or (v, u) in full_edges:
                continue
            # Store both possible directions; direction itself is not used to define hardness.
            candidates.append((float(sim[i, j]), u, v))
            candidates.append((float(sim[i, j]), v, u))

    # Break exact ties reproducibly, then rank by semantic similarity.
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: x[0], reverse=True)

    out = []
    per_source = Counter()
    source_cap = max(3, int(math.ceil(n / max(1, len(host_order)) * 8)))
    for _, u, v in candidates:
        if (u, v) in excluded:
            continue
        if per_source[u] >= source_cap:
            continue
        out.append((u, v))
        per_source[u] += 1
        if len(out) >= n:
            break
    return out


def evaluate_link_prediction(
    H: nx.DiGraph,
    host_order: Sequence[str],
    content_views: Dict[str, np.ndarray],
    repeats: int,
    test_size: float,
    graph_dim: int,
    seed: int,
    qdim: int,
    out_dir: Path,
):
    index = {h: i for i, h in enumerate(host_order)}
    rows = []
    first_quantum_payload = None
    if "boilerplate_domain_filtered" not in content_views:
        raise ValueError("content_views must include boilerplate_domain_filtered")
    Z_content_main = content_views["boilerplate_domain_filtered"]

    for r in range(repeats):
        split_seed = seed + r
        split = make_link_split(H, test_size=test_size, seed=split_seed)
        Z_graph = directed_graph_embedding(split.train_graph, host_order, graph_dim, split_seed)
        Z_multi = np.hstack([Z_graph, Z_content_main])

        semantic_neg = semantic_hard_negatives(
            H, host_order, Z_content_main, len(split.test_pos), split_seed,
            excluded=split.test_neg_random + split.test_neg_hard,
        )
        if len(semantic_neg) < len(split.test_pos):
            # Extremely unlikely for this dataset; retain a balanced test if it occurs.
            semantic_neg = split.test_neg_hard[: len(split.test_pos)]

        train_pairs = split.train_pos + split.train_neg
        y_train = np.r_[np.ones(len(split.train_pos)), np.zeros(len(split.train_neg))]

        for neg_name, neg_pairs in [
            ("random", split.test_neg_random),
            ("topology_hard", split.test_neg_hard),
            ("semantic_hard", semantic_neg),
        ]:
            test_pairs = split.test_pos + neg_pairs
            y_test = np.r_[np.ones(len(split.test_pos)), np.zeros(len(neg_pairs))]

            # Pure topology heuristics.
            hs = heuristic_scores(split.train_graph, test_pairs)
            for name, score in hs.items():
                auc, ap = safe_auc_ap(y_test, score)
                rows.append(
                    {
                        "repeat": r,
                        "seed": split_seed,
                        "negative_sampling": neg_name,
                        "model": name,
                        "roc_auc": auc,
                        "average_precision": ap,
                    }
                )

            # Learned pair models. Content views are kept separate to make
            # boilerplate/domain-stopword effects measurable rather than assumed.
            learned = [("logreg_graph", Z_graph)]
            for view_name, Zc in content_views.items():
                learned.append((f"logreg_content_{view_name}", Zc))
            learned.append(("logreg_multimodal_domain_filtered", Z_multi))

            for model_name, Z in learned:
                Xtr = pair_features(Z, train_pairs, index)
                Xte = pair_features(Z, test_pairs, index)
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=4000, class_weight="balanced", random_state=split_seed),
                )
                clf.fit(Xtr, y_train)
                prob = clf.predict_proba(Xte)[:, 1]
                auc, ap = safe_auc_ap(y_test, prob)
                rows.append(
                    {
                        "repeat": r,
                        "seed": split_seed,
                        "negative_sampling": neg_name,
                        "model": model_name,
                        "roc_auc": auc,
                        "average_precision": ap,
                    }
                )

        # Save one fixed split in low dimension for later quantum experiments.
        if r == 0:
            Xtr_full = pair_features(Z_multi, train_pairs, index)
            rand_pairs = split.test_pos + split.test_neg_random
            hard_pairs = split.test_pos + split.test_neg_hard
            sem_pairs = split.test_pos + semantic_neg
            Xrand_full = pair_features(Z_multi, rand_pairs, index)
            Xhard_full = pair_features(Z_multi, hard_pairs, index)
            Xsem_full = pair_features(Z_multi, sem_pairs, index)
            yrand = np.r_[np.ones(len(split.test_pos)), np.zeros(len(split.test_neg_random))]
            yhard = np.r_[np.ones(len(split.test_pos)), np.zeros(len(split.test_neg_hard))]
            ysem = np.r_[np.ones(len(split.test_pos)), np.zeros(len(semantic_neg))]

            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr_full)
            Xrand_s = scaler.transform(Xrand_full)
            Xhard_s = scaler.transform(Xhard_full)
            Xsem_s = scaler.transform(Xsem_full)
            q = max(2, min(qdim, Xtr_s.shape[1], Xtr_s.shape[0] - 1))
            pca = PCA(n_components=q, random_state=split_seed)
            Xtr_q = pca.fit_transform(Xtr_s)
            Xrand_q = pca.transform(Xrand_s)
            Xhard_q = pca.transform(Xhard_s)
            Xsem_q = pca.transform(Xsem_s)

            # Map roughly into angle-friendly scale [-pi, pi] using train statistics only.
            scale = np.maximum(np.max(np.abs(Xtr_q), axis=0), 1e-9)
            Xtr_q = np.clip(Xtr_q / scale, -1, 1) * np.pi
            Xrand_q = np.clip(Xrand_q / scale, -1, 1) * np.pi
            Xhard_q = np.clip(Xhard_q / scale, -1, 1) * np.pi
            Xsem_q = np.clip(Xsem_q / scale, -1, 1) * np.pi

            first_quantum_payload = {
                "X_train": Xtr_q,
                "y_train": y_train.astype(int),
                "X_test_random": Xrand_q,
                "y_test_random": yrand.astype(int),
                "X_test_hard": Xhard_q,
                "y_test_hard": yhard.astype(int),
                "X_test_semantic_hard": Xsem_q,
                "y_test_semantic_hard": ysem.astype(int),
                "seed": np.asarray([split_seed]),
                "pca_explained_variance_ratio": pca.explained_variance_ratio_,
            }

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "link_prediction_results_all.csv", index=False)

    summary = (
        results.groupby(["negative_sampling", "model"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            ap_mean=("average_precision", "mean"),
            ap_std=("average_precision", "std"),
        )
        .sort_values(["negative_sampling", "roc_auc_mean"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "link_prediction_results_summary.csv", index=False)

    if first_quantum_payload is not None:
        np.savez_compressed(out_dir / "qml_link_dataset_seed42.npz", **first_quantum_payload)

    return results, summary


def outlier_analysis(host_table: pd.DataFrame, seed: int) -> pd.DataFrame:
    feature_cols = [
        "in_degree", "out_degree", "pagerank", "betweenness", "clustering_undirected", "core_number",
        "page_count", "html_count", "html_bytes", "text_words", "n_links", "n_forms", "n_inputs",
        "n_scripts", "n_iframes", "n_download_links", "eval_hits", "atob_hits", "fromcharcode_hits",
        "document_write_hits", "base64_chunks", "porn_count", "pages_with_porn",
    ]
    cols = [c for c in feature_cols if c in host_table.columns]
    X = host_table[cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    X = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=500, contamination="auto", random_state=seed)
    iso.fit(X)
    out = host_table.copy()
    out["outlier_score"] = -iso.score_samples(X)  # larger = more unusual
    out = out.sort_values("outlier_score", ascending=False)
    return out


def plot_link_summary(summary: pd.DataFrame, out_dir: Path):
    for neg in summary["negative_sampling"].unique():
        d = summary.loc[summary["negative_sampling"] == neg].sort_values("roc_auc_mean", ascending=False)
        plt.figure(figsize=(10, 5))
        plt.bar(d["model"], d["roc_auc_mean"], yerr=d["roc_auc_std"].fillna(0), capsize=3)
        plt.ylim(0.0, 1.0)
        plt.ylabel("ROC-AUC")
        plt.title(f"Held-out cross-host link prediction ({neg} negatives)")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"link_prediction_{neg}_roc_auc.png", dpi=180)
        plt.close()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)

    nodes_all, status_nodes, core_nodes, core_edges, Gstatus, Gcore, wcc_sizes = build_page_core(nodes, edges, args.status)
    H, host_edges, hdiag = build_host_graph(core_nodes, core_edges)
    host_edges.to_csv(out_dir / "host_edges_cross_host_weighted.csv", index=False)

    html_df = extract_html_features(args.html, core_nodes)
    html_df.to_csv(out_dir / "page_html_content_features.csv", index=False)
    host_content = aggregate_host_content(html_df, core_nodes)

    hmetrics = graph_metrics(H)
    host_table = hmetrics.merge(host_content, on="host", how="left")
    host_table.to_csv(out_dir / "host_features_full.csv", index=False)

    # Text-noise prevalence.
    porn_pages = int((html_df.get("porn_present", pd.Series(dtype=float)).fillna(0) > 0).sum())
    parsed_pages = int(html_df.get("html_present", pd.Series(dtype=bool)).fillna(False).sum())

    # Three content views for ablation.
    host_order = host_table["host"].tolist()
    content_map = host_table.set_index("host")["content_text"].fillna("").to_dict()
    visible_map = host_table.set_index("host")["visible_text"].fillna("").to_dict()
    texts_raw = [visible_map.get(h, "") for h in host_order]
    texts_clean = [content_map.get(h, "") for h in host_order]

    domain_sw = set(str(x).lower() for x in args.domain_stopwords)
    clean_sw = set(ENGLISH_STOP_WORDS)
    domain_plus = sorted(clean_sw | domain_sw)

    text_views = {}
    configs = [
        ("raw_visible", texts_raw, None),
        ("boilerplate_clean", texts_clean, sorted(clean_sw)),
        ("boilerplate_domain_filtered", texts_clean, domain_plus),
    ]
    agreement_rows = []
    for name, texts, sw in configs:
        try:
            vec, X, Z, svd = fit_text_view(
                texts,
                dim=args.text_dim,
                max_features=args.max_features,
                min_df=args.min_df,
                max_df=args.max_df,
                stopwords=sw,
                seed=args.seed,
            )
        except ValueError:
            # Fallback for very small corpora / strict df settings.
            vec, X, Z, svd = fit_text_view(
                texts,
                dim=args.text_dim,
                max_features=args.max_features,
                min_df=1,
                max_df=1.0,
                stopwords=sw,
                seed=args.seed,
            )
        text_views[name] = (vec, X, Z, svd)
        top_terms(vec, X).to_csv(out_dir / f"top_terms_{name}.csv", index=False)
        agree, best = content_graph_agreement(host_table, Z, seed=args.seed)
        agree.insert(0, "text_view", name)
        agreement_rows.append(agree)

    agreement = pd.concat(agreement_rows, ignore_index=True)
    agreement.to_csv(out_dir / "content_vs_graph_community_agreement.csv", index=False)

    content_embeddings = {name: bundle[2] for name, bundle in text_views.items()}

    link_all, link_summary = evaluate_link_prediction(
        H=H,
        host_order=host_order,
        content_views=content_embeddings,
        repeats=args.repeats,
        test_size=args.test_size,
        graph_dim=args.graph_dim,
        seed=args.seed,
        qdim=args.quantum_dim,
        out_dir=out_dir,
    )
    plot_link_summary(link_summary, out_dir)

    outliers = outlier_analysis(host_table, seed=args.seed)
    outliers.to_csv(out_dir / "host_outlier_ranking.csv", index=False)

    risk_cols = [
        "id", "host", "title", "n_iframes", "n_download_links", "eval_hits", "atob_hits",
        "fromcharcode_hits", "document_write_hits", "base64_chunks", "porn_count", "text_words",
    ]
    risk_cols = [c for c in risk_cols if c in html_df.columns]
    risk = html_df[risk_cols].copy()
    score_cols = [c for c in ["n_iframes", "n_download_links", "eval_hits", "atob_hits", "fromcharcode_hits", "document_write_hits", "base64_chunks"] if c in risk]
    if score_cols:
        risk["static_indicator_sum"] = risk[score_cols].sum(axis=1)
        risk = risk.sort_values(["static_indicator_sum", "n_download_links"], ascending=False)
    risk.to_csv(out_dir / "html_static_indicator_ranking_NOT_malware_labels.csv", index=False)

    summary = {
        "status_requested": args.status,
        "status_nodes": int(len(status_nodes)),
        "status_wcc_sizes": [int(x) for x in wcc_sizes],
        "core_page_nodes": int(Gcore.number_of_nodes()),
        "core_page_edges": int(Gcore.number_of_edges()),
        "host_nodes": int(H.number_of_nodes()),
        "host_cross_edges": int(H.number_of_edges()),
        "host_graph_weak_components": int(nx.number_weakly_connected_components(H)) if H.number_of_nodes() else 0,
        "host_graph_reciprocity": float(nx.reciprocity(H) or 0.0) if H.number_of_edges() else 0.0,
        "same_host_page_edges": int(hdiag["same_host_page_edges"]),
        "cross_host_page_edges": int(hdiag["cross_host_page_edges"]),
        "html_parsed_pages": parsed_pages,
        "pages_with_exact_token_porn": porn_pages,
        "porn_page_fraction": float(porn_pages / parsed_pages) if parsed_pages else None,
        "domain_stopwords": sorted(domain_sw),
        "note": "Static HTML indicators and IsolationForest scores are outlier/risk indicators only; they are not malware labels.",
    }
    with open(out_dir / "study_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== STUDY SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== LINK PREDICTION SUMMARY ===")
    print(link_summary.to_string(index=False))
    print(f"\nOutputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
