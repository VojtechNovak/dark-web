#!/usr/bin/env python3
"""
Semantic/content analysis of a dark-web host graph.

Purpose
-------
This script answers questions that link prediction does not:
  * What kinds of content appear in the crawled Tor component?
  * Which hosts/pages form semantic clusters?
  * How do those semantic clusters relate to graph communities?
  * Which semantic groups link to which other groups?
  * Which pages best represent each cluster for human/LLM interpretation?

The recommended workflow is:
  HTML -> cleaned host text -> multilingual embeddings -> clustering ->
  representative pages/top terms -> cluster-to-cluster hyperlink analysis ->
  human/LLM-assisted cluster labels.

The LLM is intentionally used for *interpretation/labeling*, not for inventing
cluster membership. Cluster membership is produced by a reproducible numerical
pipeline. The generated ``cluster_dossiers.json`` and
``llm_cluster_labeling_prompt.md`` can be supplied to an LLM for evidence-based
labels.

Inputs (defaults match the existing study outputs)
---------------------------------------------------
  host_features_full.csv
  host_edges_cross_host_weighted.csv
  page_html_content_features.csv

Optional embedding backends
---------------------------
1) ``multilingual`` (recommended): sentence-transformers
   paraphrase-multilingual-MiniLM-L12-v2. This handles German/English/etc. in
   one semantic space. Install ``sentence-transformers`` once; model download
   may require Internet access on first use.
2) ``tfidf``: dependency-light fallback combining word and character TF-IDF
   followed by SVD. Useful offline, but less reliable across languages.
3) ``auto``: multilingual if available, otherwise TF-IDF fallback.

No HTML/JavaScript is executed and no network/Tor access is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer


SPACE_RE = re.compile(r"\s+")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hosts", default="host_features_full.csv")
    p.add_argument("--edges", default="host_edges_cross_host_weighted.csv")
    p.add_argument("--pages", default="page_html_content_features.csv")
    p.add_argument("--out", default="semantic_cluster_output")
    p.add_argument("--embedding", choices=["auto", "multilingual", "tfidf"], default="auto")
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--k", default="auto", help="Integer cluster count or 'auto' (default).")
    p.add_argument("--k-min", type=int, default=4)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--representatives", type=int, default=5)
    p.add_argument("--snippet-chars", type=int, default=700)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--top-terms", type=int, default=15)
    p.add_argument("--show-onion-hosts", action="store_true", help="Include raw onion host names in reports.")
    return p.parse_args()


def clean_text(x) -> str:
    if not isinstance(x, str):
        return ""
    return SPACE_RE.sub(" ", x).strip()


def safe_read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing input: {p}")
    return pd.read_csv(p)


def host_alias(host: str) -> str:
    return "H_" + hashlib.sha256(str(host).encode("utf-8")).hexdigest()[:10]


def embed_multilingual(texts: list[str], model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    Z = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(Z, dtype=np.float32), {"backend": "multilingual", "model": model_name}


def embed_tfidf(texts: list[str], dim: int, seed: int):
    # Character n-grams are intentionally included because they are more robust
    # than word-only TF-IDF to spelling variation and partially reduce the damage
    # from a multilingual corpus. This is still not equivalent to a true
    # multilingual semantic encoder.
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.70,
            max_features=12000,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=12000,
            sublinear_tf=True,
        )),
    ])
    X = features.fit_transform(texts)
    ncomp = max(2, min(dim, X.shape[0] - 1, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=ncomp, random_state=seed)
    Z = svd.fit_transform(X)
    Z = Normalizer(copy=False).fit_transform(Z)
    return np.asarray(Z, dtype=np.float32), {
        "backend": "tfidf_word_char_svd",
        "svd_dim": int(ncomp),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
    }


def build_embeddings(texts: list[str], mode: str, model: str, dim: int, seed: int):
    if mode in {"auto", "multilingual"}:
        try:
            return embed_multilingual(texts, model)
        except Exception as exc:
            if mode == "multilingual":
                raise
            print(f"[INFO] multilingual embedding unavailable ({type(exc).__name__}: {exc})")
            print("[INFO] falling back to word+character TF-IDF/SVD")
    return embed_tfidf(texts, dim, seed)


def choose_k(Z: np.ndarray, kmin: int, kmax: int, seed: int):
    rows = []
    best = None
    upper = min(kmax, len(Z) - 1)
    for k in range(max(2, kmin), upper + 1):
        km = KMeans(n_clusters=k, n_init=30, random_state=seed)
        labels = km.fit_predict(Z)
        sil = float(silhouette_score(Z, labels, metric="cosine"))
        rows.append({"k": k, "silhouette_cosine": sil})
        if best is None or sil > best[0]:
            best = (sil, k)
    if best is None:
        raise RuntimeError("Could not choose a valid cluster count")
    return best[1], pd.DataFrame(rows)


def fit_clusters(Z: np.ndarray, k: int, seed: int):
    km = KMeans(n_clusters=k, n_init=50, random_state=seed)
    labels = km.fit_predict(Z)
    return labels, km.cluster_centers_


def cluster_top_terms(texts: list[str], labels: np.ndarray, n_terms: int):
    # Cluster-level TF-IDF (c-TF-IDF style): one document per cluster.
    k = int(np.max(labels)) + 1
    cluster_docs = []
    for c in range(k):
        cluster_docs.append(" ".join(texts[i] for i in np.flatnonzero(labels == c)))
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        max_features=20000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(cluster_docs)
    terms = np.asarray(vec.get_feature_names_out())
    out = {}
    for c in range(k):
        row = X.getrow(c).toarray().ravel()
        idx = np.argsort(row)[::-1][:n_terms]
        out[c] = [(str(terms[j]), float(row[j])) for j in idx if row[j] > 0]
    return out


def representative_hosts(Z, labels, centroids, n):
    reps = {}
    for c in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == c)
        sims = cosine_similarity(Z[idx], centroids[c].reshape(1, -1)).ravel()
        order = idx[np.argsort(sims)[::-1][:n]]
        reps[c] = [(int(i), float(cosine_similarity(Z[i:i+1], centroids[c].reshape(1, -1))[0, 0])) for i in order]
    return reps


def page_representative(page_df: pd.DataFrame, host: str, snippet_chars: int):
    d = page_df.loc[page_df["host"].astype(str) == str(host)].copy()
    if d.empty:
        return {"title": "", "snippet": ""}
    if "text_words" in d.columns:
        d = d.sort_values("text_words", ascending=False)
    r = d.iloc[0]
    title = clean_text(r.get("title", ""))[:300]
    text = clean_text(r.get("content_text", "") or r.get("visible_text", ""))
    return {"title": title, "snippet": text[:snippet_chars]}


def cluster_graph_stats(edge_df: pd.DataFrame, host_to_cluster: dict[str, int]):
    rows = []
    for r in edge_df.itertuples(index=False):
        u = str(getattr(r, "source_host"))
        v = str(getattr(r, "target_host"))
        if u not in host_to_cluster or v not in host_to_cluster:
            continue
        w = float(getattr(r, "weight", 1.0))
        rows.append((host_to_cluster[u], host_to_cluster[v], w))
    if not rows:
        return pd.DataFrame(columns=["source_cluster", "target_cluster", "edge_pairs", "weight"])
    d = pd.DataFrame(rows, columns=["source_cluster", "target_cluster", "weight"])
    out = d.groupby(["source_cluster", "target_cluster"], as_index=False).agg(
        edge_pairs=("weight", "size"),
        weight=("weight", "sum"),
    )
    return out


def top_links_for_cluster(c: int, cluster_edges: pd.DataFrame, direction: str, topn: int = 5):
    if direction == "out":
        d = cluster_edges.loc[cluster_edges.source_cluster == c].copy()
        key = "target_cluster"
    else:
        d = cluster_edges.loc[cluster_edges.target_cluster == c].copy()
        key = "source_cluster"
    if d.empty:
        return []
    d = d.sort_values(["weight", "edge_pairs"], ascending=False).head(topn)
    total = max(float(d["weight"].sum()), 1.0)
    return [
        {
            "cluster": int(getattr(r, key)),
            "edge_pairs": int(r.edge_pairs),
            "weight": float(r.weight),
        }
        for r in d.itertuples(index=False)
    ]


def make_reports(host_df, page_df, labels, Z, centroids, topterms, cluster_edges, args, embed_meta, k_scores):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    hosts = host_df["host"].astype(str).tolist()
    aliases = {h: host_alias(h) for h in hosts}
    reps = representative_hosts(Z, labels, centroids, args.representatives)

    graph_labels = host_df.get("community_louvain")
    agreement = None
    if graph_labels is not None and graph_labels.notna().all():
        gl = graph_labels.astype(int).to_numpy()
        agreement = {
            "nmi_vs_louvain": float(normalized_mutual_info_score(gl, labels)),
            "ami_vs_louvain": float(adjusted_mutual_info_score(gl, labels)),
            "ari_vs_louvain": float(adjusted_rand_score(gl, labels)),
        }

    dossiers = []
    md = [
        "# Semantic cluster report",
        "",
        f"Embedding backend: `{embed_meta.get('backend')}`",
        f"Number of hosts: **{len(host_df)}**",
        f"Number of semantic clusters: **{len(set(labels.tolist()))}**",
    ]
    if agreement:
        md += [
            "",
            "## Content–topology agreement",
            "",
            f"NMI vs Louvain: **{agreement['nmi_vs_louvain']:.3f}**  ",
            f"AMI vs Louvain: **{agreement['ami_vs_louvain']:.3f}**  ",
            f"ARI vs Louvain: **{agreement['ari_vs_louvain']:.3f}**",
        ]

    for c in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == c)
        internal = cluster_edges.loc[(cluster_edges.source_cluster == c) & (cluster_edges.target_cluster == c)]
        outgoing = cluster_edges.loc[cluster_edges.source_cluster == c]
        internal_weight = float(internal.weight.sum()) if not internal.empty else 0.0
        outgoing_weight = float(outgoing.weight.sum()) if not outgoing.empty else 0.0
        internal_frac = internal_weight / outgoing_weight if outgoing_weight else 0.0

        reps_out = []
        for i, sim in reps[c]:
            h = hosts[i]
            pg = page_representative(page_df, h, args.snippet_chars)
            item = {
                "host_id": h if args.show_onion_hosts else aliases[h],
                "centroid_similarity": round(sim, 4),
                "title": pg["title"],
                "snippet": pg["snippet"],
                "page_count": int(host_df.iloc[i].get("page_count", 0)),
                "in_degree": int(host_df.iloc[i].get("in_degree", 0)),
                "out_degree": int(host_df.iloc[i].get("out_degree", 0)),
                "pagerank": float(host_df.iloc[i].get("pagerank", 0.0)),
            }
            reps_out.append(item)

        dossier = {
            "cluster_id": int(c),
            "n_hosts": int(len(idx)),
            "top_terms": [x[0] for x in topterms[c]],
            "representatives": reps_out,
            "internal_outgoing_weight_fraction": round(internal_frac, 4),
            "top_outgoing_clusters": top_links_for_cluster(c, cluster_edges, "out"),
            "top_incoming_clusters": top_links_for_cluster(c, cluster_edges, "in"),
        }
        dossiers.append(dossier)

        md += [
            "",
            f"## Cluster C{c} — {len(idx)} hosts",
            "",
            "**Top terms:** " + ", ".join(dossier["top_terms"]),
            "",
            f"**Fraction of outgoing hyperlink weight staying inside cluster:** {internal_frac:.1%}",
            "",
            "**Main outgoing cluster connections:** " + (
                ", ".join(f"C{x['cluster']} (weight {x['weight']:.0f})" for x in dossier["top_outgoing_clusters"]) or "none"
            ),
            "",
            "**Representative pages/hosts:**",
        ]
        for rr in reps_out:
            md += [
                "",
                f"- `{rr['host_id']}` — **{rr['title'] or '(no title)'}**",
                f"  - snippet: {rr['snippet']}",
            ]

    # Save assignments with no need to expose raw onion names in the report.
    assignments = host_df.copy()
    assignments["semantic_cluster"] = labels
    assignments["host_alias"] = assignments["host"].astype(str).map(aliases)
    assignments.to_csv(outdir / "host_semantic_clusters.csv", index=False)
    cluster_edges.to_csv(outdir / "cluster_to_cluster_edges.csv", index=False)
    k_scores.to_csv(outdir / "cluster_count_selection.csv", index=False)
    (outdir / "cluster_report.md").write_text("\n".join(md), encoding="utf-8")
    (outdir / "cluster_dossiers.json").write_text(json.dumps(dossiers, ensure_ascii=False, indent=2), encoding="utf-8")

    prompt = """# LLM cluster-labeling task\n\nYou are labeling semantic clusters from an offline Tor/dark-web research dataset.\nFor each cluster below, use ONLY the supplied top terms, representative titles/snippets, and link context.\nDo not infer illegal activity, malware, or identity unless the evidence explicitly supports it.\nContent may be multilingual. Translate/interpret the snippets as needed.\n\nReturn for each cluster:\n1. short descriptive label (3–8 words),\n2. 1–2 sentence evidence-based summary,\n3. 3–6 dominant themes,\n4. confidence: high / medium / low,\n5. any obvious mixture of unrelated subtopics,\n6. interpretation of outgoing links: what other semantic groups this cluster mainly points to.\n\nDo not reproduce onion addresses.\n\n## Cluster dossiers\n\n""" + json.dumps(dossiers, ensure_ascii=False, indent=2)
    (outdir / "llm_cluster_labeling_prompt.md").write_text(prompt, encoding="utf-8")

    meta = {
        "embedding": embed_meta,
        "n_hosts": len(host_df),
        "k": int(len(set(labels.tolist()))),
        "agreement": agreement,
        "privacy_note": "Reports use hashed host aliases unless --show-onion-hosts is requested.",
    }
    (outdir / "semantic_analysis_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    host_df = safe_read(args.hosts)
    edge_df = safe_read(args.edges)
    page_df = safe_read(args.pages)

    if "host" not in host_df.columns or "content_text" not in host_df.columns:
        raise ValueError("host_features_full.csv must contain host and content_text")
    for c in ["source_host", "target_host"]:
        if c not in edge_df.columns:
            raise ValueError(f"edge CSV must contain {c}")
    if "host" not in page_df.columns:
        raise ValueError("page HTML feature CSV must contain host")

    host_df = host_df.copy()
    host_df["content_text"] = host_df["content_text"].fillna("").map(clean_text)
    texts = host_df["content_text"].tolist()

    Z, embed_meta = build_embeddings(texts, args.embedding, args.model, args.svd_dim, args.seed)

    if str(args.k).lower() == "auto":
        k, scores = choose_k(Z, args.k_min, args.k_max, args.seed)
    else:
        k = int(args.k)
        scores = pd.DataFrame([{"k": k, "silhouette_cosine": np.nan}])
    labels, centroids = fit_clusters(Z, k, args.seed)

    topterms = cluster_top_terms(texts, labels, args.top_terms)
    h2c = dict(zip(host_df["host"].astype(str), labels.astype(int)))
    cluster_edges = cluster_graph_stats(edge_df, h2c)

    np.save(outdir / "host_semantic_embeddings.npy", Z)
    make_reports(host_df, page_df, labels, Z, centroids, topterms, cluster_edges, args, embed_meta, scores)

    print(f"Semantic clusters: {k}")
    print(f"Embedding backend: {embed_meta['backend']}")
    print(f"Outputs: {outdir.resolve()}")
    print("Main report: cluster_report.md")
    print("LLM-ready evidence: cluster_dossiers.json / llm_cluster_labeling_prompt.md")


if __name__ == "__main__":
    main()
