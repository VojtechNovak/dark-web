#!/usr/bin/env python3
"""
Classical exploratory analysis for a Tor/dark-web hyperlink dataset.

Expected inputs:
  nodes.csv: id, full_link, status, html_id
  edges.csv: source_id, target_id
  vybrane_html.zip: HTML snapshots named <html_id>.html (possibly in a subfolder)

The script NEVER executes HTML/JavaScript. It only parses HTML statically.

Outputs are written to --out.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


RANDOM_SEED = 42


def hostname(url: str) -> str | None:
    try:
        h = urlparse(str(url)).hostname
        return h.lower() if h else None
    except Exception:
        return None


def basic_graph_summary(G: nx.DiGraph, name: str) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {"name": name, "nodes": 0, "edges": 0}

    wccs = list(nx.weakly_connected_components(G))
    sccs = list(nx.strongly_connected_components(G))

    return {
        "name": name,
        "nodes": n,
        "edges": m,
        "density": nx.density(G),
        "reciprocity": nx.reciprocity(G) if m else np.nan,
        "weak_components": len(wccs),
        "largest_wcc": max(map(len, wccs), default=0),
        "strong_components": len(sccs),
        "largest_scc": max(map(len, sccs), default=0),
        "isolates": nx.number_of_isolates(G),
        "mean_in_degree": m / n,
        "mean_out_degree": m / n,
    }


def build_page_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    ids = set(nodes["id"].astype(int))
    ee = edges[
        edges["source_id"].isin(ids) & edges["target_id"].isin(ids)
    ].copy()

    G = nx.DiGraph()
    G.add_nodes_from(ids)
    G.add_edges_from(ee[["source_id", "target_id"]].itertuples(index=False, name=None))
    return G


def largest_wcc_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    if G.number_of_nodes() == 0:
        return G.copy()
    component = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(component).copy()


def compute_node_metrics(G: nx.DiGraph) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    Gu = G.to_undirected()

    pagerank = nx.pagerank(G, alpha=0.85)

    # Exact betweenness is fine for ~1k nodes; change to k=200 for larger graphs.
    if G.number_of_nodes() <= 2500:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    else:
        betweenness = nx.betweenness_centrality(
            G, k=min(200, G.number_of_nodes()), seed=RANDOM_SEED, normalized=True
        )

    try:
        hubs, authorities = nx.hits(G, max_iter=2000, normalized=True)
    except Exception:
        hubs = {n: np.nan for n in G.nodes}
        authorities = {n: np.nan for n in G.nodes}

    clustering = nx.clustering(Gu)
    core_number = nx.core_number(Gu) if Gu.number_of_edges() else {n: 0 for n in Gu.nodes}

    rows = []
    for n in G.nodes:
        rows.append(
            {
                "id": n,
                "in_degree": G.in_degree(n),
                "out_degree": G.out_degree(n),
                "total_degree": G.degree(n),
                "pagerank": pagerank.get(n, np.nan),
                "betweenness": betweenness.get(n, np.nan),
                "hub_score": hubs.get(n, np.nan),
                "authority_score": authorities.get(n, np.nan),
                "clustering_undirected": clustering.get(n, np.nan),
                "core_number": core_number.get(n, np.nan),
            }
        )

    return pd.DataFrame(rows)


def louvain_partition(G: nx.DiGraph) -> tuple[dict, float, list[set]]:
    Gu = G.to_undirected()
    if Gu.number_of_nodes() == 0:
        return {}, np.nan, []
    if Gu.number_of_edges() == 0:
        comms = [{n} for n in Gu.nodes]
        labels = {n: i for i, c in enumerate(comms) for n in c}
        return labels, np.nan, comms

    comms = nx.community.louvain_communities(
        Gu, weight="weight", resolution=1.0, seed=RANDOM_SEED
    )
    modularity = nx.community.modularity(Gu, comms, weight="weight")
    labels = {n: i for i, c in enumerate(comms) for n in c}
    return labels, modularity, comms


def build_host_graph(nodes: pd.DataFrame, page_edges: pd.DataFrame) -> tuple[nx.DiGraph, pd.DataFrame]:
    tmp = nodes[["id", "host"]].dropna().copy()
    id_to_host = dict(zip(tmp["id"], tmp["host"]))

    host_stats = (
        nodes.dropna(subset=["host"])
        .groupby("host")
        .agg(
            page_count=("id", "size"),
            html_count=("html_id", lambda s: s.notna().sum()),
        )
        .reset_index()
    )

    weights = Counter()
    for s, t in page_edges[["source_id", "target_id"]].itertuples(index=False, name=None):
        hs = id_to_host.get(s)
        ht = id_to_host.get(t)
        if not hs or not ht or hs == ht:
            continue
        weights[(hs, ht)] += 1

    H = nx.DiGraph()
    for r in host_stats.itertuples(index=False):
        H.add_node(r.host, page_count=int(r.page_count), html_count=int(r.html_count))
    for (hs, ht), w in weights.items():
        H.add_edge(hs, ht, weight=w)

    return H, host_stats


def html_zip_index(zip_path: Path) -> dict[str, str]:
    """Map filename stem -> zip member path."""
    if not zip_path.exists():
        return {}
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.lower().endswith(".html"):
                out[Path(member).stem] = member
    return out


def safe_html_features(raw: bytes) -> dict:
    """Static HTML parsing only; does not execute scripts."""
    soup = BeautifulSoup(raw, "html.parser")

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    n_scripts = len(soup.find_all("script"))

    # Remove script/style contents before extracting visible-ish text.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    return {
        "title": title[:500],
        "html_bytes": len(raw),
        "text_chars": len(text),
        "text_words": len(text.split()),
        "n_links": len(soup.find_all("a")),
        "n_forms": len(soup.find_all("form")),
        "n_inputs": len(soup.find_all("input")),
        "n_images": len(soup.find_all("img")),
        "n_scripts": n_scripts,
        "n_iframes": len(soup.find_all("iframe")),
    }


def extract_html_features(nodes: pd.DataFrame, zip_path: Path) -> pd.DataFrame:
    index = html_zip_index(zip_path)
    if not index:
        return pd.DataFrame(columns=["id", "html_present"])

    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for r in nodes[["id", "html_id"]].itertuples(index=False):
            if pd.isna(r.html_id):
                rows.append({"id": r.id, "html_present": False})
                continue

            key = str(r.html_id)
            member = index.get(key)
            if not member:
                rows.append({"id": r.id, "html_present": False})
                continue

            try:
                raw = zf.read(member)
                feats = safe_html_features(raw)
                feats.update({"id": r.id, "html_present": True})
                rows.append(feats)
            except Exception as exc:
                rows.append(
                    {
                        "id": r.id,
                        "html_present": False,
                        "html_parse_error": type(exc).__name__,
                    }
                )

    return pd.DataFrame(rows)


def plot_degree_distribution(G: nx.DiGraph, path: Path, title: str) -> None:
    deg = np.array([d for _, d in G.degree()], dtype=float)
    if len(deg) == 0:
        return

    vals, counts = np.unique(deg, return_counts=True)
    positive = vals > 0

    plt.figure(figsize=(7, 5))
    if positive.any():
        plt.loglog(vals[positive], counts[positive], marker="o", linestyle="none")
    plt.xlabel("Total degree")
    plt.ylabel("Number of nodes")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_component_sizes(G: nx.DiGraph, path: Path, title: str) -> None:
    sizes = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
    if not sizes:
        return
    plt.figure(figsize=(7, 5))
    plt.bar(np.arange(1, len(sizes) + 1), sizes)
    plt.xlabel("Weak component rank")
    plt.ylabel("Component size")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_community_sizes(comms: list[set], path: Path, title: str) -> None:
    sizes = sorted((len(c) for c in comms), reverse=True)
    if not sizes:
        return
    plt.figure(figsize=(7, 5))
    plt.bar(np.arange(len(sizes)), sizes)
    plt.xlabel("Community")
    plt.ylabel("Number of nodes")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="nodes.csv")
    ap.add_argument("--edges", default="edges.csv")
    ap.add_argument("--html", default="vybrane_html.zip")
    ap.add_argument("--out", default="eda_output")
    ap.add_argument(
        "--status",
        type=int,
        default=200,
        help="HTTP status used for the primary induced graph (default: 200)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)

    required_nodes = {"id", "full_link", "status", "html_id"}
    required_edges = {"source_id", "target_id"}
    if not required_nodes.issubset(nodes.columns):
        raise ValueError(f"nodes.csv must contain {sorted(required_nodes)}")
    if not required_edges.issubset(edges.columns):
        raise ValueError(f"edges.csv must contain {sorted(required_edges)}")

    nodes = nodes.drop_duplicates(subset=["id"]).copy()
    edges = edges.drop_duplicates(subset=["source_id", "target_id"]).copy()
    edges = edges[edges["source_id"] != edges["target_id"]].copy()

    nodes["host"] = nodes["full_link"].map(hostname)

    # ------------------------------------------------------------
    # 1) Full graph exactly as supplied
    # ------------------------------------------------------------
    G_all = build_page_graph(nodes, edges)

    # ------------------------------------------------------------
    # 2) Primary graph: induced subgraph of HTTP 200 nodes
    # ------------------------------------------------------------
    nodes_ok = nodes[nodes["status"].eq(args.status)].copy()
    ok_ids = set(nodes_ok["id"])
    edges_ok = edges[
        edges["source_id"].isin(ok_ids) & edges["target_id"].isin(ok_ids)
    ].copy()
    G_ok = build_page_graph(nodes_ok, edges_ok)

    # Recommended analysis core: largest weakly connected component.
    G_core = largest_wcc_subgraph(G_ok)
    core_ids = set(G_core.nodes)
    nodes_core = nodes_ok[nodes_ok["id"].isin(core_ids)].copy()
    edges_core = edges_ok[
        edges_ok["source_id"].isin(core_ids) & edges_ok["target_id"].isin(core_ids)
    ].copy()

    summaries = [
        basic_graph_summary(G_all, "all_supplied_nodes"),
        basic_graph_summary(G_ok, f"status_{args.status}_induced"),
        basic_graph_summary(G_core, f"status_{args.status}_largest_wcc"),
    ]

    # ------------------------------------------------------------
    # 3) Page-level graph metrics and communities
    # ------------------------------------------------------------
    page_metrics = compute_node_metrics(G_core)
    page_labels, page_modularity, page_comms = louvain_partition(G_core)
    page_metrics["community_louvain"] = page_metrics["id"].map(page_labels)
    page_metrics = page_metrics.merge(
        nodes_core[["id", "full_link", "host", "status", "html_id"]], on="id", how="left"
    )
    page_metrics.to_csv(out / "page_node_metrics.csv", index=False)

    # Community composition by host: useful to detect communities that are just websites.
    community_host = (
        page_metrics.groupby(["community_louvain", "host"], dropna=False)
        .size()
        .rename("pages")
        .reset_index()
    )
    community_sizes = (
        page_metrics.groupby("community_louvain").size().rename("community_pages")
    )
    community_host = community_host.merge(
        community_sizes, left_on="community_louvain", right_index=True
    )
    community_host["share_of_community"] = (
        community_host["pages"] / community_host["community_pages"]
    )
    community_host.sort_values(
        ["community_louvain", "pages"], ascending=[True, False]
    ).to_csv(out / "page_community_host_composition.csv", index=False)

    # ------------------------------------------------------------
    # 4) Host/service graph, cross-host links only
    # ------------------------------------------------------------
    H, host_stats = build_host_graph(nodes_core, edges_core)
    host_metrics = compute_node_metrics(H)
    host_labels, host_modularity, host_comms = louvain_partition(H)
    if not host_metrics.empty:
        host_metrics = host_metrics.rename(columns={"id": "host"})
        host_metrics["community_louvain"] = host_metrics["host"].map(host_labels)
        host_metrics = host_metrics.merge(host_stats, on="host", how="left")
        host_metrics.to_csv(out / "host_node_metrics.csv", index=False)

    # Weighted host edges
    pd.DataFrame(
        [
            {"source_host": u, "target_host": v, "weight": d.get("weight", 1)}
            for u, v, d in H.edges(data=True)
        ]
    ).to_csv(out / "host_edges_weighted.csv", index=False)

    # ------------------------------------------------------------
    # 5) HTML EDA for the connected HTTP-200 core
    # ------------------------------------------------------------
    html_features = extract_html_features(nodes_core, Path(args.html))
    html_features.to_csv(out / "html_features.csv", index=False)

    if not html_features.empty:
        merged = page_metrics.merge(html_features, on="id", how="left")
        merged.to_csv(out / "page_metrics_plus_html.csv", index=False)

    # ------------------------------------------------------------
    # 6) Diagnostics specifically relevant to this dataset
    # ------------------------------------------------------------
    id_to_host = dict(zip(nodes_core["id"], nodes_core["host"]))
    same_host = 0
    cross_host = 0
    unknown_host = 0
    for s, t in edges_core[["source_id", "target_id"]].itertuples(index=False, name=None):
        hs, ht = id_to_host.get(s), id_to_host.get(t)
        if hs is None or ht is None:
            unknown_host += 1
        elif hs == ht:
            same_host += 1
        else:
            cross_host += 1

    total_known = same_host + cross_host
    diagnostics = {
        "requested_status": args.status,
        "status_nodes_before_lcc": int(len(nodes_ok)),
        "status_nodes_in_largest_wcc": int(len(nodes_core)),
        "status_nodes_removed_outside_largest_wcc": int(len(nodes_ok) - len(nodes_core)),
        "page_louvain_communities": len(page_comms),
        "page_louvain_modularity": page_modularity,
        "host_louvain_communities": len(host_comms),
        "host_louvain_modularity": host_modularity,
        "same_host_edges_in_core": same_host,
        "cross_host_edges_in_core": cross_host,
        "same_host_edge_fraction": same_host / total_known if total_known else np.nan,
        "unknown_host_edges": unknown_host,
        "unique_hosts_in_core": int(nodes_core["host"].nunique(dropna=True)),
        "html_ids_in_core": int(nodes_core["html_id"].notna().sum()),
        "html_snapshots_found_in_zip": int(html_features.get("html_present", pd.Series(dtype=bool)).fillna(False).sum()) if not html_features.empty else 0,
    }

    with open(out / "graph_summary.json", "w", encoding="utf-8") as f:
        json.dump({"graphs": summaries, "diagnostics": diagnostics}, f, indent=2, default=float)

    pd.DataFrame(summaries).to_csv(out / "graph_summary.csv", index=False)

    # Top central nodes/services
    if not page_metrics.empty:
        page_metrics.nlargest(30, "pagerank").to_csv(out / "top_pages_pagerank.csv", index=False)
        page_metrics.nlargest(30, "betweenness").to_csv(out / "top_pages_betweenness.csv", index=False)
    if not host_metrics.empty:
        host_metrics.nlargest(30, "pagerank").to_csv(out / "top_hosts_pagerank.csv", index=False)
        host_metrics.nlargest(30, "betweenness").to_csv(out / "top_hosts_betweenness.csv", index=False)

    # ------------------------------------------------------------
    # 7) Plots
    # ------------------------------------------------------------
    plot_degree_distribution(G_core, out / "page_degree_distribution.png", "Page graph: degree distribution")
    plot_component_sizes(G_ok, out / "page_wcc_sizes_before_core_filter.png", "HTTP-200 induced graph: WCC sizes")
    plot_community_sizes(page_comms, out / "page_louvain_community_sizes.png", "Page graph: Louvain community sizes")
    plot_degree_distribution(H, out / "host_degree_distribution.png", "Host graph: degree distribution")
    plot_community_sizes(host_comms, out / "host_louvain_community_sizes.png", "Host graph: Louvain community sizes")

    # Console report
    print("\n=== DATASET CHECK ===")
    print(f"Input nodes: {len(nodes):,}")
    print(f"Input edges: {len(edges):,}")
    print(f"HTTP {args.status} nodes: {len(nodes_ok):,}")
    print(
        f"HTTP {args.status} induced graph WCCs: "
        f"{nx.number_weakly_connected_components(G_ok):,}; "
        f"largest WCC = {G_core.number_of_nodes():,} nodes"
    )

    print("\n=== PRIMARY PAGE GRAPH (largest HTTP-200 WCC) ===")
    for k, v in basic_graph_summary(G_core, "core").items():
        if k != "name":
            print(f"{k}: {v}")
    print(f"Louvain communities: {len(page_comms)}")
    print(f"Louvain modularity: {page_modularity:.4f}")

    print("\n=== HOST GRAPH (cross-host links only) ===")
    for k, v in basic_graph_summary(H, "hosts").items():
        if k != "name":
            print(f"{k}: {v}")
    print(f"Louvain communities: {len(host_comms)}")
    print(f"Louvain modularity: {host_modularity:.4f}")

    print("\n=== PAGE-LINK COMPOSITION ===")
    print(f"same-host edges: {same_host:,}")
    print(f"cross-host edges: {cross_host:,}")
    if total_known:
        print(f"same-host fraction: {same_host / total_known:.3%}")

    print(f"\nResults written to: {out.resolve()}")


if __name__ == "__main__":
    main()
