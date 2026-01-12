"""
tag-query retrieval evaluation pipeline.

- Loads a dataset of documents, each with:
    - human article text
    - llm summary text
    - llm-extracted facts (bullets)
    - human annotated tags (list of strings)
- Uses each distinct tag as a query (one query per tag).
- Evaluates retrieval quality for each query with:
    Precision@k, Recall@k, F1@k, DCG@k (binary relevance)
- Implements three retrieval setups per corpus:
    (A) BM25
    (B) Dense bi-encoder retrieval with FAISS
    (C) Dense retrieval + cross-encoder reranking
- Produces a few plots comparing:
    - human_article vs llm_article vs llm_bullets
    - BM25 vs Dense vs Dense+Rerank
"""

from __future__ import annotations

import os
import re
import math
import ast
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
from collections import Counter
import numpy as np
import pandas as pd

# Retrieval
from rank_bm25 import BM25Okapi

# Dense models
from sentence_transformers import SentenceTransformer, CrossEncoder

# FAISS
import faiss

# Plotting
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

DOC_FIELDS_DEFAULT = ["human_article", "llm_article", "llm_bullets"]

BIENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 5                 # evaluate @10
RERANK_CANDIDATES = 100    # retrieve top-100 from dense, rerank, then evaluate @10


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class Document:
    doc_id: str
    human_article: str
    llm_article: str
    llm_bullets: str
    tags: Tuple[str, ...]   # normalized tags (strings)


# -----------------------------
# Utility: normalization & tokenization
# -----------------------------

_TAG_WS_RE = re.compile(r"\s+")

def normalize_tag(tag: str) -> str:
    """
    Normalize tags for matching:
    - strip
    - collapse whitespace
    - keep original casing by default (change to .lower() if you want case-insensitive tags)
    """
    tag = (tag or "").strip()
    tag = _TAG_WS_RE.sub(" ", tag)
    return tag

_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+")

def tokenize(text: str) -> List[str]:
    """
    Simple tokenization for BM25:
    - extracts alphanumerics
    - lowercases tokens
    You can replace this with spaCy if you want, but keep it deterministic.
    """
    if not text:
        return []
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


# -----------------------------
# Loading dataset
# -----------------------------

def load_csv(path: str) -> List[Document]:
    """
    Expected CSV columns:
      index, content, llm_text, bullets, tags

    Where tags is either:
      - a JSON list string: ["Trump","Ukraine"]
      - or a delimiter-separated string: Trump|Ukraine|Deutschland

    Adjust TAGS_DELIM if needed.
    """
    df = pd.read_csv(path)
    docs: List[Document] = []

    for _, row in df.iterrows():
        doc_id = str(row["index"])

        tags_list = ast.literal_eval(row.get("tags", ""))

        tags_norm = tuple(normalize_tag(t) for t in tags_list if normalize_tag(t))

        docs.append(
            Document(
                doc_id=doc_id,
                human_article=str(row.get("content", "") or ""),
                llm_article=str(row.get("llm_text", "") or ""),
                llm_bullets=str(row.get("bullets", "") or ""),
                tags=tags_norm,
            )
        )
    return docs

# -----------------------------
# Query set (distinct tags)
# -----------------------------

def tag_level_precision_summary(results_df, k=10):
    df = results_df.copy()
    df = df[df["query_tag"] != "__MACRO_AVG__"]

    prec_col = f"precision@{k}"

    summary = (
        df.groupby("query_tag")[prec_col]
          .mean()
          .reset_index()
          .rename(columns={prec_col: "mean_precision"})
    )

    return summary.sort_values("mean_precision", ascending=False)

def analyze_tag_inventory(docs, top_k: int):
    """
    Analyze tag distribution and identify which tags are evaluable
    for Precision@K-style metrics (i.e., n_relevant >= K).
    """
    tag_counter = Counter()
    for d in docs:
        tag_counter.update(d.tags)

    evaluable_counter = Counter(
        {tag: freq for tag, freq in tag_counter.items() if freq >= top_k}
    )

    dropped_tags = set(tag_counter) - set(evaluable_counter)

    return {
        "n_distinct_tags_total": len(tag_counter),
        "n_distinct_tags_evaluable": len(evaluable_counter),
        "n_total_tag_assignments": sum(tag_counter.values()),
        "tag_frequencies": tag_counter,
        "evaluable_tag_frequencies": evaluable_counter,
        "dropped_tags": dropped_tags,
    }

def build_tag_queries(docs: List[Document]) -> List[str]:
    """
    One query per distinct tag that matches atleast k articles.
    """
    all_tags = set()
    final_tags = set()
    for d in docs:
        for t in d.tags:
            if t not in all_tags:
                all_tags.add(t)
    for tag in all_tags:
        if len(relevant_doc_indices_for_tag(docs, tag)) >= TOP_K:
            final_tags.add(tag)

    return sorted(final_tags)


def relevant_doc_indices_for_tag(docs: List[Document], tag: str) -> List[int]:
    """
    Ground truth relevance: doc is relevant if tag in doc.tags
    Returns indices into docs list.
    """
    rel = []
    for i, d in enumerate(docs):
        if tag in d.tags:
            rel.append(i)
    return rel


# -----------------------------
# Metrics @k
# -----------------------------

def reciprocal_rank(retrieved, relevant):
    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0

def precision_at_k(retrieved: List[int], relevant: set[int], k: int) -> float:
    top = retrieved[:k]

    if k == 0:
        return 0.0
    return sum(1 for idx in top if idx in relevant) / float(k)

def recall_at_k(retrieved: List[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0

    top = retrieved[:k]
    return sum(1 for idx in top if idx in relevant) / float(len(relevant))

def f1(p: float, r: float) -> float:
    if (p + r) == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)

def dcg_at_k(retrieved: List[int], relevant: set[int], k: int) -> float:
    """
    Binary DCG:
      DCG@k = sum_{i=1..k} rel_i / log2(i+1)
    where rel_i = 1 if retrieved[i] relevant else 0
    """
    score = 0.0
    for rank, idx in enumerate(retrieved[:k], start=1):
        rel_i = 1.0 if idx in relevant else 0.0
        score += rel_i / math.log2(rank + 1)
    return score

def ndcg_at_k(retrieved, relevant, k):
    dcg = dcg_at_k(retrieved, relevant, k)

    ideal_rels = [1] * min(len(relevant), k)
    idcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(ideal_rels)
    )

    return dcg / idcg if idcg > 0 else 0.0



# -----------------------------
# BM25 Retriever (per field)
# -----------------------------

class BM25Retriever:
    def __init__(self, docs_text: List[str]):
        """
        docs_text: list aligned with docs[] indices
        """
        self.corpus_tokens = [tokenize(t) for t in docs_text]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int) -> List[int]:
        q_tok = tokenize(query)
        scores = self.bm25.get_scores(q_tok)  # numpy array
        # argsort descending
        idx = np.argsort(-scores)[:top_k]
        return idx.tolist()


# -----------------------------
# Dense Retriever (bi-encoder + FAISS)
# -----------------------------

class DenseRetriever:
    def __init__(self, embeddings: np.ndarray, use_inner_product: bool = True):
        """
        embeddings: shape (N, D), float32
        If use_inner_product=True, embeddings are expected to be L2-normalized,
        and FAISS uses IndexFlatIP to approximate cosine similarity.
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.emb = embeddings
        n, d = embeddings.shape

        if use_inner_product:
            self.index = faiss.IndexFlatIP(d)
        else:
            self.index = faiss.IndexFlatL2(d)

        self.index.add(embeddings)

    @staticmethod
    def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, eps, None)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[int]:
        """
        query_vec: shape (D,) or (1, D), float32, already normalized if using IP/cosine
        """
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        scores, idx = self.index.search(query_vec, top_k)
        return idx[0].tolist()


# -----------------------------
# Cross-encoder reranker
# -----------------------------

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidate_indices: List[int],
        docs_text: List[str],
        top_k: int,
    ) -> List[int]:
        """
        Rerank candidates with cross-encoder on (query, doc_text).
        Returns reranked indices, top_k long.
        """
        pairs = [(query, docs_text[i]) for i in candidate_indices]
        scores = self.model.predict(pairs)  # shape (len(candidates),)
        order = np.argsort(-np.asarray(scores))
        reranked = [candidate_indices[i] for i in order[:top_k]]
        return reranked


# -----------------------------
# Evaluation runner
# -----------------------------

def evaluate_run(
    docs: List[Document],
    queries: List[str],
    run_name: str,
    search_fn,
    k: int = TOP_K,
) -> pd.DataFrame:
    """
    Generic evaluation over tag-queries.

    search_fn: callable(query: str) -> List[int] (doc indices in rank order)
    Returns per-query metrics + macro averages.
    """
    rows = []
    for tag in queries:
        rel = set(relevant_doc_indices_for_tag(docs, tag))

        retrieved = search_fn(tag)

        p = precision_at_k(retrieved, rel, k)
        r = recall_at_k(retrieved, rel, k)
        f = f1(p, r)
        d = dcg_at_k(retrieved, rel, k)
        n = ndcg_at_k(retrieved, rel, k)
        m = reciprocal_rank(retrieved, rel)

        rows.append(
            {
                "run": run_name,
                "query_tag": tag,
                "n_relevant": len(rel),
                f"precision@{k}": p,
                f"recall@{k}": r,
                f"f1@{k}": f,
                f"dcg@{k}": d,
                f"ndcg@{k}": n,
                f"MRR@{k}": m
            }
        )

    df = pd.DataFrame(rows)

    # Add macro row (simple mean over queries)
    macro = {
        "run": run_name,
        "query_tag": "__MACRO_AVG__",
        "n_relevant": df["n_relevant"].mean(),
        f"precision@{k}": df[f"precision@{k}"].mean(),
        f"recall@{k}": df[f"recall@{k}"].mean(),
        f"f1@{k}": df[f"f1@{k}"].mean(),
        f"dcg@{k}": df[f"dcg@{k}"].mean(),
        f"ndcg@{k}": df[f"ndcg@{k}"].mean(),
        f"MRR@{k}": df[f"MRR@{k}"].mean()
    }
    df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)
    return df


# -----------------------------
# Plotting
# -----------------------------

def plot_precision_vs_frequency(tag_precision, tag_counter, out_path=None):
    df = tag_precision.copy()
    df["frequency"] = df["query_tag"].map(tag_counter)

    plt.figure(figsize=(6, 4))
    plt.scatter(df["frequency"], df["mean_precision"], alpha=0.6)
    plt.xscale("log")
    plt.xlabel("Documents per tag (log)")
    plt.ylabel(f"Mean Precision@{TOP_K}")
    plt.title("Tag frequency vs retrieval performance")

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()

def plot_tag_frequency_loglog(tag_counter, out_path=None):
    freqs = np.array(list(tag_counter.values()))

    plt.figure(figsize=(7, 4))
    plt.hist(freqs, bins=np.logspace(np.log10(1), np.log10(freqs.max()), 50))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Documents per tag (log)")
    plt.ylabel("Number of tags (log)")
    plt.title("Tag frequency distribution (log-log)")

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()

def plot_tag_frequency_histogram(tag_counter, out_path=None):
    freqs = np.array(list(tag_counter.values()))

    plt.figure(figsize=(7, 4))
    plt.hist(freqs, bins=50)
    plt.xlabel("Number of documents per tag")
    plt.ylabel("Number of tags")
    plt.title("Tag frequency distribution")

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()





def plot_macro_bars(all_results: pd.DataFrame, k: int, out_dir: str, n_q: int) -> None:
    """
    Bar plots for macro-averaged metrics across runs.
    """
    os.makedirs(out_dir, exist_ok=True)

    macro = all_results[all_results["query_tag"] == "__MACRO_AVG__"].copy()
    metrics = [f"precision@{k}", f"recall@{k}", f"f1@{k}", f"dcg@{k}", f"ndcg@{k}", f"MRR@{k}"]

    for m in metrics:
        plt.figure()
        plt.bar(macro["run"], macro[m])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(m)
        plt.title(f"{m} across runs,  n={n_q}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"macro_{m}.png"), dpi=160)
        plt.close()


def plot_field_comparison(all_results: pd.DataFrame, k: int, out_dir: str, n_q: int) -> None:
    """
    Field vs. method comparison with visual difference cues:
    - grouped bars
    - dashed vertical connectors between methods
    - horizontal reference lines (best per field)
    - delta annotations
    """
    os.makedirs(out_dir, exist_ok=True)
    macro = all_results[all_results["query_tag"] == "__MACRO_AVG__"].copy()

    if not macro["run"].str.contains("::").all():
        return

    macro["method"] = macro["run"].str.split("::").str[0]
    macro["field"] = macro["run"].str.split("::").str[1]

    methods = sorted(macro["method"].unique())

    for metric in [f"precision@{k}", f"ndcg@{k}", f"MRR@{k}"]:
        pivot = macro.pivot(index="field", columns="method", values=metric)

        fig, ax = plt.subplots(figsize=(8, 4))

        x = np.arange(len(pivot.index))
        width = 0.8 / len(methods)

        # --- Bars ---
        for j, method in enumerate(methods):
            ax.bar(
                x + j * width,
                pivot[method].values,
                width,
                label=method
            )

        # --- Dashed vertical connectors (method-to-method deltas) ---
        for i, field in enumerate(pivot.index):
            values = pivot.loc[field, methods].values
            for j in range(len(values) - 1):
                x_pos = x[i] + j * width + width / 2
                ax.plot(
                    [x_pos, x_pos + width],
                    [values[j], values[j + 1]],
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7
                )

                # delta annotation
                delta = values[j + 1] - values[j]
                ax.text(
                    x_pos + width / 2,
                    (values[j] + values[j + 1]) / 2,
                    f"{delta:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

        # --- Horizontal reference line: best method per field ---
        for i, field in enumerate(pivot.index):
            best = pivot.loc[field].max()
            ax.hlines(
                y=best,
                xmin=x[i] - 0.05,
                xmax=x[i] + width * len(methods),
                linestyles="dotted",
                linewidth=1,
                alpha=0.6
            )

        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(pivot.index)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}: corpora vs methods, n={n_q}")
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            os.path.join(out_dir, f"field_vs_method_{metric}_extended.png"),
            dpi=160
        )
        plt.close(fig)



# -----------------------------
# Main pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset (.jsonl or .csv).")
    parser.add_argument("--out_dir", type=str, default="retrieval_eval_out", help="Where to write results/plots.")
    parser.add_argument("--doc_fields", type=str, default=",".join(DOC_FIELDS_DEFAULT),
                        help="Comma-separated document fields to evaluate.")
    parser.add_argument("--k", type=int, default=TOP_K, help="Evaluate metrics @k.")
    parser.add_argument("--rerank_candidates", type=int, default=RERANK_CANDIDATES,
                        help="Candidates to retrieve for reranking (dense stage).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Optional: set SentenceTransformer device (e.g. 'cuda' or 'cpu').")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load data
    docs = load_csv(args.data_path)
    if not docs:
        raise RuntimeError("No documents loaded.")

    print(f"Number of articles: {len(docs)}")
    doc_fields = [f.strip() for f in args.doc_fields.split(",") if f.strip()]

    for f in doc_fields:
        if not hasattr(docs[0], f):
            raise ValueError(f"Unknown doc field: {f}")

    # 2) Build queries (= distinct tags)
    queries = build_tag_queries(docs)
    q_n = len(queries)
    print(f"Number of query tags with >= K articles: {q_n}")

    if not queries:
        raise RuntimeError("No tags found -> no queries to evaluate.")

    # 3) Load models (once)
    biencoder = SentenceTransformer(BIENCODER_NAME, device=args.device) if args.device else SentenceTransformer(BIENCODER_NAME)
    reranker = CrossEncoderReranker(CROSS_ENCODER_NAME)

    all_results = []

    # 4) Evaluate for each doc field
    for field in doc_fields:
        docs_text = [getattr(d, field) or "" for d in docs]
        # -------- BM25 --------
        bm25 = BM25Retriever(docs_text)
        run_name = f"bm25::{field}"
        df_bm25 = evaluate_run(
            docs=docs,
            queries=queries,
            run_name=run_name,
            search_fn=lambda q, bm25=bm25, k=args.k: bm25.search(q, top_k=k),
            k=args.k,
        )
        all_results.append(df_bm25)

        # -------- Dense bi-encoder + FAISS --------
        # Encode docs
        doc_emb = biencoder.encode(
            docs_text,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity via inner product
        ).astype(np.float32)

        dense = DenseRetriever(doc_emb, use_inner_product=True)

        def dense_search(q: str, top_k: int) -> List[int]:
            q_emb = biencoder.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
            return dense.search(q_emb, top_k=top_k)

        run_name = f"dense::{field}"
        df_dense = evaluate_run(
            docs=docs,
            queries=queries,
            run_name=run_name,
            search_fn=lambda q, k=args.k: dense_search(q, top_k=k),
            k=args.k,
        )
        all_results.append(df_dense)

        # -------- Dense + Cross-encoder rerank --------
        def dense_rerank_search(q: str, top_k: int) -> List[int]:
            candidates = dense_search(q, top_k=args.rerank_candidates)
            reranked = reranker.rerank(q, candidates, docs_text, top_k=top_k)
            return reranked

        run_name = f"dense_rerank::{field}"
        df_rerank = evaluate_run(
            docs=docs,
            queries=queries,
            run_name=run_name,
            search_fn=lambda q, k=args.k: dense_rerank_search(q, top_k=k),
            k=args.k,
        )
        all_results.append(df_rerank)

    # 5) Save results
    results = pd.concat(all_results, ignore_index=True)
    results_path = os.path.join(out_dir, "results_per_query.csv")
    results.to_csv(results_path, index=False)

    # TAG Analysis
    tag_stats = analyze_tag_inventory(docs, top_k=TOP_K)

    print("Distinct tags (total):", tag_stats["n_distinct_tags_total"])
    print("Distinct tags (evaluable):", tag_stats["n_distinct_tags_evaluable"])
    print("Total tag assignments:", tag_stats["n_total_tag_assignments"])
    print("Dropped tags (n_relevant < K):", len(tag_stats["dropped_tags"]))

    tag_precision = tag_level_precision_summary(results, k=TOP_K)

    best_20 = tag_precision.head(20)
    worst_20 = tag_precision.tail(20)

    print("\nTop 20 best-performing tags:")
    print(best_20.to_string(index=False))

    print("\nTop 20 worst-performing tags:")
    print(worst_20.to_string(index=False))
    
    plot_tag_frequency_histogram(
    tag_stats["tag_frequencies"],
    out_path="tag_frequency_histogram.png"
    )

    plot_precision_vs_frequency(
    tag_precision,
    tag_stats["tag_frequencies"],
    out_path="precision_vs_frequency.png")

    # 6) Plots (macro)
    plot_macro_bars(results, k=args.k, out_dir=out_dir, n_q=q_n)
    plot_field_comparison(results, k=args.k, out_dir=out_dir, n_q=q_n)

    # 7) Print a small macro summary to stdout
    macro = results[results["query_tag"] == "__MACRO_AVG__"].copy()

    print("\nMacro summary:")
    cols = ["run", f"precision@{args.k}", f"recall@{args.k}", f"f1@{args.k}", f"dcg@{args.k}", f"ndcg@{args.k}", f"MRR@{args.k}"]
    print(macro[cols].to_string(index=False))

    print(f"\nWrote:\n- {results_path}\n- plots into: {out_dir}")


if __name__ == "__main__":
    main()
