"""
exp/exp08_cluster_pseudo_siebert_sched_smooth/analyze.py
=========================================================
Step 0 (optional): visualize the embedding space and evaluate cluster purity
BEFORE running train.py.

Encoder: siebert/sentiment-roberta-large-english (1024-dim, sentiment fine-tuned)

Outputs (saved to results/figures/exp08/)
-----------------------------------------
  tsne_by_label.png          : t-SNE scatter colored by true label
  purity_hist.png            : distribution of cluster purities
  dendrogram_centroids.png   : hierarchical clustering of K-means centroids
  pseudo_label_stats.txt     : purity threshold sweep summary

Usage
-----
    python exp/exp08_cluster_pseudo_siebert_sched_smooth/analyze.py
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram as sp_dendrogram
from scipy.cluster.hierarchy import linkage as sp_linkage
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, RESULTS_DIR, TEST_FILE, TRAIN_FILE


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labeled_csv(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["TEXT"].strip())
            labels.append(int(row["LABEL"].strip()))
    return texts, labels


def load_unlabeled_csv(path: Path) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["TEXT"].strip())
    return texts


@torch.no_grad()
def encode_texts(
    texts: list[str],
    encoder,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """Mean-pooled embeddings → numpy [N, hidden_dim]."""
    encoder.eval()
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc  = tokenizer(batch, max_length=max_length, padding="max_length",
                         truncation=True, return_tensors="pt")
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        tok  = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m    = mask.unsqueeze(-1).float()
        vec  = (tok * m).sum(1) / m.sum(1)
        vecs.append(vec.cpu().numpy())
        if (i // batch_size + 1) % 20 == 0 or i + batch_size >= len(texts):
            print(f"  [{min(i + batch_size, len(texts))}/{len(texts)}]")
    return np.vstack(vecs)


def cluster_purity_stats(
    labeled_cluster_ids: np.ndarray,
    labeled_labels: list[int],
    unlabeled_cluster_ids: np.ndarray,
    n_clusters: int,
    min_labeled: int,
) -> tuple[list[float], dict]:
    labels_arr = np.array(labeled_labels)
    purities, cluster_label = [], {}

    for c in range(n_clusters):
        idx = np.where(labeled_cluster_ids == c)[0]
        n = len(idx)
        if n == 0:
            continue
        pos     = int(labels_arr[idx].sum())
        purity  = max(pos, n - pos) / n
        purities.append(purity)
        if n >= min_labeled:
            cluster_label[c] = (1 if pos >= n - pos else 0, purity, n)

    return purities, cluster_label


def main() -> None:
    cfg    = load_config(EXP_DIR / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mo = cfg["model"]
    cl = cfg["clustering"]
    tr = cfg["training"]

    fig_dir = RESULTS_DIR / "figures" / cfg["experiment"]["name"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    labeled_texts, labeled_labels = load_labeled_csv(TRAIN_FILE)
    unlabeled_texts               = load_unlabeled_csv(TEST_FILE)
    all_texts = labeled_texts + unlabeled_texts
    n_labeled, n_unlabeled = len(labeled_texts), len(unlabeled_texts)
    print(f"Labeled   : {n_labeled}")
    print(f"Unlabeled : {n_unlabeled}")
    print(f"Total     : {len(all_texts)}")

    # -----------------------------------------------------------------------
    # Encode
    # -----------------------------------------------------------------------
    pretrained = MODEL_REGISTRY[mo["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)
    encoder    = AutoModel.from_pretrained(pretrained).to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    print(f"\nEncoding all texts with {pretrained}...")
    embeddings = encode_texts(all_texts, encoder, tokenizer, device,
                              max_length=tr["max_length"])
    print(f"Embeddings shape: {embeddings.shape}")

    # -----------------------------------------------------------------------
    # K-means
    # -----------------------------------------------------------------------
    K = cl["n_clusters"]
    print(f"\nRunning K-means (K={K})...")
    km = MiniBatchKMeans(n_clusters=K, random_state=cl["random_state"], n_init=5)
    cluster_ids           = km.fit_predict(embeddings)
    labeled_cluster_ids   = cluster_ids[:n_labeled]
    unlabeled_cluster_ids = cluster_ids[n_labeled:]

    # -----------------------------------------------------------------------
    # Purity analysis
    # -----------------------------------------------------------------------
    min_labeled = cl["min_labeled_per_cluster"]
    purities, cluster_info = cluster_purity_stats(
        labeled_cluster_ids, labeled_labels,
        unlabeled_cluster_ids, K, min_labeled,
    )

    stats_lines = [
        f"K-means Cluster Analysis  [{pretrained}]",
        f"{'='*50}",
        f"n_clusters              : {K}",
        f"min_labeled_per_cluster : {min_labeled}",
        f"",
        f"Mean cluster purity     : {np.mean(purities):.4f}",
        f"Median cluster purity   : {np.median(purities):.4f}",
        f"",
        "Threshold sweep:",
    ]
    print("\n" + "\n".join(stats_lines))

    for thr in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]:
        qualifying = {c: v for c, v in cluster_info.items() if v[1] >= thr}
        n_pseudo   = sum((unlabeled_cluster_ids == c).sum() for c in qualifying)
        line = (f"  purity ≥ {thr:.2f} → "
                f"{len(qualifying):3d} clusters, "
                f"{n_pseudo:5d} pseudo-labels "
                f"({n_pseudo / n_unlabeled:.1%})")
        stats_lines.append(line)
        print(line)

    stats_path = fig_dir / "pseudo_label_stats.txt"
    stats_path.write_text("\n".join(stats_lines), encoding="utf-8")
    print(f"\nStats saved: {stats_path}")

    # -----------------------------------------------------------------------
    # Purity histogram
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(purities, bins=20, edgecolor="white", color="#4C72B0", alpha=0.85)
    ax.axvline(cl["purity_threshold"], color="red", linestyle="--",
               label=f"threshold = {cl['purity_threshold']}")
    ax.set_xlabel("Cluster Purity")
    ax.set_ylabel("Number of Clusters")
    ax.set_title(f"Cluster Purity Distribution  (K={K})")
    ax.legend()
    plt.tight_layout()
    path = fig_dir / "purity_hist.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Purity histogram saved: {path}")

    # -----------------------------------------------------------------------
    # Dendrogram of K-means centroids
    # -----------------------------------------------------------------------
    la = np.array(labeled_labels)
    centroid_tag = {}
    for c in range(K):
        idx = np.where(labeled_cluster_ids == c)[0]
        n = len(idx)
        if n < min_labeled:
            centroid_tag[c] = "mixed"
        else:
            pos = int(la[idx].sum())
            purity_c = max(pos, n - pos) / n
            if purity_c >= cl["purity_threshold"]:
                centroid_tag[c] = "positive" if pos >= n - pos else "negative"
            else:
                centroid_tag[c] = "mixed"

    leaf_color_map = {"positive": "#2196F3", "negative": "#F44336", "mixed": "#BBBBBB"}
    Z = sp_linkage(km.cluster_centers_, method="ward")

    fig, ax = plt.subplots(figsize=(20, 6))
    ddata = sp_dendrogram(Z, ax=ax, no_labels=True, color_threshold=0,
                          above_threshold_color="#999999")
    leaf_order = ddata["leaves"]
    xs = np.arange(5, K * 10, 10)

    _, y_max = ax.get_ylim()
    marker_y = -y_max * 0.02
    ax.set_ylim(-y_max * 0.05, y_max * 1.02)
    for x, c_id in zip(xs, leaf_order):
        ax.plot(x, marker_y, "s",
                color=leaf_color_map[centroid_tag[c_id]], markersize=5)

    legend_handles = [
        Patch(facecolor="#2196F3", label="positive"),
        Patch(facecolor="#F44336", label="negative"),
        Patch(facecolor="#BBBBBB", label="mixed / low-count"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title(f"Dendrogram of K-means Centroids  (K={K})")
    ax.set_ylabel("Ward Distance")
    ax.set_xticks([])
    plt.tight_layout()
    path = fig_dir / "dendrogram_centroids.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Dendrogram saved: {path}")

    # -----------------------------------------------------------------------
    # t-SNE (subsample for speed)
    # -----------------------------------------------------------------------
    n_tsne = min(len(all_texts), 5000)
    print(f"\nRunning t-SNE on {n_tsne} samples (may take a few minutes)...")
    rng      = np.random.default_rng(42)
    tsne_idx = rng.choice(len(all_texts), n_tsne, replace=False)

    coords = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000,
                  init="pca").fit_transform(embeddings[tsne_idx])

    is_labeled   = tsne_idx < n_labeled
    labels_arr   = np.array(labeled_labels)

    fig, ax = plt.subplots(figsize=(10, 7))

    ul_idx = np.where(~is_labeled)[0]
    ax.scatter(coords[ul_idx, 0], coords[ul_idx, 1],
               c="lightgray", s=5, alpha=0.4, label="unlabeled", zorder=1)

    l_idx        = np.where(is_labeled)[0]
    orig_indices = tsne_idx[l_idx]
    l_labels     = labels_arr[orig_indices]

    pos = l_idx[l_labels == 1]
    neg = l_idx[l_labels == 0]

    ax.scatter(coords[pos, 0], coords[pos, 1],
               c="#2196F3", s=18, alpha=0.8, label="positive", zorder=3)
    ax.scatter(coords[neg, 0], coords[neg, 1],
               c="#F44336", s=18, alpha=0.8, label="negative", zorder=3)

    ax.set_title(f"t-SNE of Embeddings  (n={n_tsne})")
    ax.legend(markerscale=2, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    path = fig_dir / "tsne_by_label.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved: {path}")

    print("\nDone. Review the figures before running train.py.")


if __name__ == "__main__":
    main()
