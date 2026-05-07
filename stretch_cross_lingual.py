"""
Stretch 6B-S2: Cross-Lingual Embedding Comparison
Uses bert-base-multilingual-cased to compare English/Arabic climate texts.
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "bert-base-multilingual-cased"
N_PER_LANG = 10
MAX_LEN = 128
CSV_PATH = "data/climate_articles.csv"

# ---------- 1. Load + filter ----------
df = pd.read_csv(CSV_PATH)

# auto-detect the text column
text_col = next(
    (c for c in ["text", "content", "article", "body"] if c in df.columns),
    None,
)
if text_col is None:
    raise ValueError(f"Couldn't find text column in: {df.columns.tolist()}")

en_df = df[df["language"] == "en"].head(N_PER_LANG).reset_index(drop=True)
ar_df = df[df["language"] == "ar"].head(N_PER_LANG).reset_index(drop=True)
print(f"Loaded {len(en_df)} English and {len(ar_df)} Arabic texts")

# ---------- 2. Load mBERT ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def mean_pool(last_hidden, attention_mask):
    """Mean-pool token embeddings, masking padding."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def embed(texts):
    vecs = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(
                t, return_tensors="pt",
                truncation=True, max_length=MAX_LEN, padding=True,
            )
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            vecs.append(pooled.squeeze(0).numpy())
    return np.vstack(vecs)

print("Embedding English texts...")
en_emb = embed(en_df[text_col].tolist())
print("Embedding Arabic texts...")
ar_emb = embed(ar_df[text_col].tolist())

# ---------- 3. Similarity matrices ----------
cross_sim = cosine_similarity(en_emb, ar_emb)            # 10x10 EN -> AR
en_within = cosine_similarity(en_emb)                    # 10x10 EN -> EN
ar_within = cosine_similarity(ar_emb)                    # 10x10 AR -> AR

# off-diagonal means (exclude self-similarity)
def off_diag_mean(m):
    return (m.sum() - np.trace(m)) / (m.size - len(m))

print(f"\nCross-lingual mean similarity:   {cross_sim.mean():.4f}")
print(f"English within-language mean:    {off_diag_mean(en_within):.4f}")
print(f"Arabic  within-language mean:    {off_diag_mean(ar_within):.4f}")

# top cross-lingual pairs (useful for the analysis)
print("\nTop 5 EN-AR cross-lingual pairs:")
flat = [(i, j, cross_sim[i, j]) for i in range(N_PER_LANG) for j in range(N_PER_LANG)]
for i, j, s in sorted(flat, key=lambda x: -x[2])[:5]:
    print(f"  {s:.3f} | EN: {en_df[text_col].iloc[i][:60]}")
    print(f"         | AR: {ar_df[text_col].iloc[j][:60]}")

# ---------- 4. Heatmap ----------
en_labels = [t[:40] for t in en_df[text_col].tolist()]
ar_labels = [t[:40] for t in ar_df[text_col].tolist()]

plt.figure(figsize=(14, 10))
sns.heatmap(
    cross_sim,
    xticklabels=ar_labels,
    yticklabels=en_labels,
    annot=True, fmt=".2f",
    cmap="viridis",
    cbar_kws={"label": "cosine similarity"},
)
plt.xlabel("Arabic texts (first 40 chars)")
plt.ylabel("English texts (first 40 chars)")
plt.title("Cross-Lingual Cosine Similarity — mBERT (mean-pooled)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("cross_lingual_heatmap.png", dpi=150, bbox_inches="tight")
print("\nSaved cross_lingual_heatmap.png")

# ---------- 5. Combined 20x20 (optional — the spec says 20x20) ----------
all_emb = np.vstack([en_emb, ar_emb])
all_labels = [f"[EN] {t[:35]}" for t in en_df[text_col]] + \
             [f"[AR] {t[:35]}" for t in ar_df[text_col]]
full_sim = cosine_similarity(all_emb)

plt.figure(figsize=(16, 13))
sns.heatmap(
    full_sim, xticklabels=all_labels, yticklabels=all_labels,
    annot=False, cmap="viridis",
    cbar_kws={"label": "cosine similarity"},
)
plt.title("20x20 Full Similarity Matrix (EN + AR)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("cross_lingual_heatmap_20x20.png", dpi=150, bbox_inches="tight")
print("Saved cross_lingual_heatmap_20x20.png")