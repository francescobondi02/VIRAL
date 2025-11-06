# analyze_results.py
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ANALYSIS_DIR = "./outputs/analysis"
OUTPUT_DIR = "./outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_results():
    # Cerca solo i file con pattern "_<start>_<end>.json" e NON quelli "partial"
    pattern = os.path.join(ANALYSIS_DIR, "feature_coherence_summary_*_*[0-9].json")
    files = sorted(
        [f for f in glob.glob(pattern) if "partial" not in os.path.basename(f)]
    )

    if not files:
        raise RuntimeError("No interval JSON results found.")
    print(f"[INFO] Found {len(files)} interval result files:")
    for f in files:
        print("   ", os.path.basename(f))

    data = []
    for path in files:
        with open(path, "r") as f:
            try:
                data.extend(json.load(f))
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON: {path}")
    print(f"[INFO] Loaded {len(data)} total object entries.")
    return pd.DataFrame(data)


def summarize(df: pd.DataFrame):
    print("\n===== GLOBAL STATS =====")
    print(f"Total objects: {len(df)}")
    print(f"Unique images: {df['image_id'].nunique()}")
    print(f"Unique categories: {df['object'].nunique()}")
    print(f"Mean contrast: {df['contrast'].mean():.4f}")
    print(f"Std contrast:  {df['contrast'].std():.4f}")
    print(f"Median contrast: {df['contrast'].median():.4f}")

    print("\n===== TOP / BOTTOM CATEGORIES =====")
    cat_means = df.groupby("object")["contrast"].mean().sort_values(ascending=False)
    print("Top 10:")
    print(cat_means.head(10))
    print("\nBottom 10:")
    print(cat_means.tail(10))


def make_plots(df: pd.DataFrame):
    print("\n[INFO] Generating plots...")

    # Histogram of contrast
    plt.figure(figsize=(6, 4))
    plt.hist(df["contrast"], bins=60, color="steelblue", alpha=0.8)
    plt.xlabel("Contrast (intra - inter)")
    plt.ylabel("Count")
    plt.title("Distribution of feature contrast")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "contrast_histogram.png"))
    plt.close()

    # Scatter intra vs inter
    plt.figure(figsize=(6, 6))
    plt.scatter(df["inter"], df["intra"], alpha=0.2, s=5, color="darkorange")
    plt.xlabel("Inter-similarity")
    plt.ylabel("Intra-similarity")
    plt.title("Intra vs Inter similarity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "intra_vs_inter.png"))
    plt.close()

    # Category-wise bar plot
    cat_means = df.groupby("object")["contrast"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 4))
    cat_means.head(20).plot(kind="bar", color="seagreen")
    plt.ylabel("Mean contrast")
    plt.title("Top 20 categories by mean contrast")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top20_categories.png"))
    plt.close()

    print(f"[INFO] Plots saved in {OUTPUT_DIR}")


def main():
    df = load_all_results()
    if df.empty:
        print("[WARN] No data loaded.")
        return
    summarize(df)
    make_plots(df)


if __name__ == "__main__":
    main()
