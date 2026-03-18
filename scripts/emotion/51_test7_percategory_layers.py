"""
Per-category emotion direction breakdown at the key layers identified in script 50.

Focus layers: 1-3 (fear peak), 4 (sadness/disgust), 6-7 (happiness peak).
For each layer × emotion direction: self vs other diff and p-value, split by category.

No model loading needed — uses existing hidden states and direction files.
"""

import os, glob, json
import numpy as np
import torch
from scipy import stats

DATA_DIR    = r"G:\LLM\experiment\data\emotion"
RESULTS_DIR = r"G:\LLM\experiment\results\emotion"
DIRS_SUBDIR = os.path.join(RESULTS_DIR, "emotion_directions")

EMOTION_CATS   = ["threat", "existential", "praise", "harm_caused"]
DIRECTION_CATS = ["anger", "sadness", "happiness", "fear", "disgust"]
MODEL_KEY      = "llama"

FOCUS_LAYERS = {
    1: ["fear"],
    2: ["fear"],
    3: ["fear"],
    4: ["sadness", "disgust"],
    6: ["happiness"],
    7: ["happiness", "anger", "sadness"],
    8: ["sadness"],
}


def load_hidden_states(data_dir):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "test7_hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "test7_meta_chunk_*.jsonl")))
    X_list, metas = [], []
    for pt, mf in zip(pt_files, meta_files):
        X_list.append(torch.load(pt, weights_only=True).numpy())
        with open(mf, encoding="utf-8") as f:
            metas.extend(json.loads(line) for line in f)
    return np.concatenate(X_list, axis=0), metas


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled = np.sqrt(((n1-1)*np.var(a,ddof=1) + (n2-1)*np.var(b,ddof=1)) / (n1+n2-2))
    return float((np.mean(a) - np.mean(b)) / (pooled + 1e-12))


def main():
    data_dir = os.path.join(DATA_DIR, f"emotion_runs_{MODEL_KEY}")
    X, metas = load_hidden_states(data_dir)
    T, L, H  = X.shape
    print(f"Loaded: {T} × {L} × {H}\n")

    # Load emotion direction vectors for all layers
    dirs_arr = []
    for layer in range(L):
        p = os.path.join(DIRS_SUBDIR, f"{MODEL_KEY}_emotion_dirs_layer_{layer:03d}.npy")
        dirs_arr.append(np.load(p))
    dirs_arr = np.array(dirs_arr)  # [L, 5, H]

    cats = np.array([m["category"]  for m in metas])
    dirs = np.array([m["direction"] for m in metas])

    for layer, focus_dirs in sorted(FOCUS_LAYERS.items()):
        depth_pct = layer / (L - 1) * 100
        print(f"{'='*65}")
        print(f"Layer {layer} ({depth_pct:.1f}%)")
        print(f"{'='*65}")

        for dc in focus_dirs:
            dc_idx  = DIRECTION_CATS.index(dc)
            d_vec   = unit(dirs_arr[layer, dc_idx])

            # Overall
            s_all = X[dirs == "self",  layer, :] @ d_vec
            o_all = X[dirs == "other", layer, :] @ d_vec
            _, p_all = stats.ttest_ind(s_all, o_all)
            d_all    = cohens_d(s_all, o_all)
            diff_all = s_all.mean() - o_all.mean()
            print(f"\n  [{dc}]  overall: self={s_all.mean():+.4f}  other={o_all.mean():+.4f}  "
                  f"diff={diff_all:+.4f}  d={d_all:+.3f}  p={p_all:.4f}")

            # Per category
            for cat in EMOTION_CATS:
                s = (X[(dirs=="self")  & (cats==cat), layer, :] @ d_vec)
                o = (X[(dirs=="other") & (cats==cat), layer, :] @ d_vec)
                if len(s) < 2 or len(o) < 2:
                    continue
                _, p = stats.ttest_ind(s, o)
                d  = cohens_d(s, o)
                diff = s.mean() - o.mean()
                sig = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
                print(f"    {cat:12s}:  self={s.mean():+.4f}  other={o.mean():+.4f}  "
                      f"diff={diff:+.4f}  d={d:+.3f}  p={p:.4f}{sig}")

    # Bonus: happiness at all significant layers, per category, tabulated
    print(f"\n{'='*65}")
    print("Happiness direction: self−other diff across key layers × category")
    print(f"{'='*65}")
    hap_idx = DIRECTION_CATS.index("happiness")
    header = f"{'cat':12s}" + "".join(f"  L{l:02d}_diff  L{l:02d}_p" for l in [4,6,7,8,10])
    print(header)
    for cat in EMOTION_CATS:
        row = f"{cat:12s}"
        for layer in [4, 6, 7, 8, 10]:
            d_vec = unit(dirs_arr[layer, hap_idx])
            s = X[(dirs=="self")  & (cats==cat), layer, :] @ d_vec
            o = X[(dirs=="other") & (cats==cat), layer, :] @ d_vec
            if len(s) < 2 or len(o) < 2:
                row += "        n/a       n/a"
                continue
            _, p = stats.ttest_ind(s, o)
            diff = s.mean() - o.mean()
            row += f"  {diff:+.4f}  {p:.3f}"
        print(row)

    # And fear, layers 1-3
    print(f"\n{'='*65}")
    print("Fear direction: self−other diff across layers 1-3 × category")
    print(f"{'='*65}")
    fear_idx = DIRECTION_CATS.index("fear")
    header = f"{'cat':12s}" + "".join(f"  L0{l}_diff  L0{l}_p" for l in [1,2,3])
    print(header)
    for cat in EMOTION_CATS:
        row = f"{cat:12s}"
        for layer in [1, 2, 3]:
            d_vec = unit(dirs_arr[layer, fear_idx])
            s = X[(dirs=="self")  & (cats==cat), layer, :] @ d_vec
            o = X[(dirs=="other") & (cats==cat), layer, :] @ d_vec
            if len(s) < 2 or len(o) < 2:
                row += "        n/a       n/a"
                continue
            _, p = stats.ttest_ind(s, o)
            diff = s.mean() - o.mean()
            row += f"  {diff:+.4f}  {p:.3f}"
        print(row)


if __name__ == "__main__":
    main()
