import os, glob, json, sys
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Usage:
#   python 17_breakout_layer_analysis.py <DATA_DIR> [THRESHOLD]
#
# Examples:
#   python 17_breakout_layer_analysis.py G:\LLM\experiment\data\scale_runs_qwen 0.80
#   python 17_breakout_layer_analysis.py G:\LLM\experiment\data\scale_runs_gemma
#
# For each (pair, seed), finds the shallowest layer where Macro-F1 >= THRESHOLD.
# Reports mean±std of raw layer index and proportional depth across seeds 0-4.

DATA_DIR  = sys.argv[1] if len(sys.argv) > 1 else r"G:\LLM\experiment\data\scale_runs_qwen"
THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.80
SEEDS     = [0, 1, 2, 3, 4]
PAIRS     = [("R", "N"), ("R", "A"), ("A", "N")]

def normalize_label(lbl):
    l = (lbl or "").strip().lower()
    if l in ["routine", "r"]:         return "R"
    if l in ["nonroutine", "non-routine", "conceptual", "n"]: return "N"
    if l in ["ambiguous", "a"]:       return "A"
    raise ValueError(f"Unknown label: {lbl!r}")

def load_all(data_dir):
    pt_files   = sorted(glob.glob(os.path.join(data_dir, "hidden_chunk_*.pt")))
    meta_files = sorted(glob.glob(os.path.join(data_dir, "meta_chunk_*.jsonl")))
    if not pt_files:
        raise RuntimeError(f"No chunk files in {data_dir}")
    X_list, labels, prompts = [], [], []
    for pt_path, meta_path in zip(pt_files, meta_files):
        x     = torch.load(pt_path).numpy()
        metas = [json.loads(line) for line in open(meta_path, encoding="utf-8")]
        X_list.append(x)
        labels.extend([normalize_label(m["label"]) for m in metas])
        prompts.extend([m.get("task_prompt", "") for m in metas])
    return np.concatenate(X_list, axis=0), np.array(labels), np.array(prompts)

def split_by_prompt(prompts, train_frac=0.7, seed=0):
    unique = sorted(set(prompts.tolist()))
    rng    = np.random.default_rng(seed)
    rng.shuffle(unique)
    split        = max(1, int(train_frac * len(unique)))
    train_set    = set(unique[:split])
    test_set     = set(unique[split:])
    train_mask   = np.array([p in train_set for p in prompts])
    test_mask    = np.array([p in test_set  for p in prompts])
    return train_mask, test_mask

def main():
    X_all, labels_all, prompts_all = load_all(DATA_DIR)
    T, L, H = X_all.shape

    print(f"Data dir   : {DATA_DIR}")
    print(f"Shape      : {T} examples, {L} layers, {H} hidden dim")
    print(f"Threshold  : F1 >= {THRESHOLD}")
    print(f"Seeds      : {SEEDS}")
    print()

    all_results = {}

    for PAIR_A, PAIR_B in PAIRS:
        code = f"{PAIR_A}{PAIR_B}"
        keep = np.array([(lbl == PAIR_A or lbl == PAIR_B) for lbl in labels_all])
        X       = X_all[keep]
        labels  = labels_all[keep]
        prompts = prompts_all[keep]
        y       = np.array([0 if lbl == PAIR_A else 1 for lbl in labels], dtype=np.int64)

        breakout_layers = []
        never_reached   = 0

        for seed in SEEDS:
            train_mask, test_mask = split_by_prompt(prompts, train_frac=0.7, seed=seed)
            breakout = None
            for layer in range(L):
                feats = X[:, layer, :]
                clf   = LogisticRegression(max_iter=3000)
                clf.fit(feats[train_mask], y[train_mask])
                pred  = clf.predict(feats[test_mask])
                f1    = f1_score(y[test_mask], pred, average="macro", labels=[0, 1])
                if f1 >= THRESHOLD:
                    breakout = layer
                    break

            if breakout is None:
                never_reached += 1
                print(f"  {code} seed={seed}: NEVER reached F1={THRESHOLD:.2f}")
            else:
                prop = breakout / (L - 1)
                breakout_layers.append(breakout)
                print(f"  {code} seed={seed}: breakout layer={breakout:3d}  prop={prop:.3f}")

        if breakout_layers:
            arr       = np.array(breakout_layers, dtype=float)
            mean_l    = float(np.mean(arr))
            std_l     = float(np.std(arr))
            mean_p    = mean_l / (L - 1)
            std_p     = std_l  / (L - 1)
        else:
            mean_l = std_l = mean_p = std_p = None

        all_results[code] = dict(
            breakout_layers = breakout_layers,
            never_reached   = never_reached,
            n_seeds         = len(SEEDS),
            total_layers    = L,
            mean_layer      = mean_l,
            std_layer       = std_l,
            mean_prop       = mean_p,
            std_prop        = std_p,
        )

        if mean_l is not None:
            print(f"  {code} SUMMARY  breakout={mean_l:.1f} +/- {std_l:.1f}  "
                  f"prop={mean_p:.3f} +/- {std_p:.3f}  "
                  f"never={never_reached}/{len(SEEDS)}")
        else:
            print(f"  {code} SUMMARY  never reached threshold in any seed")
        print()

    # Final compact table
    print("=== BREAKOUT LAYER TABLE ===")
    print(f"{'Pair':<5}  {'Mean Layer':>10}  {'Std':>6}  {'Mean Prop':>10}  {'Std Prop':>9}  {'Never':>7}")
    print("-" * 57)
    for code in ["RN", "RA", "AN"]:
        r = all_results[code]
        if r["mean_layer"] is not None:
            print(f"{code:<5}  {r['mean_layer']:>10.2f}  {r['std_layer']:>6.2f}  "
                  f"{r['mean_prop']:>10.4f}  {r['std_prop']:>9.4f}  "
                  f"{r['never_reached']:>3}/{r['n_seeds']}")
        else:
            print(f"{code:<5}  {'n/a':>10}  {'n/a':>6}  {'n/a':>10}  {'n/a':>9}  "
                  f"{r['never_reached']:>3}/{r['n_seeds']}")
    print(f"\nTotal layers={L}  Threshold={THRESHOLD}  Divisor={L-1}")

if __name__ == "__main__":
    main()
