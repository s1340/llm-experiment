import os, glob, json
from collections import defaultdict

DATA_DIR = r"G:\LLM\experiment\data\scale_runs"

def main():
    meta_files = sorted(glob.glob(os.path.join(DATA_DIR, "meta_chunk_*.jsonl")))
    if not meta_files:
        raise RuntimeError("No meta_chunk_*.jsonl found")

    rows = []
    for mp in meta_files:
        with open(mp, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

    print("Total rows:", len(rows))

    # Group by exact prompt text
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r.get("task_prompt","")].append(r)

    dup_groups = [(p, items) for p, items in by_prompt.items() if len(items) >= 2]
    dup_groups.sort(key=lambda x: -len(x[1]))

    print("Duplicate prompt groups:", len(dup_groups))
    for p, items in dup_groups[:10]:
        labels = sorted(set(i["label"] for i in items))
        ids = sorted(set(i["task_id"] for i in items))
        print("\n--- DUP PROMPT (count =", len(items), ") ---")
        print("Labels:", labels)
        print("Task IDs:", ids)
        print("Prompt:", p)

if __name__ == "__main__":
    main()