#!/usr/bin/env bash
# Run pairwise probes for seeds 1-4 across all three models.
# Seed 0 already exists; this fills in the remaining seeds.

CONDA="/c/Users/User/miniforge3/Scripts/conda.exe"
PY="$CONDA run -p G:/miniconda_installed/envs/llmstate python"
SCRIPT="G:/LLM/experiment/scripts/15_probe_pairwise_prompt_holdout.py"
SUMMARIZE="G:/LLM/experiment/scripts/16_summarize_pairwise_multiseed.py"

declare -A DATA_DIRS
DATA_DIRS[qwen]="G:/LLM/experiment/data/scale_runs_qwen"
DATA_DIRS[gemma]="G:/LLM/experiment/data/scale_runs_gemma"
DATA_DIRS[llama]="G:/LLM/experiment/data/scale_runs_llama"

declare -A RESULT_DIRS
RESULT_DIRS[qwen]="G:/LLM/experiment/results/pairwise_qwen"
RESULT_DIRS[gemma]="G:/LLM/experiment/results/pairwise_gemma"
RESULT_DIRS[llama]="G:/LLM/experiment/results/pairwise_llama"

SEEDS=(1 2 3 4)
PAIRS=("R,N" "R,A" "A,N")
PAIR_CODES=("RN" "RA" "AN")

TOTAL=0
DONE=0

for model in qwen gemma llama; do
    for seed in "${SEEDS[@]}"; do
        for i in 0 1 2; do
            TOTAL=$((TOTAL + 1))
        done
    done
done

echo "=== Pairwise multi-seed runner ==="
echo "Models: qwen, gemma, llama | Seeds: 1-4 | Pairs: RN, RA, AN"
echo "Total runs: $TOTAL"
echo ""

for model in qwen gemma llama; do
    DATA_DIR="${DATA_DIRS[$model]}"
    RESULT_DIR="${RESULT_DIRS[$model]}"

    echo "--- Model: $model ---"

    for seed in "${SEEDS[@]}"; do
        for i in 0 1 2; do
            pair="${PAIRS[$i]}"
            code="${PAIR_CODES[$i]}"
            outfile="$RESULT_DIR/seed${seed}_${code}.txt"

            DONE=$((DONE + 1))
            echo "[$DONE/$TOTAL] $model seed=$seed pair=$pair -> $outfile"

            $PY "$SCRIPT" "$DATA_DIR" "$seed" "$pair" > "$outfile" 2>&1
            status=$?
            if [ $status -ne 0 ]; then
                echo "  ERROR (exit $status) — check $outfile"
            else
                # Pull the best macro-F1 line for a quick sanity check
                grep "Best layer by Macro-F1:" "$outfile" | tail -1 | sed 's/^/  /'
            fi
        done
    done

    echo "  Running summary for $model..."
    $PY "$SUMMARIZE" "$RESULT_DIR" 2>&1
    echo ""
done

echo "=== All done ==="
