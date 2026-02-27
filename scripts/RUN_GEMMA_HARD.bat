@echo off
setlocal
call G:\miniconda_installed\Scripts\activate.bat
call conda activate llmstate
python -c "import torch; print('TORCH OK', torch.__version__)"
cd /d G:\LLM\experiment\scripts

echo === GEMMA (hard tasks): clearing old chunks ===
del /q G:\LLM\experiment\data\scale_runs_gemma\hidden_chunk_*.pt 2>nul
del /q G:\LLM\experiment\data\scale_runs_gemma\meta_chunk_*.jsonl 2>nul

echo === GEMMA: extraction ===
python 05_extract_hidden_states_gemma.py

echo === GEMMA: probes ===
python 06_probe_routine_vs_other.py G:\LLM\experiment\data\scale_runs_gemma
python 07_probe_task_holdout.py     G:\LLM\experiment\data\scale_runs_gemma
python 09_probe_prompt_holdout.py   G:\LLM\experiment\data\scale_runs_gemma

echo === DONE (GEMMA) ===
pause