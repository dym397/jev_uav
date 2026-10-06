#!/bin/bash
# E5 closed loop for the official Jev API (jev-official) with first/third_person_semantic, run on this machine (no
# GPU; the key stays here). Same endpoint, request and tolerance as its E1 runs. Set JEV_API_KEY in the environment
# first; it is read from there only. --max-invalid 3 stops a style at the fourth failed call (e.g. HTTP 402 when the
# balance runs out) instead of flying the rest on the MAINTAIN fallback; rerun to resume from the last whole episode.
# usage: bash scripts/e5/run_e5_official.sh      (from the repository root)
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=python
[ -n "${JEV_API_KEY:-}" ] || { echo "Set JEV_API_KEY first."; exit 2; }
mkdir -p results/e5 logs
for style in first_person_semantic third_person_semantic; do
  $PY run_e5.py model --backend systemone --endpoint https://jevtypesafeai.com/api/v1/decide --model-id jev-official \
      --request-model "" --api-key-env JEV_API_KEY --sum-tolerance 0.06 --max-invalid 3 --styles $style \
      --output results/e5/jev-official_$style.jsonl > logs/e5_jev-official_$style.log 2>&1 \
    && echo DONE_jev-official_$style || { echo "FAILED_jev-official_$style (see logs/e5_jev-official_$style.log)"; exit 1; }
done
echo ALL_DONE
