#!/bin/bash
# E5 closed loop for the five combinations picked from E1: Open-Jev-9B + first_person_derived, kev-27b +
# first/third_person_semantic, Open-Jev-27B and pplx-decider-v1-27b + first_person_semantic. Each model behind its own
# server (scripts/with_server.sh; the 27B ones over both GPUs), one model at a time. Episodes resume from the output.
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0 HF_HOME=/home/mydisk1/jev_uav/hf_cache
PY=~/anaconda3/envs/jev_uav/bin/python
R=/home/mydisk1/jev_uav/results/e5
L=/home/mydisk1/jev_uav/logs

run() {  # run <model> <gpu|all> <port> <style...>
  local model=$1 gpu=$2 port=$3; shift 3
  for style in "$@"; do
    bash scripts/with_server.sh $model $gpu $port sh -c '"$0" run_e5.py model --backend systemone --endpoint "$ENDPOINT" \
        --request-model "$REQUEST_MODEL" --model-id "$1" --styles "$2" --output "$3"' \
        $PY $model $style $R/${model}_$style.jsonl > $L/e5_${model}_$style.log 2>&1 && echo DONE_${model}_$style
  done
}

run Open-Jev-9B 0 8791 first_person_derived
run kev-27b all 8008 first_person_semantic third_person_semantic
run Open-Jev-27B all 8791 first_person_semantic
run pplx-decider-v1-27b all 8800 first_person_semantic

$PY run_e5.py analyze $R/baselines.jsonl $R/JevK5-9B.jsonl $R/decider-4b.jsonl $R/Qwen3.5-4B.jsonl \
    $R/Open-Jev-9B_third_person_semantic.jsonl $R/Open-Jev-9B_first_person_semantic.jsonl \
    $R/Open-Jev-9B_first_person_derived.jsonl $R/kev-27b_first_person_semantic.jsonl \
    $R/kev-27b_third_person_semantic.jsonl $R/Open-Jev-27B_first_person_semantic.jsonl \
    $R/pplx-decider-v1-27b_first_person_semantic.jsonl --output $R/e5_table.md > $L/e5_analyze.log 2>&1
echo ALL_DONE
