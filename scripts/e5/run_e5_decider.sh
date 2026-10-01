#!/bin/bash
# E5 closed loop for decider-4b (best E1 conflict score) and its zero-shot base Qwen3.5-4B.
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0
PY=~/anaconda3/envs/jev_uav/bin/python
M=/home/mydisk1/jev_uav/models
R=/home/mydisk1/jev_uav/results/e5
L=/home/mydisk1/jev_uav/logs
S="--styles third_person_semantic first_person_semantic"

$PY run_e5.py model --backend decider --model-path $M/decider-4b --model-id decider-4b --device cuda:0 \
    $S --output $R/decider-4b.jsonl > $L/e5_decider4b.log 2>&1 && echo DONE_decider-4b &
$PY run_e5.py model --backend decider --model-path $M/Qwen3.5-4B --model-id Qwen3.5-4B --device cuda:1 \
    $S --output $R/Qwen3.5-4B.jsonl > $L/e5_qwen.log 2>&1 && echo DONE_Qwen3.5-4B &
wait

$PY run_e5.py analyze $R/baselines.jsonl $R/JevK5-9B.jsonl $R/decider-4b.jsonl $R/Qwen3.5-4B.jsonl \
    --output $R/e5_table.md > $L/e5_analyze.log 2>&1
echo ALL_DONE
