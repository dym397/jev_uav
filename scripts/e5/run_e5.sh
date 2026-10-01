#!/bin/bash
# E5 closed-loop pilot: JevK5-9B (semantic prompts) vs D3QN, constant and random baselines.
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0
PY=~/anaconda3/envs/jev_uav/bin/python
M=/home/mydisk1/jev_uav/models
R=/home/mydisk1/jev_uav/results/e5
L=/home/mydisk1/jev_uav/logs
mkdir -p $R

$PY run_e5.py baselines --d3qn-checkpoint $M/d3qn_baseline_1000/best.pt \
    --output $R/baselines.jsonl > $L/e5_baselines.log 2>&1 && echo DONE_baselines

$PY run_e5.py model --backend jevk5 --model-path $M/JevK5-9B --model-id JevK5-9B --device cuda:0 \
    --styles third_person_semantic first_person_semantic \
    --output $R/JevK5-9B.jsonl > $L/e5_jevk5.log 2>&1 && echo DONE_JevK5-9B

$PY run_e5.py analyze $R/baselines.jsonl $R/JevK5-9B.jsonl --output $R/e5_table.md > $L/e5_analyze.log 2>&1
echo ALL_DONE
