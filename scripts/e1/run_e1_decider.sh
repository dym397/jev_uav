#!/bin/bash
# E1 semantic styles: decider family plus a zero-shot letter-logit control.
# Qwen3.5-4B (stock, no Jev training) is read by the same decider-ai prompt and readout,
# so decider-4b vs Qwen3.5-4B isolates what the Jev-style training adds.
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0 HF_HOME=/home/mydisk1/jev_uav/hf_cache
PY=~/anaconda3/envs/jev_uav/bin/python
M=/home/mydisk1/jev_uav/models
R=/home/mydisk1/jev_uav/results/e1
C=$R/semantic
L=/home/mydisk1/jev_uav/logs
P=$R/probes.jsonl
S="--styles first_person_semantic third_person_semantic"

run() {  # run <model-dir> <device>
  $PY run_e1.py run --backend decider --model-path $M/$1 --model-id $1 --device $2 \
      --probes $P $S --output-dir $C/$1 && echo DONE_$1
}

( run decider-4b cuda:0; run decider-0.8b cuda:0 ) > $L/e1dec_gpu0.log 2>&1 &
( run decider-2b cuda:1; run Qwen3.5-4B cuda:1 ) > $L/e1dec_gpu1.log 2>&1 &
wait

$PY run_e1.py analyze --probes $P $C/JevK5-9B $C/decider-4b $C/decider-2b $C/decider-0.8b $C/Qwen3.5-4B \
    $C/laya $C/laya-typed-decisions $C/NanoJev --output $C/e1_semantic_decider_table.md > $L/e1dec_analyze.log 2>&1
echo ALL_DONE
