#!/bin/bash
# E1 full run on the fixed probe set; each run resumes if interrupted.
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0
PY=~/anaconda3/envs/jev_uav/bin/python
M=/home/mydisk1/jev_uav/models
R=/home/mydisk1/jev_uav/results/e1
L=/home/mydisk1/jev_uav/logs
P=$R/probes.jsonl

(
  $PY run_e1.py run --backend jevk5 --model-path $M/JevK5-9B --model-id JevK5-9B \
      --device cuda:0 --probes $P --output-dir $R/JevK5-9B && echo DONE_JevK5-9B
) > $L/e1_gpu0.log 2>&1 &

(
  $PY run_e1.py run --backend nano --checkpoint-dir $M/NanoJev-unified --model-id NanoJev-unified-games-v1 \
      --device cuda:1 --probes $P --output-dir $R/NanoJev && echo DONE_NanoJev
  CUDA_VISIBLE_DEVICES=1 $PY run_e1.py run --backend laya --model-path $M/laya --model-id laya \
      --probes $P --output-dir $R/laya && echo DONE_laya
  CUDA_VISIBLE_DEVICES=1 $PY run_e1.py run --backend laya --model-path $M/laya-typed-decisions \
      --model-id laya-typed-decisions --probes $P --output-dir $R/laya-typed-decisions && echo DONE_laya-td
) > $L/e1_gpu1.log 2>&1 &

wait
$PY run_e1.py analyze --probes $P $R/JevK5-9B $R/NanoJev $R/laya $R/laya-typed-decisions \
    --output $R/e1_table.md > $L/e1_analyze.log 2>&1
echo ALL_DONE
