#!/bin/bash
# E1 control: semantic styles (graded words, no numbers), same probe set.
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0
PY=~/anaconda3/envs/jev_uav/bin/python
M=/home/mydisk1/jev_uav/models
R=/home/mydisk1/jev_uav/results/e1
C=$R/semantic
L=/home/mydisk1/jev_uav/logs
P=$R/probes.jsonl
S="--styles first_person_semantic third_person_semantic"

(
  $PY run_e1.py run --backend jevk5 --model-path $M/JevK5-9B --model-id JevK5-9B \
      --device cuda:0 --probes $P $S --output-dir $C/JevK5-9B && echo DONE_JevK5-9B
) > $L/e1sem_gpu0.log 2>&1 &

(
  $PY run_e1.py run --backend nano --checkpoint-dir $M/NanoJev-unified --model-id NanoJev-unified-games-v1 \
      --device cuda:1 --probes $P $S --output-dir $C/NanoJev && echo DONE_NanoJev
  CUDA_VISIBLE_DEVICES=1 $PY run_e1.py run --backend laya --model-path $M/laya --model-id laya \
      --probes $P $S --output-dir $C/laya && echo DONE_laya
  CUDA_VISIBLE_DEVICES=1 $PY run_e1.py run --backend laya --model-path $M/laya-typed-decisions \
      --model-id laya-typed-decisions --probes $P $S --output-dir $C/laya-typed-decisions && echo DONE_laya-td
) > $L/e1sem_gpu1.log 2>&1 &

wait
$PY run_e1.py analyze --probes $P $C/JevK5-9B $C/NanoJev $C/laya $C/laya-typed-decisions \
    --output $C/e1_semantic_table.md > $L/e1sem_analyze.log 2>&1
echo ALL_DONE
