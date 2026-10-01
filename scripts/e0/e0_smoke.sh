#!/bin/bash
# E0 end-to-end smoke: every model, pilot scenarios, three prompt styles.
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0
PY=~/anaconda3/envs/jev_uav/bin/python
M=/home/mydisk1/jev_uav/models
OUT=/home/mydisk1/jev_uav/results/e0_smoke
STYLES="paper14 first_person_polar third_person_polar"
COMMON="--split smoke --state-styles $STYLES --question-styles balanced --max-steps 30"

(
  $PY benchmark_open_models.py --backend jevk5 --model-path $M/JevK5-9B --model-id JevK5-9B \
      --device cuda:0 $COMMON --output-dir $OUT/JevK5-9B && echo DONE_JevK5-9B
) > /home/mydisk1/jev_uav/logs/e0_jevk5.log 2>&1 &

(
  $PY benchmark_open_models.py --backend nano --checkpoint-dir $M/NanoJev-unified \
      --device cuda:1 $COMMON --output-dir $OUT/NanoJev && echo DONE_NanoJev
  CUDA_VISIBLE_DEVICES=1 $PY benchmark_open_models.py --backend laya --model-path $M/laya \
      --model-id laya $COMMON --output-dir $OUT/laya && echo DONE_laya
  CUDA_VISIBLE_DEVICES=1 $PY benchmark_open_models.py --backend laya \
      --model-path $M/laya-typed-decisions --model-id laya-typed-decisions $COMMON \
      --output-dir $OUT/laya-typed-decisions && echo DONE_laya-typed-decisions
) > /home/mydisk1/jev_uav/logs/e0_gpu1.log 2>&1 &

wait
echo ALL_DONE
