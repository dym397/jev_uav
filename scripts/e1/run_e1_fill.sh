#!/bin/bash
# Fill the gaps of the E1 state-BA heatmaps (scripts/e1/plot_state_ba.py) with the same backends that made each
# model's other cells: the 9 matrix styles for the early models (into matrix/, or their top-level dir for NanoJev and
# laya), the 2 semantic styles for the later ones (into semantic/). Every run resumes from its records.jsonl.
# usage: run_e1_fill.sh small | pplx | 35b      (small: both GPUs in two lanes; pplx and 35b: both GPUs, sharded)
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0 HF_HOME=/home/mydisk1/jev_uav/hf_cache
PY=~/anaconda3/envs/jev_uav/bin/python
M=/home/mydisk1/jev_uav/models
R=/home/mydisk1/jev_uav/results/e1
L=/home/mydisk1/jev_uav/logs
P=$R/probes.jsonl
MATRIX="first_person_polar third_person_polar paper14 paper14_prose paper14_semantic
        first_person_derived first_person_clock first_person_world first_person_list"
MATRIX=$(echo $MATRIX)
SEMANTIC="first_person_semantic third_person_semantic"

served() {  # served <model> <gpu|all> <port> <out dir> <styles...>
  local model=$1 gpu=$2 port=$3 out=$4; shift 4
  bash scripts/with_server.sh $model $gpu $port sh -c '"$0" run_e1.py run --backend systemone --endpoint "$ENDPOINT" \
      --request-model "$REQUEST_MODEL" --model-id "$1" --probes "$2" --styles '"$*"' --output-dir "$3/$1"' \
      $PY $model $P $out > $L/e1fill_$model.log 2>&1 && echo DONE_$model
}
decider() {  # decider <model> <device> [UAV_MAX_MEMORY]
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True UAV_MAX_MEMORY=${3:-} $PY run_e1.py run --backend decider \
      --model-path $M/$1 --model-id $1 --device $2 --probes $P --styles $MATRIX --output-dir $R/matrix/$1 \
      > $L/e1fill_$1.log 2>&1 && echo DONE_$1
}
nano() {
  $PY run_e1.py run --backend nano --checkpoint-dir $M/NanoJev-unified --model-id NanoJev-unified-games-v1 \
      --device cuda:1 --probes $P --styles $MATRIX --output-dir $R/NanoJev > $L/e1fill_NanoJev.log 2>&1 && echo DONE_NanoJev
}
laya() {  # laya <model>
  CUDA_VISIBLE_DEVICES=1 $PY run_e1.py run --backend laya --model-path $M/$1 --model-id $1 \
      --probes $P --styles $MATRIX --output-dir $R/$1 > $L/e1fill_$1.log 2>&1 && echo DONE_$1
}

case $1 in
  small)
    ( decider decider-0.8b cuda:0; decider Qwen3.5-4B cuda:0; served kev-4b 0 8008 $R/matrix $MATRIX
      served imajev-9b 0 8765 $R/semantic $SEMANTIC; served JPT-9B 0 8080 $R/semantic $SEMANTIC
      served Winnow-12B 0 8091 $R/semantic $SEMANTIC ) &
    ( nano; laya laya; laya laya-typed-decisions; decider decider-2b cuda:1
      served Wald-4B 1 8100 $R/semantic $SEMANTIC; served CLM-v0.1-8B 1 8700 $R/semantic $SEMANTIC
      served Decision-2.0-Lux-9B 1 8790 $R/semantic $SEMANTIC ) &
    wait ;;
  pplx) served pplx-decider-v1-27b all 8800 $R/semantic $SEMANTIC ;;
  35b) decider decider-35b-a3b sharded "0=17GiB,1=18GiB,cpu=100GiB" ;;   # ~12.5 s/call
  *) echo "usage: $0 small|pplx|35b"; exit 2 ;;
esac
echo ALL_DONE
