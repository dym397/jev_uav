#!/bin/bash
# E1 semantic styles for the checkpoints past one card, one at a time over both GPUs:
#   kev-27b          bf16, layers past the GPUs in host memory (scripts/kev_serve_sharded.py)
#   Open-Jev-27B     8-bit (bf16 needs 54 GB; its loader refuses offload)
#   decider-35b-a3b  bf16 MoE (bnb does not quantize its fused experts), layers past the GPUs in host memory
#   decider-12b      bf16 over both GPUs
# LIMIT=N runs only the first N probes (smoke run).
cd /home/mydisk1/jev_uav/code
PY=~/anaconda3/envs/jev_uav/bin/python
R=/home/mydisk1/jev_uav/results/e1
C=$R/semantic
L=/home/mydisk1/jev_uav/logs
export HF_HUB_OFFLINE=1 USE_TF=0 HF_HOME=/home/mydisk1/jev_uav/hf_cache
limit=${LIMIT:+--limit $LIMIT}
styles="first_person_semantic third_person_semantic"

served() {  # served <model> <port>
  bash scripts/with_server.sh $1 all $2 sh -c '"$0" run_e1.py run --backend systemone --endpoint "$ENDPOINT" \
      --request-model "$REQUEST_MODEL" --model-id "$1" --probes "$2" --styles '"$styles"' --output-dir "$3/$1" '"$limit" \
      $PY $1 $R/probes.jsonl $C > $L/e1_$1.log 2>&1 && echo DONE_$1
}

in_process() {  # in_process <decider model> [UAV_MAX_MEMORY]
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True UAV_MAX_MEMORY=${2:-} $PY run_e1.py run --backend decider --model-path ../models/$1 --model-id $1 --device sharded \
      --probes $R/probes.jsonl --styles $styles --output-dir $C/$1 $limit > $L/e1_$1.log 2>&1 && echo DONE_$1
}

in_process decider-12b "0=16GiB,1=22GiB"
served kev-27b 8008
served Open-Jev-27B 8791
in_process decider-35b-a3b "0=17GiB,1=18GiB,cpu=100GiB"  # headroom for the fused-expert concat at load; ~13 s/call
echo ALL_DONE
