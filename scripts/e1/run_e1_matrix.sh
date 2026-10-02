#!/bin/bash
# E1 prompt matrix (step 2/3): the polar pair, the paper14 equal-information chain and the
# frame/format variants, for the strong checkpoints that fit one card (daytime queue).
# decider-12b runs last over both GPUs. Each run resumes from its records.jsonl.
# LIMIT=N runs only the first N probes (smoke run).
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0 HF_HOME=/home/mydisk1/jev_uav/hf_cache
PY=~/anaconda3/envs/jev_uav/bin/python
M=/home/mydisk1/jev_uav/models
R=/home/mydisk1/jev_uav/results/e1
C=$R/matrix
L=/home/mydisk1/jev_uav/logs
P=$R/probes.jsonl
limit=${LIMIT:+--limit $LIMIT}
styles="first_person_polar third_person_polar paper14 paper14_prose paper14_semantic
        first_person_derived first_person_clock first_person_world first_person_list"
styles=$(echo $styles)

served() {  # served <model> <gpu> <port>
  bash scripts/with_server.sh $1 $2 $3 sh -c '"$0" run_e1.py run --backend systemone --endpoint "$ENDPOINT" \
      --request-model "$REQUEST_MODEL" --model-id "$1" --probes "$2" --styles '"$styles"' --output-dir "$3/$1" '"$limit" \
      $PY $1 $P $C > $L/e1m_$1.log 2>&1 && echo DONE_$1
}

decider() {  # decider <model> <device> [UAV_MAX_MEMORY]
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True UAV_MAX_MEMORY=${3:-} $PY run_e1.py run --backend decider \
      --model-path $M/$1 --model-id $1 --device $2 --probes $P --styles $styles --output-dir $C/$1 $limit \
      > $L/e1m_$1.log 2>&1 && echo DONE_$1
}

jevk5() {
  $PY run_e1.py run --backend jevk5 --model-path $M/JevK5-9B --model-id JevK5-9B --device cuda:0 \
      --probes $P --styles $styles --output-dir $C/JevK5-9B $limit > $L/e1m_JevK5-9B.log 2>&1 && echo DONE_JevK5-9B
}

( decider decider-4b cuda:0; served kev-9b 0 8009; jevk5 ) &
served Open-Jev-9B 1 8791 &
wait
decider decider-12b sharded "0=16GiB,1=22GiB"

$PY run_e1.py analyze --probes $P $C/decider-4b $C/kev-9b $C/JevK5-9B $C/Open-Jev-9B $C/decider-12b \
    --output $C/e1_matrix_table.md > $L/e1m_analyze.log 2>&1
echo ALL_DONE
