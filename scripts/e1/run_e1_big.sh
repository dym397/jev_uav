#!/bin/bash
# E1 prompt matrix (the 9 styles of run_e1_matrix.sh) for the 26-31B replications, one at a time over both GPUs
# (bf16, layers past the GPUs in host memory; see scripts/with_server.sh).
# usage: run_e1_big.sh <model>...   Each run resumes from its records.jsonl. LIMIT=N runs only the first N probes.
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0 HF_HOME=/home/mydisk1/jev_uav/hf_cache
PY=~/anaconda3/envs/jev_uav/bin/python
R=/home/mydisk1/jev_uav/results/e1
C=${OUT:-$R/matrix}
L=/home/mydisk1/jev_uav/logs
P=$R/probes.jsonl
limit=${LIMIT:+--limit $LIMIT}
styles="first_person_polar third_person_polar paper14 paper14_prose paper14_semantic
        first_person_derived first_person_clock first_person_world first_person_list"
styles=${STYLES:-$(echo $styles)}

for model in "$@"; do
  bash scripts/with_server.sh $model all 8800 sh -c '"$0" run_e1.py run --backend systemone --endpoint "$ENDPOINT" \
      --request-model "$REQUEST_MODEL" --model-id "$1" --probes "$2" --styles '"$styles"' --output-dir "$3/$1" '"$limit" \
      $PY $model $P $C > $L/e1m_$model.log 2>&1 && echo DONE_$model
done
echo ALL_DONE
