#!/bin/bash
# E1 prompt matrix (the 9 styles of run_e1_matrix.sh) for the replications added on 2026-10-03, each through its
# official server (see scripts/with_server.sh): imajev-9b on GPU0 alongside Wald-4B then CLM-v0.1-8B on GPU1.
# Each run resumes from its records.jsonl. LIMIT=N runs only the first N probes (smoke run).
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

served() {  # served <model> <gpu> <port>
  bash scripts/with_server.sh $1 $2 $3 sh -c '"$0" run_e1.py run --backend systemone --endpoint "$ENDPOINT" \
      --request-model "$REQUEST_MODEL" --model-id "$1" --probes "$2" --styles '"$styles"' --output-dir "$3/$1" '"$limit" \
      $PY $1 $P $C > $L/e1m_$1.log 2>&1 && echo DONE_$1
}

served imajev-9b 0 8765 &
( served Wald-4B 1 8100; served CLM-v0.1-8B 1 8700 ) &
wait
echo ALL_DONE
