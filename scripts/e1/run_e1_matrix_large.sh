#!/bin/bash
# E1 prompt matrix (the 9 styles of run_e1_matrix.sh) for the 27B checkpoints, one at a time over both GPUs:
#   kev-27b       bf16, layers past the GPUs in host memory (~2.7 s/call, ~15 h)
#   Open-Jev-27B  8-bit (~2.4 s/call, ~13 h)
# Each run resumes from its records.jsonl. LIMIT=N runs only the first N probes (smoke run).
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0 HF_HOME=/home/mydisk1/jev_uav/hf_cache
PY=~/anaconda3/envs/jev_uav/bin/python
R=/home/mydisk1/jev_uav/results/e1
C=$R/matrix
L=/home/mydisk1/jev_uav/logs
P=$R/probes.jsonl
limit=${LIMIT:+--limit $LIMIT}
styles="first_person_polar third_person_polar paper14 paper14_prose paper14_semantic
        first_person_derived first_person_clock first_person_world first_person_list"
styles=$(echo $styles)

served() {  # served <model> <port>
  bash scripts/with_server.sh $1 all $2 sh -c '"$0" run_e1.py run --backend systemone --endpoint "$ENDPOINT" \
      --request-model "$REQUEST_MODEL" --model-id "$1" --probes "$2" --styles '"$styles"' --output-dir "$3/$1" '"$limit" \
      $PY $1 $P $C > $L/e1m_$1.log 2>&1 && echo DONE_$1
}

served kev-27b 8008
served Open-Jev-27B 8791

$PY run_e1.py analyze --probes $P $C/decider-4b $C/kev-9b $C/kev-27b $C/JevK5-9B $C/Open-Jev-9B $C/Open-Jev-27B \
    $C/decider-12b --output $C/e1_matrix_table.md > $L/e1m_analyze.log 2>&1
echo ALL_DONE
