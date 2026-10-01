#!/bin/bash
# E1 semantic styles for kev (jaredpalmer/kev) and Open-Jev (ZefanCai/Open-Jev), each behind its own server.
cd /home/mydisk1/jev_uav/code
PY=~/anaconda3/envs/jev_uav/bin/python
R=/home/mydisk1/jev_uav/results/e1
C=$R/semantic
L=/home/mydisk1/jev_uav/logs

run() {  # run <model> <gpu> <port>
  bash scripts/with_server.sh $1 $2 $3 sh -c '"$0" run_e1.py run --backend systemone --endpoint "$ENDPOINT" \
      --request-model "$REQUEST_MODEL" --model-id "$1" --probes "$2" \
      --styles first_person_semantic third_person_semantic --output-dir "$3/$1"' $PY $1 $R/probes.jsonl $C \
      > $L/e1_$1.log 2>&1 && echo DONE_$1
}

( run kev-9b 1 8009; run Open-Jev-9B 1 8791 ) &
run kev-4b 0 8008 &
wait

$PY run_e1.py analyze --probes $R/probes.jsonl $C/JevK5-9B $C/decider-4b $C/kev-9b $C/kev-4b $C/Open-Jev-9B \
    $C/decider-2b $C/decider-0.8b $C/Qwen3.5-4B $C/laya $C/laya-typed-decisions $C/NanoJev \
    --output $C/e1_semantic_all_table.md > $L/e1_all_analyze.log 2>&1
echo ALL_DONE
