#!/bin/bash
# E5 closed loop for Open-Jev-9B (highest E1 argmax safe/good among Jev replications, order-invariant).
# One server per GPU, one state style each; the analyzer reads both files.
cd /home/mydisk1/jev_uav/code
PY=~/anaconda3/envs/jev_uav/bin/python
R=/home/mydisk1/jev_uav/results/e5
L=/home/mydisk1/jev_uav/logs

run() {  # run <gpu> <port> <style>
  bash scripts/with_server.sh Open-Jev-9B $1 $2 sh -c '"$0" run_e5.py model --backend systemone --endpoint "$ENDPOINT" \
      --request-model "$REQUEST_MODEL" --model-id Open-Jev-9B --styles "$1" --output "$2"' \
      $PY $3 $R/Open-Jev-9B_$3.jsonl > $L/e5_openjev_$3.log 2>&1 && echo DONE_$3
}

run 0 8790 third_person_semantic &
run 1 8791 first_person_semantic &
wait

$PY run_e5.py analyze $R/baselines.jsonl $R/JevK5-9B.jsonl $R/decider-4b.jsonl $R/Qwen3.5-4B.jsonl \
    $R/Open-Jev-9B_third_person_semantic.jsonl $R/Open-Jev-9B_first_person_semantic.jsonl \
    --output $R/e5_table.md > $L/e5_analyze.log 2>&1
echo ALL_DONE
