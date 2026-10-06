#!/bin/bash
# E5 closed loop for imajev-9b and Decision-2.0-Lux-9B with first/third_person_semantic and first_person_derived:
# in E1 both pair consistent risk and action under the semantic styles and say "conflict" yet hold course under
# derived. One model per GPU, one style at a time. Episodes resume from the output. Then the E5 table over every arm
# on the server (the official Jev arms run locally through its API: scripts/e5/run_e5_official.sh).
cd /home/mydisk1/jev_uav/code
export HF_HUB_OFFLINE=1 USE_TF=0 HF_HOME=/home/mydisk1/jev_uav/hf_cache
PY=~/anaconda3/envs/jev_uav/bin/python
R=/home/mydisk1/jev_uav/results/e5
L=/home/mydisk1/jev_uav/logs
STYLES="first_person_semantic third_person_semantic first_person_derived"

run() {  # run <model> <gpu> <port>
  for style in $STYLES; do
    bash scripts/with_server.sh $1 $2 $3 sh -c '"$0" run_e5.py model --backend systemone --endpoint "$ENDPOINT" \
        --request-model "$REQUEST_MODEL" --model-id "$1" --styles "$2" --output "$3"' \
        $PY $1 $style $R/${1}_$style.jsonl > $L/e5_${1}_$style.log 2>&1 && echo DONE_${1}_$style
  done
}

run imajev-9b 0 8765 &
run Decision-2.0-Lux-9B 1 8790 &
wait

$PY run_e5.py analyze $R/baselines.jsonl $(ls $R/*.jsonl | grep -v baselines) --output $R/e5_table.md \
    > $L/e5_analyze.log 2>&1
echo ALL_DONE
