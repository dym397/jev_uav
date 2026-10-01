#!/bin/bash
# Run a command against kev's or Open-Jev's own /v1/systemone server, then stop the server.
# Each runs in its own conda env (their transformers/peft/torch pins clash with jev_uav's).
# The command sees ENDPOINT and REQUEST_MODEL (the model name that server accepts).
# usage: with_server.sh <kev-4b|kev-9b|Open-Jev-9B> <gpu> <port> <command...>
set -u
export HF_HUB_OFFLINE=1 USE_TF=0 HF_HOME=/home/mydisk1/jev_uav/hf_cache
M=/home/mydisk1/jev_uav/models
name=$1 gpu=$2 port=$3
shift 3

case $name in
  kev-*)
    # head.pt pins the base (Qwen3.5-{4,9}B-Base) by revision; bf16, CUDA graphs and fused kernels are serve defaults.
    CUDA_VISIBLE_DEVICES=$gpu ~/anaconda3/envs/kev/bin/python -m kev.serve --run $M/$name --port $port &
    ready=/v1/models; REQUEST_MODEL=kev-latest ;;
  Open-Jev-*)
    # model.json pins Qwen/Qwen3.5-9B by revision; the server accepts only its own model names.
    CUDA_VISIBLE_DEVICES=$gpu ~/anaconda3/envs/openjev/bin/python -m jev.server \
        --checkpoint $M/$name/package/checkpoint --device cuda:0 --batch-size 1 --port $port &
    ready=/health; REQUEST_MODEL=open-jev ;;
  *) echo "unknown model $name"; exit 2 ;;
esac
server=$!
trap 'kill $server' EXIT
until curl -sf 127.0.0.1:$port$ready > /dev/null; do
  kill -0 $server 2> /dev/null || { echo "server for $name exited"; exit 1; }
  sleep 5
done
export ENDPOINT=http://127.0.0.1:$port/v1/systemone REQUEST_MODEL
"$@"
