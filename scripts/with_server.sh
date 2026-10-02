#!/bin/bash
# Run a command against kev's or Open-Jev's own /v1/systemone server, then stop the server.
# Each runs in its own conda env (their transformers/peft/torch pins clash with jev_uav's).
# The command sees ENDPOINT and REQUEST_MODEL (the model name that server accepts).
# usage: with_server.sh <kev-4b|kev-9b|Open-Jev-9B> <gpu> <port> <command...>
# gpu=all: a model past one card over both GPUs. kev-27b goes through scripts/kev_serve_sharded.py (bf16, layers
# past the GPUs in host memory); Open-Jev-27B in 8-bit (bf16 does not fit 48 GB and its loader refuses offload).
set -u
export HF_HUB_OFFLINE=1 USE_TF=0 HF_HOME=/home/mydisk1/jev_uav/hf_cache
M=/home/mydisk1/jev_uav/models
name=$1 gpu=$2 port=$3
shift 3
code=$(cd "$(dirname "$0")/.." && pwd)

case $name in
  kev-*)
    if [ "$gpu" = all ]; then
      KEV_CUDA_GRAPHS=0 KEV_FUSED=0 ~/anaconda3/envs/kev/bin/python $code/scripts/kev_serve_sharded.py --run $M/$name --port $port &
      ready=/v1/models; REQUEST_MODEL=kev-latest
    else
    # head.pt pins the base (Qwen3.5-{4,9}B-Base) by revision; bf16, CUDA graphs and fused kernels are serve defaults.
    CUDA_VISIBLE_DEVICES=$gpu ~/anaconda3/envs/kev/bin/python -m kev.serve --run $M/$name --port $port &
    ready=/v1/models; REQUEST_MODEL=kev-latest
    fi ;;
  Open-Jev-*)
    if [ "$gpu" = all ]; then
      JEV_LOAD_8BIT=1 JEV_DEVICE_MAP=sequential JEV_MAX_MEMORY="0=20GiB,1=22GiB" \
        ~/anaconda3/envs/openjev/bin/python -m jev.server \
        --checkpoint $M/$name/package/checkpoint --device cuda:0 --batch-size 1 --port $port &
    else
    # model.json pins Qwen/Qwen3.5-9B by revision; the server accepts only its own model names.
    CUDA_VISIBLE_DEVICES=$gpu ~/anaconda3/envs/openjev/bin/python -m jev.server \
        --checkpoint $M/$name/package/checkpoint --device cuda:0 --batch-size 1 --port $port &
    fi
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
