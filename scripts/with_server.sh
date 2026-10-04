#!/bin/bash
# Run a command against kev's or Open-Jev's own /v1/systemone server, then stop the server.
# Each runs in its own conda env (their transformers/peft/torch pins clash with jev_uav's).
# The command sees ENDPOINT and REQUEST_MODEL (the model name that server accepts).
# usage: with_server.sh <kev-*|Open-Jev-*|imajev-*|Wald-4B|CLM-v0.1-8B|JPT-9B|Decision-2.0-Lux-9B|Winnow-12B> <gpu> <port> <command...>
# JPT and Lux run in env jev_sys1; Winnow is its own llama.cpp build (Q8_0 GGUF).
# imajev runs in env jev_imajev, Wald and CLM in jev_vllm (vLLM 0.30.0+cu129); Wald and CLM also take port+1 for vLLM.
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
  imajev-*)
    # Official torch backend, set up as for its published JevBench numbers (--fast --merge-lora --rotations 1 --calibration).
    # The LoRA r16 + readout adapter sits on Qwen/Qwen3.5-*B at the revision pinned in artifacts/model-qwen*b.json.
    X=/home/mydisk1/jev_uav/external/imajev
    size=${name#imajev-}
    adapter=$(ls -d $HF_HOME/hub/models--mohit67890--$name/snapshots/*/ | head -1)
    rev=$(sed -n 's/.*"revision": "\(.*\)".*/\1/p' $X/artifacts/model-qwen$size.json)
    bundle=/tmp/imajev_bundle_$size.json
    echo "{\"path\": \"$HF_HOME/hub/models--Qwen--Qwen3.5-${size^^}/snapshots/$rev\"}" > $bundle
    (cd $X && CUDA_VISIBLE_DEVICES=${gpu/all/0,1} PYTHONPATH=src:scripts exec ~/anaconda3/envs/jev_imajev/bin/python \
        scripts/playground/server.py --backend torch --model-bundle $bundle --adapter $adapter \
        --calibration $adapter/calibration.json --model-name $name --fast --merge-lora --rotations 1 --port $port) &
    ready=/v1/models; REQUEST_MODEL="" ;;
  Wald-*)
    # Official wald-serve: it starts its own vLLM 0.30.0 sidecar and reads serving.json (effort none, repeat_state_plain)
    # and temperature.json from the weights dir. Context capped at 8192 so the KV cache fits a 3090.
    weights=$(ls -d $HF_HOME/hub/models--org2ai--$name/snapshots/*/ | head -1)
    CUDA_VISIBLE_DEVICES=${gpu/all/0} VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false \
        ~/anaconda3/envs/jev_vllm/bin/wald-serve --model $weights --port $port --vllm-port $((port + 1)) \
        --max-model-len 8192 &
    ready=/health; REQUEST_MODEL="" ;;
  CLM-*)
    # Official pair: Qwen3-8B as a vLLM pooling encoder (serve_qwen3_8b.sh settings) behind clm-serve's InfoNCE heads.
    ckpt=$(ls $HF_HOME/hub/models--Contrastive-LM--$name/snapshots/*/CLM_*.pt | head -1)
    CUDA_VISIBLE_DEVICES=${gpu/all/0} VLLM_NO_USAGE_STATS=1 ~/anaconda3/envs/jev_vllm/bin/vllm serve Qwen/Qwen3-8B \
        --served-model-name qwen3-8b --runner pooling --enforce-eager --enable-prefix-caching --max-model-len 2048 \
        --gpu-memory-utilization 0.80 --max-num-seqs 32 --port $((port + 1)) &
    encoder=$!
    CUDA_VISIBLE_DEVICES=${gpu/all/0} ~/anaconda3/envs/jev_vllm/bin/clm-serve --port $port \
        --emb-url http://127.0.0.1:$((port + 1))/v1/embeddings --ckpt $ckpt &
    ready=/health; REQUEST_MODEL=clm-latest ;;
  JPT-*)
    # llm2jev's in-process HF backend (bf16) with the card's single temperature T = 1.087 and its default chat prompt.
    weights=$(ls -d $HF_HOME/hub/models--kirp--${name,,}/snapshots/*/ | head -1)
    CUDA_VISIBLE_DEVICES=${gpu/all/0,1} ~/anaconda3/envs/jev_sys1/bin/llm2jev --model $weights --backend hf \
        --temperature 1.087 --host 127.0.0.1 --port $port &
    ready=/health; REQUEST_MODEL="" ;;
  Decision-2.0-*)
    # The package ships no server; scripts/serve_lux.py wraps its system_one (runtime numerics: bf16, fp32 head).
    weights=$(ls -d $HF_HOME/hub/models--vllm-sr--$name/snapshots/*/ | head -1)
    CUDA_VISIBLE_DEVICES=${gpu/all/0} ~/anaconda3/envs/jev_sys1/bin/python $code/scripts/serve_lux.py $weights \
        --port $port &
    ready=/health; REQUEST_MODEL="" ;;
  Winnow-*)
    # Official winnow-server build (llama.cpp fork) with its CUDA profile; Q8_0 is the largest release that fits a
    # 3090, text-only (no vision projector).
    weights=$(ls $HF_HOME/hub/models--EldanRing--$name/snapshots/*/gguf/$name-Q8_0.gguf | head -1)
    (cd /home/mydisk1/jev_uav/external/winnow-inference && exec python3 scripts/serve.py --profile 5070ti-64k \
        --model $weights --text-only --gpu ${gpu/all/0} --port $port) &
    ready=/health; REQUEST_MODEL="" ;;
  *) echo "unknown model $name"; exit 2 ;;
esac
server=$!
trap 'kill $server ${encoder:-}' EXIT
# CLM's /health answers before its encoder is up, so wait for the encoder too.
until curl -sf 127.0.0.1:$port$ready > /dev/null \
      && { [ -z "${encoder:-}" ] || curl -sf 127.0.0.1:$((port + 1))/v1/models > /dev/null; }; do
  kill -0 $server 2> /dev/null || { echo "server for $name exited"; exit 1; }
  [ -z "${encoder:-}" ] || kill -0 $encoder 2> /dev/null || { echo "encoder for $name exited"; exit 1; }
  sleep 5
done
export ENDPOINT=http://127.0.0.1:$port/v1/systemone REQUEST_MODEL
"$@"
