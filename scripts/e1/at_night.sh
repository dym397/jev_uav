#!/bin/bash
# Run a command once the GPUs are free and it is past 20:00 (daytime runs stay ahead of the night queue).
# usage: at_night.sh <command...>
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] \
      && [ "$(date +%H)" -ge 20 ]; do
  sleep 60
done
date
exec "$@"
