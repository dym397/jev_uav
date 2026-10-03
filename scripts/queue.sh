#!/bin/bash
# GPU job queue: runs the first line of logs/queue.txt, drops it, repeats; waits while the file is empty.
# Each line is one shell command run from code/ with its own log redirect. Append or reorder lines at any time;
# a line is removed only once it has started. Progress goes to logs/queue.log.
# usage: setsid nohup bash scripts/queue.sh > ../logs/queue.log 2>&1 < /dev/null & disown
cd /home/mydisk1/jev_uav/code
Q=/home/mydisk1/jev_uav/logs/queue.txt
touch $Q
while true; do
  job=$(sed -n '/[^[:space:]]/{p;q}' $Q)
  if [ -z "$job" ]; then sleep 60; continue; fi
  sed -i "0,/[^[:space:]]/{/[^[:space:]]/d}" $Q
  echo "$(date '+%F %T') START $job"
  bash -c "$job"
  echo "$(date '+%F %T') END rc=$? $job"
done
