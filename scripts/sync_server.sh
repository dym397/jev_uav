#!/bin/bash
# Push the local working tree (every file git would track) to the experiment server,
# then verify that each file's sha256 matches on both sides.
#
#   bash scripts/sync_server.sh            # copy + verify
#   bash scripts/sync_server.sh --check    # verify only, copy nothing
#   bash scripts/sync_server.sh --test     # copy + verify + run pytest on the server
#
# Server-only files are listed but never deleted.
set -euo pipefail
export LC_ALL=C

HOST=${JEV_SERVER:-cv1124@192.168.1.180}
REMOTE=${JEV_REMOTE_DIR:-/home/mydisk1/jev_uav/code}
PY=${JEV_REMOTE_PY:-'$HOME/anaconda3/envs/jev_uav/bin/python'}
SSH=(ssh -o BatchMode=yes "$HOST")

copy=1 test=0
for arg in "$@"; do
  case $arg in
    --check) copy=0 ;;
    --test) test=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"
list=$(mktemp) local_sums=$(mktemp) remote_sums=$(mktemp)
trap 'rm -f "$list" "$local_sums" "$remote_sums"' EXIT

# Tracked plus untracked-but-not-ignored files, so new work syncs before it is committed.
git ls-files -z --cached --others --exclude-standard | sort -z > "$list"
count=$(tr -cd '\0' < "$list" | wc -c)

if (( copy )); then
  tar --null -T "$list" -cf - | "${SSH[@]}" "mkdir -p '$REMOTE' && tar -xf - -C '$REMOTE'"
  echo "copied $count files to $HOST:$REMOTE"
fi

xargs -0 sha256sum -- < "$list" | sed 's/ \*/  /' > "$local_sums"
"${SSH[@]}" "cd '$REMOTE' && xargs -0 sha256sum --" < "$list" > "$remote_sums" 2>/dev/null || true

if diff -q "$local_sums" "$remote_sums" > /dev/null; then
  echo "hash check OK: $count files identical"
else
  echo "hash check FAILED:" >&2
  diff "$local_sums" "$remote_sums" >&2 || true
  exit 1
fi

# Report (do not delete) server files that git would not track locally.
extra=$("${SSH[@]}" "cd '$REMOTE' && find . -type f \
    -not -path './external/*' -not -path './paper/*' -not -path './.claude/*' \
    -not -path '*/__pycache__/*' -not -path './.pytest_cache/*' | sed 's#^\./##'" \
  | sort | comm -23 - <(tr '\0' '\n' < "$list" | sort))
if [[ -n $extra ]]; then
  echo "server-only files (left in place):"
  echo "$extra" | sed 's/^/  /'
fi

if (( test )); then
  "${SSH[@]}" "cd '$REMOTE' && HF_HUB_OFFLINE=1 USE_TF=0 $PY -m pytest -q 2>&1 | tail -3"
fi
