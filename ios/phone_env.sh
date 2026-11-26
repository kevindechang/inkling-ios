#!/bin/bash
# phone_env.sh <log> '<json env>' <app args...>: launch InklingBench with environment variables.
# Required env: IOS_UDID, IOS_BUNDLE_ID
set -euo pipefail
: "${IOS_UDID:?set IOS_UDID}"
: "${IOS_BUNDLE_ID:?set IOS_BUNDLE_ID}"
log=$1; env_json=$2; shift 2
limit=${RUN_S:-300}; [ "$limit" -gt 300 ] && limit=300
perl -e "alarm $limit; exec @ARGV" xcrun devicectl device process launch --device "$IOS_UDID" --console \
  --terminate-existing -e "$env_json" "$IOS_BUNDLE_ID" -- "$@" > "$log" 2>&1 || echo "launch exit $?"
