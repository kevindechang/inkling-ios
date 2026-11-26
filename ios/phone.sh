#!/bin/bash
# Device helper for InklingBench on iPhone.
# Required env: IOS_UDID, IOS_BUNDLE_ID
# Usage:
#   phone.sh push <file>              copy file to Documents/models/<basename>
#   phone.sh ls                       list Documents/models/
#   phone.sh run <log> <args...>      launch app with args, stream console to <log>
#   phone.sh thermal                  print current thermal state
set -euo pipefail
: "${IOS_UDID:?set IOS_UDID}"
: "${IOS_BUNDLE_ID:?set IOS_BUNDLE_ID}"
DOMAIN=(--device "$IOS_UDID" --domain-type appDataContainer --domain-identifier "$IOS_BUNDLE_ID")
MAX_RUN_S=300

case ${1:-} in
  push)
    xcrun devicectl device copy to "${DOMAIN[@]}" --source "$2" --destination "Documents/models/$(basename "$2")"
    xcrun devicectl device info files "${DOMAIN[@]}" --subdirectory Documents/models | tail -10 ;;
  ls)
    xcrun devicectl device info files "${DOMAIN[@]}" --subdirectory Documents/models ;;
  run)
    log=$2; shift 2
    limit=${RUN_S:-$MAX_RUN_S}; [ "$limit" -gt "$MAX_RUN_S" ] && limit=$MAX_RUN_S
    perl -e "alarm $limit; exec @ARGV" xcrun devicectl device process launch --device "$IOS_UDID" --console \
      --terminate-existing "$IOS_BUNDLE_ID" -- "$@" > "$log" 2>&1 || echo "launch exit $?" ;;
  thermal)
    t=$(mktemp); "$0" run "$t" info; grep -o 'system.thermal: [a-z]*' "$t" | awk '{print $2}'; rm -f "$t" ;;
  *)
    echo "usage: $0 {push|ls|run|thermal} ..." >&2; exit 2 ;;
esac
