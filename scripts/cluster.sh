#!/bin/bash
# Every cluster command from this Mac goes through here: one shared SSH login (ControlMaster, set up in ~/.ssh/config),
# at most MAX_PER_HOUR new logins an hour, and no retries after a refusal: frequent fresh logins and connections
# left half-open (a VPN drop) can make the login node refuse an address for a while ("Not allowed at this time").
#
#   scripts/cluster.sh 'squeue -u $USER'                 run a command on the cluster
#   scripts/cluster.sh --put FILE... REMOTE_DIR          copy files up (scp through the shared login)
#   scripts/cluster.sh --get REMOTE_FILE LOCAL_PATH      copy a file down
#   scripts/cluster.sh --status                          shared login open? new logins in the last hour?
set -uo pipefail
HOST=${CLUSTER_HOST:-jgwong.2023@10.193.104.102}
LOG=${CLUSTER_LOGIN_LOG:-$HOME/.ssh/cluster-logins.log}
MAX_PER_HOUR=${CLUSTER_MAX_LOGINS_PER_HOUR:-3}
WAIT_AFTER_REFUSAL=1800  # seconds; one attempt after that, never a loop
touch "$LOG"
now=$(date +%s)

recent() { awk -v since=$((now - $1)) -v kind="$2" '$1 >= since && $3 == kind' "$LOG"; }

shared_open() { ssh -O check "$HOST" >/dev/null 2>&1; }

status() {
    if shared_open; then echo "shared login: open"; else echo "shared login: closed"; fi
    echo "new logins in the last hour: $(recent 3600 new | wc -l | tr -d ' ') of at most $MAX_PER_HOUR"
    last=$(recent 86400 refused | tail -1)
    [ -n "$last" ] && echo "last refusal: $(echo "$last" | cut -d' ' -f2-)"
    return 0
}

ensure_shared() {
    shared_open && return 0
    refused=$(recent $WAIT_AFTER_REFUSAL refused | tail -1)
    if [ -n "$refused" ]; then
        echo "STOP: the cluster refused a login at $(echo "$refused" | cut -d' ' -f2) (less than 30 min ago);" \
             "no new attempt until $(date -r $(( $(echo "$refused" | cut -d' ' -f1) + WAIT_AFTER_REFUSAL )) +%H:%M). Tell the user." >&2
        exit 3
    fi
    n=$(recent 3600 new | wc -l | tr -d ' ')
    if [ "$n" -ge "$MAX_PER_HOUR" ]; then
        echo "STOP: $n new logins in the last hour already (limit $MAX_PER_HOUR); the shared login keeps dying" \
             "(VPN? Mac asleep?). Tell the user instead of logging in again." >&2
        exit 3
    fi
    echo "$now $(date +%H:%M) new" >> "$LOG"
    if ! err=$(ssh -M -N -f -o BatchMode=yes -o ConnectTimeout=20 "$HOST" 2>&1); then
        echo "$now $(date +%H:%M) refused ${err//$'\n'/ }" >> "$LOG"
        echo "STOP: login refused: $err" >&2
        echo "No retry: wait 30+ minutes or reconnect the VPN, then try once. Tell the user." >&2
        exit 2
    fi
}

case "${1:-}" in
    --status) status ;;
    --put)
        shift; [ $# -ge 2 ] || { echo "usage: $0 --put FILE... REMOTE_DIR" >&2; exit 64; }
        ensure_shared; dest=${!#}; set -- "${@:1:$#-1}"
        scp -q -o ControlMaster=no -o BatchMode=yes "$@" "$HOST:$dest" ;;
    --get)
        shift; [ $# -eq 2 ] || { echo "usage: $0 --get REMOTE_FILE LOCAL_PATH" >&2; exit 64; }
        ensure_shared; scp -q -o ControlMaster=no -o BatchMode=yes "$HOST:$1" "$2" ;;
    ''|-h|--help) sed -n '2,10p' "$0"; exit 64 ;;
    *)
        ensure_shared
        ssh -o ControlMaster=no -o BatchMode=yes "$HOST" "$*" 2>&1 | grep -v '^tput: ' ;;
esac
