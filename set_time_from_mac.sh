#!/bin/bash
# Push this Mac's clock to a Jetson over ssh.
#
# The Jetsons have no RTC battery and no NTP on the robot's network, so they
# drift. Not always to something obviously wrong: snapper was found reading a
# perfectly plausible 2026-07-20 against a real 2026-07-23, sourced from a Go2
# that was behind by the same two and a half days. Session folders are named to
# the second, so once that happens they stop sorting in the order they were
# recorded and `ls -t` points at the wrong one.
#
# chat-manager can pull the time itself when BFF_TIME_HOST is set in the
# Jetson's .env and `set_time.py --serve` is running here. This is the version
# that needs neither: $(date) expands on the Mac before ssh sends the command,
# so the Jetson is handed a timestamp with nothing listening on this end.
#
#   ./set_time_from_mac.sh                 # default host, snapper
#   ./set_time_from_mac.sh helper.local    # somewhere else
#   ./set_time_from_mac.sh snapper.local --check    # report, change nothing
#
# Needs the passwordless-date sudoers drop-in on the target
# (/etc/sudoers.d/99-bff-clock) to set the system clock; without it set_time.py
# reports an offset instead and only chat-manager's own timestamps are fixed.

set -uo pipefail

# A leading -flag means the host was omitted, not that the host is called
# "--check": `./set_time_from_mac.sh --check` should still mean snapper.
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  HOST="$1"
  shift
else
  HOST="cohab@snapper.local"
fi

# Bare hostname is fine - assume the usual account on the dog.
case "$HOST" in
  *@*) ;;
  *)   HOST="cohab@$HOST" ;;
esac

REMOTE_DIR="${BFF_REMOTE_DIR:-~/code/bff-code2}"

echo "Mac time:  $(date)"
echo "Target:    $HOST"

# Expanded here, on the Mac, before ssh ever runs.
ssh "$HOST" "cd $REMOTE_DIR && python3 set_time.py --epoch $(date -u +%s) $*"
RC=$?

if [ $RC -ne 0 ]; then
  echo "set_time.py exited $RC - the clock may not have been set." >&2
  exit $RC
fi

echo "Target now: $(ssh "$HOST" date)"
