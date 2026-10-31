#!/bin/zsh
# The nightly run (docs/NIGHT.md): re-evaluate the watch list on fresh data,
# sweep the graveyard by regime on Sundays, reconcile paper, write the morning
# report. Installed as a launchd agent by ops/com.qlab.night.plist at 03:20,
# after the recorder (03:05). Resumes a stopped night where it left off.
#
# Owner, 2026-10-26: the night must not run into the working day and must be over
# before 08:00 Minsk (stop time 07:10). It cannot start earlier: the night reads the last closed
# daily candle, and that closes at 00:00 UTC (03:00 Minsk).
# Owner, 2026-10-31: «отключаем 7.10 пока. я хочу понять и видеть что так долго отрабатывает».
# So there is NO stop time now: the night runs to the end and the report shows how long each stage
# took. To bring the stop back, set it: QLAB_STOP_AT=07:10 ops/night.sh (nothing new starts after
# it; a started piece of work finishes, the rest goes to the next night). A night run by hand names
# its own stop the same way: QLAB_STOP_AT=23:00 ops/night.sh.
# Owner, 2026-10-31: the night is a table of steps (night/pipeline.yaml, docs/PIPELINE.md). The console's
# "Пайплайн" page starts this same script on a process of its own (QLAB_TRIGGER=button).
STOP_AT="${QLAB_STOP_AT:-}"
# Binance instruments fetched two at a time (docs/NIGHT_SPEED.md; owner 2026-10-29: «да, включай 2
# потока»): its funding limit is 500 requests per 5 min and one thread makes about 30 a minute.
# Hyperliquid stays one at a time whatever this says. Back to 1 if the log shows 429 refusals.
export QLAB_FETCH_WORKERS=2
set -u
# The whole script stays awake, the wait for the recorder too: on 2026-10-30 the Mac was asleep at
# 03:05, the recorder crept through 45-second dark wakes until 08:13 while this script waited for
# it, and the night started after its stop time. -i holds on battery, -s on power.
if [[ -z "${QLAB_AWAKE:-}" ]]; then
  QLAB_AWAKE=1 exec caffeinate -i -s "$0" "$@"
fi
cd "$(dirname "$0")/.." || exit 1
LOG="data/night/logs/$(date -u +%Y-%m-%d).log"
mkdir -p data/night/logs
{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) start"
  # The night reads what the recorder writes: wait for it if it is still running.
  while pgrep -f ops/record-daily.sh > /dev/null; do sleep 30; done
  # Arguments go through to `qlab night run` (the console's button passes --off / --restart; one run
  # at a time: the run takes a lock, and a second start ends at once with exit 75).
  if [[ -n "$STOP_AT" ]]; then
    uv run qlab night run --stop-at "$STOP_AT" "$@"
  else
    uv run qlab night run "$@"
  fi
  rc=$?
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) end (exit $rc)"
} >> "$LOG" 2>&1
