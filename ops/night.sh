#!/bin/zsh
# The nightly run (docs/NIGHT.md): re-evaluate the watch list on fresh data,
# sweep the graveyard by regime on Sundays, reconcile paper, write the morning
# report. Installed as a launchd agent by ops/com.qlab.night.plist at 03:20,
# after the recorder (03:05). Resumes a stopped night where it left off.
#
# Owner, 2026-10-26: the night must not run into the working day and must be over
# before 08:00 Minsk. It cannot start earlier: the night reads the last closed
# daily candle, and that closes at 00:00 UTC (03:00 Minsk). Nothing new starts
# after STOP_AT; a started piece of work finishes (the longest seen, a bv2 run on
# Binance hourly data, ~46 min), the rest goes to the next night.
STOP_AT="07:10"
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="data/night/logs/$(date -u +%Y-%m-%d).log"
mkdir -p data/night/logs
{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) start"
  # caffeinate -i: the Mac must not fall back asleep mid-run. On 2026-10-10 it
  # dozed between short wakes and two hours of work took until midday.
  # The night reads what the recorder writes: wait for it if it is still running.
  while pgrep -f ops/record-daily.sh > /dev/null; do sleep 30; done
  caffeinate -i uv run qlab night run --stop-at "$STOP_AT"
  rc=$?
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) end (exit $rc)"
} >> "$LOG" 2>&1
