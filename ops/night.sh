#!/bin/zsh
# The nightly run (docs/NIGHT.md): re-evaluate the watch list on fresh data,
# sweep the graveyard by regime on Sundays, reconcile paper, write the morning
# report. Installed as a launchd agent by ops/com.qlab.night.plist, after the
# recorder (04:10). Resumes a stopped night where it left off.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="data/night/logs/$(date -u +%Y-%m-%d).log"
mkdir -p data/night/logs
{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) start"
  # caffeinate -i: the Mac must not fall back asleep mid-run. On 2026-10-10 it
  # dozed between short wakes and two hours of work took until midday.
  caffeinate -i uv run qlab night run
  rc=$?
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) end (exit $rc)"
} >> "$LOG" 2>&1
