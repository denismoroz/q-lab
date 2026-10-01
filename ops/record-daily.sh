#!/bin/zsh
# Daily run of q-lab's own market-data recorder (docs/TASKS.md T33).
# Installed as a launchd agent by ops/com.qlab.recorder.plist. Each source is
# its own call: intervals apply to the source they are written next to.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="data/recorded/logs/$(date -u +%Y-%m-%d).log"
mkdir -p data/recorded/logs
{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) start"
  # xyz HIP-3: the venue erases delisted instruments' whole history.
  uv run qlab data record --source hyperliquid-xyz --interval 1d --interval 1h
  # Main market: daily history is fully served; hourly only ~5000 bars back.
  uv run qlab data record --source hyperliquid --interval 1h
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) end"
} >> "$LOG" 2>&1
