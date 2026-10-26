#!/bin/zsh
# The registry's API for the console (qlab.api), kept running by launchd
# (ops/com.qlab.api.plist). Started from the app's browser pane it died whenever
# the app closed the pane (2026-10-25, 2026-10-26); as an agent it runs whenever
# the Mac is on. No --reload: a service, not a dev loop -- `launchctl kickstart -k
# gui/$(id -u)/com.qlab.api` picks up new code.
cd "$(dirname "$0")/.." || exit 1
exec uv run uvicorn qlab.api:app --host 127.0.0.1 --port 8890
