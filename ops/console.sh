#!/bin/zsh
# The console (web/, Vite on port 5890, proxying /api to the API on 8890), kept
# running by launchd (ops/com.qlab.console.plist); see ops/api.sh for why.
# Vite's own file watching keeps the page current with the code.
cd "$(dirname "$0")/../web" || exit 1
exec npm run dev
