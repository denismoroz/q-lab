#!/bin/zsh
# The console (web/, port 5890, proxying /api to the API on 8890), kept running by launchd
# (ops/com.qlab.console.plist); see ops/api.sh for why.
#
# It serves the BUILT page (web/dist) with `vite preview`, not the dev server. Until 2026-11-03 it
# was `npm run dev`, which serves the working tree: a half-edited file, or a branch checked out for
# an hour, showed up on the owner's page («похоже сломалась панель»). Now the page changes only when
# ops/console-build.sh builds it from main (docs/CONSOLE.md). For development: `npm run dev` in web/
# on the same port, with this job stopped (docs/CONSOLE.md, «Вернуться к dev-серверу»).
cd "$(dirname "$0")/../web" || exit 1
if [ ! -f dist/index.html ]; then
  # first start on a machine that never built the page: build it from main once, do not loop
  echo "web/dist is missing: building it from main" >&2
  ../ops/console-build.sh --no-restart || exit 1
fi
exec npm run preview
