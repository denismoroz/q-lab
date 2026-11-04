#!/bin/zsh
# Build the console page (web/dist) from the committed `main`, then restart the console.
#
# Why from main and not from the working tree: the working tree may hold a half-edited file or a
# branch (docs/CONSOLE.md); the page the owner opens must be a committed one. The web/ directory of
# the branch is exported with `git archive`, built in a temporary directory, and only then swapped
# in for web/dist, so a failed build leaves the page that is running untouched.
#
# Run it after every merge into main that touches web/ (the night does not need it):
#     ops/console-build.sh
# BRANCH=<ref> builds another committed ref; --no-restart only builds (no launchd call).
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BRANCH="${BRANCH:-main}"
RESTART=1
[ "${1:-}" = "--no-restart" ] && RESTART=0

WORK="$(mktemp -d "${TMPDIR:-/tmp}/qlab-console-build.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

git -C "$REPO" archive "$BRANCH" web | tar -x -C "$WORK"
SHA="$(git -C "$REPO" rev-parse --short "$BRANCH")"
(
  cd "$WORK/web"
  npm ci --no-audit --no-fund --silent
  npx vite build --outDir "$WORK/dist" --emptyOutDir
)

# swap: the old page goes only after the new one is complete
rm -rf "$REPO/web/dist.old"
[ -d "$REPO/web/dist" ] && mv "$REPO/web/dist" "$REPO/web/dist.old"
mv "$WORK/dist" "$REPO/web/dist"
rm -rf "$REPO/web/dist.old"
echo "console page built from $BRANCH ($SHA) into web/dist"

if [ "$RESTART" = 1 ]; then
  if launchctl kickstart -k "gui/$(id -u)/com.qlab.console" 2>/dev/null; then
    echo "console restarted: http://localhost:5890"
  else
    echo "launchd job com.qlab.console is not loaded: nothing restarted (docs/CONSOLE.md)" >&2
  fi
  # The page reads the API: a merge that changed both must not leave the old API serving the new
  # page (2026-11-04: the whole-period chart was built and merged, the API still said «рисуется
  # только для прошедших»).
  if launchctl kickstart -k "gui/$(id -u)/com.qlab.api" 2>/dev/null; then
    echo "api restarted"
  else
    echo "launchd job com.qlab.api is not loaded: the API was not restarted" >&2
  fi
fi
