#!/bin/zsh
# Daily off-machine copy of q-lab's state to dis@10.8.0.5:qlab-backup/ (docs/BACKUP.md).
# Owner, 2026-10-28: «я думаю уже можно настраивать бекап на 10.8.0.5»; «по qlab - держать
# последние 14 дней в бекапе»; «удаляй старые локальные копии после бекапа».
# Installed as a launchd agent by ops/com.qlab.backup.plist, after the night has ended.
# The recorder's own data (data/recorded/) is copied by ops/record-daily.sh.
set -u
cd "$(dirname "$0")/.." || exit 1
HOST="dis@10.8.0.5"
DEST="qlab-backup"
KEEP_DAILY=14
DAY="$(date -u +%Y-%m-%d)"
WORK="data/backup"
LOG="$WORK/logs/$DAY.log"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
mkdir -p "$WORK/logs" "$WORK/tmp"
FAILED=0
{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

  # 1. The registry. A plain file copy of a live SQLite database can be torn; SQLite's own
  # online backup gives a consistent copy while the night or the console write to it. The
  # copy is checked before it leaves the machine, then compressed under its day's name.
  COPY="$WORK/tmp/qlab.db"
  GZ="$WORK/tmp/qlab-$DAY.db.gz"
  rm -f "$COPY" "$GZ"
  if sqlite3 data/qlab.db ".backup '$COPY'" && [[ "$(sqlite3 "$COPY" 'PRAGMA quick_check;')" == "ok" ]]; then
    gzip -c "$COPY" > "$GZ"
    echo "registry copy ok: $(du -h "$COPY" | cut -f1) -> $(du -h "$GZ" | cut -f1) compressed"
    if rsync -a --partial -e "${SSH[*]}" "$GZ" "$HOST:$DEST/registry/"; then
      echo "registry sent"
      # Keep the last KEEP_DAILY days' copies on the host (names sort by date).
      "${SSH[@]}" "$HOST" "cd $DEST/registry && ls -1r qlab-*.db.gz | tail -n +$((KEEP_DAILY + 1)) \
        | while read f; do rm -f \"\$f\"; done; ls -1 qlab-*.db.gz | wc -l" \
        | sed 's/^ */copies on the host: /' || echo "HOST PRUNE FAILED (copies kept)"
    else
      echo "REGISTRY SEND FAILED"; FAILED=1
    fi
  else
    echo "REGISTRY COPY FAILED (backup or integrity check)"; FAILED=1
  fi
  rm -f "$COPY" "$GZ"

  # 2. Everything else the lab keeps outside git: snapshots the trials point to, paper books,
  # ledgers, queues, forward series. No --delete: a file gone here stays there. Left out:
  # the registry and its local backups (above), the recorder's data (its own job), the
  # programmer's worktrees (branches live in git) and this job's own folder.
  STATS="$(rsync -a --partial --stats -e "${SSH[*]}" \
    --exclude 'qlab.db' --exclude 'qlab.db-*' --exclude 'qlab.db.bak-*' \
    --exclude 'recorded/' --exclude 'programmer/worktrees/' --exclude 'backup/' \
    data/ "$HOST:$DEST/data/" 2>&1)"
  RC=$?
  print -r -- "$STATS" | grep -E "Number of files transferred|Total transferred file size|rsync"
  if [[ $RC -ne 0 ]]; then echo "DATA SEND FAILED (rsync exit $RC)"; FAILED=1; else echo "data sent"; fi

  # 3. After a good backup, the local registry copies older than a day go: the off-machine copy
  # replaces them. A copy younger than a day is kept -- it may be the restore point of an
  # operation still running (agents back the registry up before a risky change).
  if [[ $FAILED -eq 0 ]]; then
    OLD=(data/qlab.db.bak-*(N.mh+24))
    if (( ${#OLD} )); then
      echo "local copies removed: ${#OLD} ($(du -ch "${OLD[@]}" | tail -1 | cut -f1))"
      rm -f -- "${OLD[@]}"
    fi
  fi

  if [[ $FAILED -ne 0 ]]; then echo "BACKUP FAILED"; else echo "backup ok"; fi
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) end"
} >> "$LOG" 2>&1
exit $FAILED
