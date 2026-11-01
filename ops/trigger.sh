# Who started this run, for the pipeline journal's `trigger` (sourced by ops/night.sh; docs/PIPELINE.md).
#
#   schedule  the night launchd starts at 03:20 (ops/com.qlab.night.plist)
#   button    the console's "Запустить" (it sets QLAB_TRIGGER itself)
#   hand      a person's command
#
# 2026-11-01: run 31 was started by launchd and the journal said `hand`: the label was read from
# XPC_SERVICE_NAME in the Python process, which launchd's job did not hand down. So the script
# decides, before it replaces itself with caffeinate, and exports the answer.
#
# qlab_trigger EXISTING XPC_SERVICE_NAME PPID STDIN_IS_A_TTY(0|1) prints the label:
#   * a label already given (the button's, or a person's own QLAB_TRIGGER) is kept;
#   * launchd's own service name says `schedule`;
#   * a process whose parent is launchd (pid 1) and that has no terminal is launchd's job as well;
#   * anything else is a hand.
qlab_trigger() {
  local existing="$1" xpc="$2" ppid="$3" tty="$4"
  if [[ -n "$existing" ]]; then
    print -r -- "$existing"
  elif [[ "$xpc" == "com.qlab.night" || ( "$ppid" == "1" && "$tty" != "1" ) ]]; then
    print -r -- "schedule"
  else
    print -r -- "hand"
  fi
}
