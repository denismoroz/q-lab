"""The anomaly detector and its investigator (docs/ANOMALIES.md).

* `config` -- `night/anomalies.yaml`, every line with its rationale;
* `rerun` -- a stored trial re-run and taken apart by coin and day (read-only);
* `flags` -- the seven flags as pure functions; `scan` -- feeds them from a re-run and the
  registry; `store` -- the `anomaly_scan` and `anomaly_investigation` tables;
* `toolkit` and `cli` -- the investigator's fixed read-only commands (`qlab anomaly ...`);
* `investigator` -- the agent, its checks and the dossier; `stage` -- the night's stage
  `anomalies_stage(day, end, state, ...)`; `report` -- the words for the console and the
  morning report.
"""
