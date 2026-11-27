"""Experiment log: record each run so results never evaporate.

Appends one line per run to reports/experiments.csv.

Run standalone:  .venv/bin/python log_experiment.py
Also called automatically at the end of evaluate.py.
"""

import csv
import datetime
import json
from pathlib import Path

import config

LOG = Path("reports/experiments.csv")
FIELDS = ["model", "dataset", "set", "catch_rate", "false_alarm_rate",
          "threshold", "date"]


def log_run():
    evaluation = json.load(open("data/evaluation.json"))
    threshold = json.load(open("data/threshold.json"))
    row = {
        "model": config.MODEL_NAME,
        "dataset": config.DATASET,
        "set": evaluation.get("set", "test"),
        "catch_rate": round(evaluation["catch_rate"], 4),
        "false_alarm_rate": round(evaluation["false_alarm_rate"], 4),
        "threshold": round(threshold["threshold"], 4),
        "date": datetime.date.today().isoformat(),
    }
    if not LOG.exists():
        with open(LOG, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    with open(LOG, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    return row


if __name__ == "__main__":
    print("logged:", log_run())
