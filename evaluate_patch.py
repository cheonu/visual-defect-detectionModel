"""Evaluate the patch-level detector on validation or test.

Run:  .venv/bin/python evaluate_patch.py validation
      .venv/bin/python evaluate_patch.py test
"""

import sys
import json
import numpy as np

from embedder import Embedder
import config
import log_experiment
from patch_scoring import image_score


def main():
    set_name = sys.argv[1] if len(sys.argv) > 1 else "test"
    if set_name == "validation":
        good_folders = [config.VALIDATION_GOOD]
        defective_folders = [config.VALIDATION_DEFECTIVE]
    else:
        good_folders = [config.TEST_GOOD]
        defective_folders = [config.TEST_DEFECTIVE]

    bank = np.load("data/patch_bank.npy")
    threshold = json.load(open("data/patch_threshold.json"))["threshold"]
    embedder = Embedder()

    results = []
    for folders, actual in [(good_folders, "good"),
                            (defective_folders, "defective")]:
        for folder in folders:
            for path in config.list_images(folder):
                score = image_score(embedder.embed_patches(path), bank)
                results.append({
                    "filename": path.name,
                    "score": round(score, 3),
                    "predicted": "defective" if score > threshold else "good",
                    "actual": actual,
                })

    good = [r for r in results if r["actual"] == "good"]
    bad = [r for r in results if r["actual"] == "defective"]
    caught = sum(1 for r in bad if r["predicted"] == "defective")
    missed = sum(1 for r in bad if r["predicted"] == "good")
    false_alarms = sum(1 for r in good if r["predicted"] == "defective")
    passed = sum(1 for r in good if r["predicted"] == "good")

    total_defects = caught + missed
    total_good = false_alarms + passed
    catch_rate = caught / total_defects if total_defects else 0.0
    false_alarm_rate = false_alarms / total_good if total_good else 0.0

    print(f"set: {set_name} (patch-level)")
    print("confusion matrix")
    print(f"  defects:  caught {caught}   missed {missed}   (of {total_defects})")
    print(f"  good:     passed {passed}   flagged {false_alarms}   (of {total_good})")
    print(f"  catch rate       {catch_rate:.1%}")
    print(f"  false alarm rate {false_alarm_rate:.1%}")

    failures = ([r for r in bad if r["predicted"] == "good"] +
                [r for r in good if r["predicted"] == "defective"])
    print(f"\nfailure cases ({len(failures)})")
    for r in failures:
        print(f"  {r['actual']:>9}  {r['filename']:>24}  score {r['score']:.3f}")

    report = {
        "set": set_name,
        "method": "patch",
        "threshold": threshold,
        "caught": caught,
        "missed": missed,
        "false_alarms": false_alarms,
        "passed": passed,
        "catch_rate": catch_rate,
        "false_alarm_rate": false_alarm_rate,
        "results": results,
    }
    with open("data/evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved data/evaluation.json ({len(results)} results)")
    log_experiment.log_run()


if __name__ == "__main__":
    main()
