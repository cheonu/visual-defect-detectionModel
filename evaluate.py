"""Unit 3: the evaluation harness.

Runs the detector over the full test set and counts the confusion matrix:
defects caught, defects missed, good parts flagged, good parts passed.

Run:  .venv/bin/python evaluate.py
"""

import json
from pathlib import Path

import nearest
from embedder import Embedder
import config


GOOD_FOLDERS = [config.TEST_GOOD, config.HELDOUT_GOOD]
DEFECTIVE_FOLDERS = [config.TEST_DEFECTIVE, config.HELDOUT_DEFECTIVE]


def classify(distance, threshold):
    """One distance + the threshold -> one verdict."""
    return "defective" if distance > threshold else "good"


def evaluate(embedder, vectors, threshold, folders, actual):
    """Embed and score every image in the folders; return a list of result dicts."""
    results = []
    for folder in folders:
        for path in sorted(Path(folder).glob("*.png")):
            distance = nearest.nearest_distance(embedder.embed(path), vectors)
            results.append({
                "filename": path.name,
                "distance": round(float(distance), 3),
                "predicted": classify(distance, threshold),
                "actual": actual,
            })
    return results


def confusion_counts(good_results, bad_results):
    """Return (caught, missed, false_alarms, passed)."""
    caught = sum(1 for r in bad_results if r["predicted"] == "defective")
    missed = sum(1 for r in bad_results if r["predicted"] == "good")
    false_alarms = sum(1 for r in good_results if r["predicted"] == "defective")
    passed = sum(1 for r in good_results if r["predicted"] == "good")
    return caught, missed, false_alarms, passed


def main():
    embedder = Embedder()
    names, vectors = nearest.load_store()
    threshold = json.load(open("data/threshold.json"))["threshold"]

    good = evaluate(embedder, vectors, threshold, GOOD_FOLDERS, "good")
    bad = evaluate(embedder, vectors, threshold, DEFECTIVE_FOLDERS, "defective")

    caught, missed, false_alarms, passed = confusion_counts(good, bad)

    total_defects = caught + missed
    total_good = false_alarms + passed
    catch_rate = caught / total_defects if total_defects else 0.0
    false_alarm_rate = false_alarms / total_good if total_good else 0.0

    print("confusion matrix")
    print(f"  defects:  caught {caught}   missed {missed}   (of {total_defects})")
    print(f"  good:     passed {passed}   flagged {false_alarms}   (of {total_good})")
    print(f"  catch rate       {catch_rate:.1%}")
    print(f"  false alarm rate {false_alarm_rate:.1%}")

    failures = ([r for r in bad if r["predicted"] == "good"] +
                [r for r in good if r["predicted"] == "defective"])
    print(f"\nfailure cases ({len(failures)})")
    for r in failures:
        print(f"  {r['actual']:>9}  {r['filename']:>24}  dist {r['distance']:.3f}")

    report = {
        "threshold": threshold,
        "caught": caught,
        "missed": missed,
        "false_alarms": false_alarms,
        "passed": passed,
        "catch_rate": catch_rate,
        "false_alarm_rate": false_alarm_rate,
        "results": good + bad,
    }
    with open("data/evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved data/evaluation.json ({len(good) + len(bad)} results)")


if __name__ == "__main__":
    main()
