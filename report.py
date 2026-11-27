"""Unit 4: render the evaluation into a shadow-mode report.

Run:  .venv/bin/python report.py   ->   reports/shadow_mode_report.md
"""

import json
from pathlib import Path

from PIL import Image

import config

BAR = 0.95                 # the catch-rate bar from the pilot offer
THUMB_DIR = Path("reports/thumbnails")
THUMB_SIZE = (160, 160)


def load(path):
    with open(path) as f:
        return json.load(f)


def defect_type(filename):
    """'broken_small_007.png' -> 'broken_small'."""
    return filename.rsplit("_", 1)[0]


def make_thumbnail(source_path, name):
    """Write a 160px thumbnail into reports/thumbnails/name."""
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    dest = THUMB_DIR / name
    with Image.open(source_path) as image:
        image.convert("RGB").resize(THUMB_SIZE).save(dest)
    return dest


def find_image(filename):
    """Return the full path of an image file by name, searching the eval folders."""
    for folder in [config.TEST_GOOD, config.TEST_DEFECTIVE,
                   config.VALIDATION_GOOD, config.VALIDATION_DEFECTIVE]:
        candidate = Path(folder) / filename
        if candidate.exists():
            return candidate
    return None


def conclusion(catch_rate, false_alarm_rate, bar):
    """The honest verdict, in one sentence."""
    if catch_rate >= bar:
        return (f"Clears the bar: caught {catch_rate:.1%} of defects "
                f"with a {false_alarm_rate:.1%} false-alarm rate.")
    short = (bar - catch_rate) * 100
    return (f"Does NOT clear the bar: caught {catch_rate:.1%} of defects "
            f"({short:.1f} points below the {bar:.0%} bar) "
            f"with a {false_alarm_rate:.1%} false-alarm rate.")


def main():
    evaluation = load("data/evaluation.json")
    threshold = load("data/threshold.json")
    results = evaluation["results"]

    caught = [r for r in results
              if r["actual"] == "defective" and r["predicted"] == "defective"]
    missed = [r for r in results
              if r["actual"] == "defective" and r["predicted"] == "good"]

    for r in missed:                        # the honest part: show the misses
        source = find_image(r["filename"])
        r["thumb"] = make_thumbnail(source, r["filename"]) if source else None

    lines = []
    lines.append("# Shadow-Mode Pilot Report")
    lines.append("")
    lines.append("**Bench demo — MVTec AD \"bottle\" category (benchmark data, not client data).**")
    lines.append("")
    lines.append("## Parts inspected")
    lines.append("")
    total_good = evaluation["passed"] + evaluation["false_alarms"]
    total_defects = evaluation["caught"] + evaluation["missed"]
    lines.append(f"- {total_good + total_defects} parts: "
                 f"{total_good} good, {total_defects} defective")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| | predicted good | predicted defective |")
    lines.append("|---|---|---|")
    lines.append(f"| actually good | {evaluation['passed']} | {evaluation['false_alarms']} |")
    lines.append(f"| actually defective | {evaluation['missed']} | {evaluation['caught']} |")
    lines.append("")
    lines.append(f"- Catch rate: **{evaluation['catch_rate']:.1%}** (bar >= 95%)")
    lines.append(f"- False alarm rate: **{evaluation['false_alarm_rate']:.1%}**")
    lines.append(f"- Threshold: {threshold['threshold']:.3f} "
                 f"(from {threshold['data']}, {threshold['date']})")
    lines.append("")
    lines.append("## Failure cases (defects missed)")
    lines.append("")
    for r in missed:
        lines.append(f"- **{r['filename']}** ({defect_type(r['filename'])}) — distance {r['distance']}")
        if r.get("thumb"):
            lines.append("")
            lines.append(f"![{r['filename']}](thumbnails/{r['filename']})")
            lines.append("")
    lines.append("## Defects flagged")
    lines.append("")
    lines.append(f"{len(caught)} of 63 defects flagged correctly:")
    lines.append("")
    lines.append(", ".join(r["filename"] for r in caught))
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(conclusion(evaluation["catch_rate"],
                            evaluation["false_alarm_rate"], BAR))
    lines.append("")
    lines.append("## Data note")
    lines.append("")
    lines.append("VisA (Visual Anomaly) images — Zou et al., \"SPot-the-Difference\", ECCV 2022. "
                 "Licensed CC BY 4.0 (attribution required): "
                 "github.com/amazon-science/spot-diff. "
                 "Benchmark data, not a client result.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```")
    lines.append(".venv/bin/python evaluate.py")
    lines.append(".venv/bin/python report.py")
    lines.append("```")

    out = Path("reports/shadow_mode_report.md")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}  ({len(missed)} failure thumbnails)")


if __name__ == "__main__":
    main()
