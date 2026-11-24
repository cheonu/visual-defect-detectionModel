# Build plan

One unit at a time. A unit is done when its finish line is met, not when the code exists.

| # | Unit | Concept named | Finish line | Status |
|---|------|---------------|-------------|--------|
| 0a | Bench rig check | linear scan over a pixel stack, `max - min` | 20 photos of one good part agree; verdict `rig is stable` | code scaffolded, `stack_ranges` pending (Che) |
| 0b | Dataset layout | partition; held-out set as the only honest witness | folders exist, counts 209 / 20 / 63, held-out split carved out | folders created, data not downloaded |
| 1 | Loader + baseline features | array/vector, embedding, nearest-neighbour search by linear scan | every training image loads to a vector of the same length, end to end | not started |
| 2 | Detector | threshold = decision boundary on one number | yes/no per test image, distance + threshold + date saved beside each decision | not started |
| 3 | Evaluation harness | confusion matrix 2x2, catch rate and false alarm rate as ratios | one-page output, four counts, two rates, failure images listed, one command to reproduce | not started |
| 4 | Shadow-mode report | — | report a plant manager would accept, same headings as the pilot offer | not started |
| 5 | 60-second video | — | clip under 90s, flag visible, report page at the end | not started |
| 6 | Real cap photos | data swap only, pipeline unchanged | second report on Che's own photos, zero MVTec images | not started |

Order: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6. Unit 4 is the payoff; do not skip it for a fancier model.

## Not building

No line integration, no camera purchase, no production deployment, no multi-defect
model, no real-time streaming. Those belong in a paid pilot, after a plant manager
has seen this demo.
