# After-compare outputs

Rerun of `claude-run-1` after the codex-run-1 comparison (aerial `yolo11s-obb.pt`, auto imgsz 1920 portrait / 1280 landscape, traced maps from `config/examples/`).

| photo | total | empty | occupied | occupancy image | auto grid vs traced map |
|---|---:|---:|---:|---|---|
| nigh1 | 191 | 146 | 45 | `nigh1-occupancy.png` | `nigh1-auto-vs-traced.png` |
| nigh2 | 209 | 178 | 31 | `nigh2-occupancy.png` | `nigh2-auto-vs-traced.png` |
| park1 | 54 | 1 | 53 | `park1-occupancy.png` | `park1-auto-vs-traced.png` |
| park2 | 37 | 1 | 36 | `park2-occupancy.png` | `park2-auto-vs-traced.png` |
| sparse | 155 | 149 | 6 | `sparse-occupancy.png` | `sparse-auto-vs-traced.png` |

`*-occupancy.png`: green = empty, red = occupied, index per slot, header with Total | Empty | Occupied and index lists.
`*-auto-vs-traced.png`: blue = hand-traced bays (`config/examples/`), yellow = `detect_slots` auto grid (`output/auto/`).
`*-state.json`: metrics JSON for frame 0.

Reproduce: `python -m parking_vision.main <photo> --slots config/examples/<key>.json --vote 1 --save output/after-compare/<key>-occupancy.png`
