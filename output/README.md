# Corrected parking-photo evaluation

Open **index.html** for the current gallery, or **report-screenshot.png** for its screenshot.
Current annotated PNGs are `images/nigh1-output.png`, `nigh2-output.png`, `park1-output.png`, `park2-output.png`, and `sparse-output.png`.

## Root causes and fixes

1. **Wrong model domain:** COCO `yolo11s.pt` returned no vehicles in the overhead screenshots. The default is now the pretrained aerial `yolo11s-obb.pt`; no fine-tuning or additional dependency was introduced.
2. **Different class IDs and output type:** DOTA uses vehicle classes 9/10 in these weights, not COCO's 2/3/5/7. The detector resolves vehicle names automatically and reads both `result.obb` and ordinary `result.boxes`. The bounding-box center rule is unchanged.
3. **Inference scaling:** `--width 1280` alone did not prevent YOLO from internally reducing the input to 640. Auto inference now uses 1280 for landscape frames and 1920 for portrait frames. `--imgsz` overrides this. Raising the size universally degraded some landscape results, so it is not used universally.
4. **Incomplete slot coverage:** the original default contained two example slots and the initial image evaluation used only selected bays. The CLI now requires `--slots`. Each current photo has a manually traced visible-bay map in `config/examples/`; source hashes guard against stale annotations. Model predictions never generate the map or consume its evaluation labels.
5. **Usability and evaluation:** the CLI saves PNG/JPG overlays directly, checks polygon bounds, centers compact labels, and abbreviates the header's long empty list while preserving complete JSON. Evaluation keys are stable filenames, not list positions.

## Actual default-CLI results

| Image | Mapped visible bays | Predicted empty | Predicted occupied | Known false empty | Known false occupied |
|---|---:|---:|---:|---:|---:|
| nigh1 | 191 | 146 | 45 | 1 | 0 |
| nigh2 | 209 | 178 | 31 | 0 | 0 |
| park1 | 54 | 1 | 53 | 0 | 1 |
| park2 | 37 | 1 | 36 | 0 | 0 |
| sparse | 155 | 149 | 6 | 0 | 0 |

The maps contain **646 visible bays**. Independent visual labels cover **620 bays** (167 occupied, 453 empty); **26 ambiguous labels** are excluded from error measurement. The original model missed all 167 labeled occupied bays. The corrected model has **one false empty and one false occupied** on those 620 labels. These five development images are a diagnostic set, not a held-out accuracy benchmark.

Remaining errors are explicit in `fix-summary.json`: nigh1 slot 170 is an obscured vehicle; park1 slot 4 is an empty accessible bay whose markings look like a vehicle to the model. Do not treat the model's predictions as perfect availability ground truth. Trees, cropped/hidden spaces and ambiguous paint boundaries limit what these screenshots establish about physical capacity. Totals apply to the mapped visible footprint, not unseen portions of a lot.

## Verification

- `pytest tests/`: **46 passing offline tests**. Includes class mapping, oriented-box extraction, automatic inference size, image output, coordinate checks and source/map consistency.
- All five photos were also run through `python -m parking_vision.main`; every JSON result and every saved PNG matched the evaluator exactly.
- Source image hashes and model settings are recorded with the maps and evaluation results.

## Reproduce

From the repository root:

```bash
source .venv/bin/activate
python output/run_detection.py
python -m parking_vision.main "Example photos/example photos/park2.png" \
  --slots config/examples/park2.json --vote 1 --device cpu \
  --save output/images/park2-output.png
```

The annotation source is `output/annotate_samples.py`. Run it only after reviewing or editing the manually traced rows; it exports the polygons and evaluation labels, and never performs automatic slot discovery. `config/slots.json` remains an explicitly selected example only.

`fix-summary.json` contains current before/after detections and known errors; `<name>-state.json` contains the unchanged metrics schema. The numbered PNGs/JSON files and `summary.json` preserve the initial failed-run evidence. Some original photo names were subsequently changed or removed; that old snapshot is historical and is superseded by the current manifest and gallery.
