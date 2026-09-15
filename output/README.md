# Example photo test outputs

Open `index.html` for the image gallery. `report-screenshot.png` is a browser screenshot of the gallery. All seven rendered output PNGs are in `images/`.

## Result

The default `yolo11s.pt` detector returned zero vehicles in all six parking photos. It also correctly returned zero vehicles in the calendar screenshot. Lowering confidence from 0.25 to 0.10 still returned zero vehicles in every image.

The per-slot check used 56 manually selected visible bays across the six parking photos. Of these, 35 were visually labeled occupied and 21 empty. The program reported all 56 empty: 35 false-empty results. This is a small manually labeled diagnostic sample, not a full-lot accuracy benchmark. Green polygons show the actual model-driven output; they do not establish that those spaces are empty.

Settings: width 1280, aspect ratio preserved, vehicle classes 2/3/5/7, confidence 0.25, CPU, vote 1 for independent still images. Slot coordinates are saved at the inference resolution. No model or application changes were made for this evaluation.

## Files

- `images/01-output.png`: calendar negative control.
- `images/02-output.png` through `07-output.png`: parking photo occupancy overlays.
- `summary.json`: source filenames, sampled counts, errors, and image paths.
- `detections.json`: raw boxes and inference settings.
- `manual-labels.json`, `*-slots.json`: manual visual labels and polygons for the sampled bays only.
- `*-state.json`, `states.jsonl`: occupancy metrics in the application's schema; frame 0 for each independent image, ordered as images 2 through 7.
- `lower-confidence-diagnostic.json`: results at confidence 0.10.

## Reproduce from the repository root

```bash
.venv/bin/python output/run_detection.py
.venv/bin/python output/annotate_samples.py
.venv/bin/python output/render_results.py
```

These scripts are evaluation artifacts, separate from the application package. `config/slots.json` is unchanged.
