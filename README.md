# Plant Tray Phenotyping Dashboard

Streamlit Community Cloud app for tray, chamber, and seedling phenotyping.

## Included workflows

- Potato / soybean `2x2` trays
- Lettuce `2x2` trays
- Arabidopsis `4x5` trays
- Growth chamber level scenes with many plants
- Universal custom grids
- Seedlings from flat roots + shoots images
- Linked seedling experiments with:
  - top-view shoots
  - side-view shoot height
  - flat roots + shoots

## Tray batch workflow

Standard tray batches now support:

- green and purple / anthocyanin foliage segmentation on the same tray
- optional `ColorChecker Classic` calibration, either per-image or from a separate reference target image, for more stable canopy and leaf color traits across experiments
- an explicit `Dataset mode` switch between `Independent batch` and `Time series batch`
- a `Batch processing mode` switch, with `Low-memory sequential` recommended for large browser uploads on Streamlit Community Cloud
- per-image frame ordering with parsed labels such as `round32`, `day5`, or `frame12` in time-series mode
- batch leaf tracking across ordered uploads for the same tray sequence in time-series mode
- interactive growth-curve plots for batch, plant, and tracked-leaf traits in time-series mode

Outputs now include:

- `image_summary.csv`
- `plant_summary.csv`
- `leaf_details.csv`
- `leaf_tracks.csv`
- a full results ZIP with originals, overlays, and masks

## Linked seedling workflow

In `Tray layout = Seedlings`, switch `Seedling workflow` to:

- `Linked experiment: top view + side view + flat roots`

Then upload the three modalities separately. The app pairs experiments by upload order and links seedlings left-to-right across the three views.

Outputs include:

- `experiment_summary.csv`
- `seedling_multiview_summary.csv`
- `top_view_leaf_details.csv`
- `flat_view_leaf_details.csv`
- a linked results ZIP with originals, overlays, and masks

## Calibration notes

- Use `Tray long side (cm)` when a tray scale is visible.
- Use `Pixels Per Cm Override` when no tray scale is available.
- Use `Color calibration target = Auto-detect ColorChecker Classic` when each image contains a visible ColorChecker Classic card.
- Use `Color calibration target = Use separate ColorChecker reference image` when you have one checker-only reference shot and want to apply that same calibration to the plant-image batch.
- In linked seedling experiments, top, side, and flat images each have their own manual `pixels / cm` input.
- For flat seedling root images on dark backgrounds, `Full image` is usually the right container mode.

## Local run

```bash
cd "/Users/viniciuslube/Downloads/PlantTrayDashboard-StreamlitCloud-Repo"
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/streamlit run streamlit_app.py
```

## Streamlit Community Cloud

- Repository: `vince-npec/psi`
- Branch: `main`
- Main file: `streamlit_app.py`
- Python: `3.12`

## Notes

- ZIP uploads are supported directly in the browser.
- Invalid files inside ZIP archives are skipped instead of crashing the batch.
- Full results ZIPs are prepared on demand to keep memory use lower on Streamlit Community Cloud.
- Leaf tracking works best when one upload batch represents a single repeated imaging series of the same tray.
