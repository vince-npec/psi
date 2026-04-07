# Plant Tray Phenotyping Dashboard

Streamlit Community Cloud app for tray, chamber, and seedling phenotyping.

## Included workflows

- Potato / soybean `2x2` trays
- Arabidopsis `4x5` trays
- Universal custom grids
- Seedlings from flat roots + shoots images
- Linked seedling experiments with:
  - top-view shoots
  - side-view shoot height
  - flat roots + shoots

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
