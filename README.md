# Plant Tray Phenotyping Dashboard

Streamlit Community Cloud-ready app for potato, soybean, and Arabidopsis tray phenotyping.

## Included

- `streamlit_app.py`: Streamlit entrypoint for Community Cloud
- `tray_analyzer.py`: tray segmentation, ownership, and trait extraction
- `requirements.txt`: Python dependencies for deployment
- `batch_input/` and `batch_output/`: optional leftover local batch folders

## GitHub Push

This folder is already initialized as a local Git repository.

If you want to push it with GitHub CLI:

```bash
cd /Users/viniciuslube/Downloads/PlantTrayDashboard-StreamlitCloud-Repo
git add .
git commit -m "Initial Streamlit Community Cloud app"
gh repo create plant-tray-phenotyping-dashboard --private --source=. --remote=origin --push
```

If you prefer GitHub Desktop:

1. Add this folder as an existing local repository.
2. Publish it to GitHub.
3. Keep `main` as the default branch.

## Deploy On Streamlit Community Cloud

Official deployment docs:

- [Deploy your app on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

Recommended settings:

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
2. Click `Create app`.
3. Select your new GitHub repository.
4. Branch: `main`
5. File path: `streamlit_app.py`
6. In `Advanced settings`, use Python `3.12` unless you need another supported version.
7. Deploy.

## Notes

- The app supports `Auto`, `Potato / Soybean (2x2)`, and `Arabidopsis (4x5)` tray layouts.
- The deployed Community Cloud app is browser-upload only. Users can upload one image, a batch of images, or one or more `.zip` archives directly in the web UI.
- Each uploaded image can keep the default tray long side or use its own per-image scale override before export.
- The app can return a single downloadable results zip containing combined CSVs, per-image CSVs, original images, overlays, and mask outputs.
- Invalid or non-image items inside uploads are skipped automatically so one bad file does not stop a batch run.
- Physical traits are normalized from the tray long side, defaulting to `33 cm`.
