from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import potato_leaf_dashboard.tray_analyzer as tray_analyzer
except ImportError:  # pragma: no cover - Streamlit script execution fallback
    import tray_analyzer

tray_analyzer.MAX_PLANT_ANALYSIS_DIM = max(int(getattr(tray_analyzer, "MAX_PLANT_ANALYSIS_DIM", 0) or 0), 4000)
analyze_tray_image = tray_analyzer.analyze_tray_image
TRAY_PROFILE_OPTIONS = tray_analyzer.get_tray_profile_options()
TRAY_PROFILE_LABELS = {key: label for key, label in TRAY_PROFILE_OPTIONS}
DEFAULT_TRAY_PROFILE_KEY = getattr(tray_analyzer, "AUTO_TRAY_PROFILE_KEY", TRAY_PROFILE_OPTIONS[0][0])
DEFAULT_TRAY_LONG_SIDE_CM = float(getattr(tray_analyzer, "TRAY_LONG_SIDE_CM", 33.0))


st.set_page_config(
    page_title="Plant Tray Phenotyping Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MAX_PREVIEW_EDGE = 1400
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
APP_ROOT = Path(__file__).resolve().parent
LOGO_PATH = APP_ROOT / "assets" / "NPEC-logo-black.png"
FOOTER_TEXT = "© 2026 NPEC Innovation - Visualization Dashboard by Dr. Vinicius Lube | Phenomics Engineer Innovation Lead"


@st.cache_data(show_spinner=False, max_entries=32)
def _analyze_upload(image_bytes: bytes, tray_profile_key: str, tray_long_side_cm: float):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return analyze_tray_image(
        np.array(image),
        tray_profile_key=tray_profile_key,
        tray_long_side_cm=float(tray_long_side_cm),
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _preview_from_bytes(image_bytes: bytes, max_edge: int) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _resize_for_preview(np.array(image), max_edge=max_edge)


def _resize_for_preview(image_rgb: np.ndarray, max_edge: int = MAX_PREVIEW_EDGE) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= max_edge:
        return image_rgb

    scale = max_edge / float(longest_edge)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.array(Image.fromarray(image_rgb).resize(new_size, Image.Resampling.LANCZOS))


def _csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=4)
def _image_data_uri(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None

    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@st.cache_data(show_spinner=False, max_entries=256)
def _load_local_file_bytes(path_str: str, mtime_ns: int) -> bytes:
    _ = mtime_ns
    return Path(path_str).read_bytes()


def _discover_image_paths(folder_str: str) -> list[Path]:
    folder = Path(folder_str).expanduser()
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def _unique_export_stems(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    stems: list[str] = []
    for name in names:
        stem = Path(name).stem or "tray_analysis"
        count = seen.get(stem, 0)
        seen[stem] = count + 1
        stems.append(stem if count == 0 else f"{stem}_{count + 1}")
    return stems


def _build_batch_payload(
    file_items: list[dict[str, Any]],
    tray_profile_key: str,
    tray_long_side_cm: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    plant_frames = []
    leaf_frames = []
    export_stems = _unique_export_stems([str(item["name"]) for item in file_items])

    for item, export_stem in zip(file_items, export_stems):
        image_bytes = item["bytes"]
        result = _analyze_upload(image_bytes, tray_profile_key, float(tray_long_side_cm))
        record = {
            "name": item["name"],
            "export_stem": export_stem,
            "bytes": image_bytes,
            "result": result,
            "source_path": item.get("source_path", ""),
        }
        records.append(record)

        plant_df = result.plant_summary_df.copy()
        plant_df.insert(0, "Image", item["name"])
        if item.get("source_path"):
            plant_df.insert(1, "Source Path", item["source_path"])
        plant_frames.append(plant_df)

        leaf_df = result.leaf_detail_df.copy()
        leaf_df.insert(0, "Image", item["name"])
        if item.get("source_path"):
            leaf_df.insert(1, "Source Path", item["source_path"])
        leaf_frames.append(leaf_df)

        image_rows.append(
            {
                "Image": item["name"],
                "Source Path": item.get("source_path", ""),
                "Tray Profile": result.tray_profile_name,
                "Grid Rows": result.grid_rows,
                "Grid Columns": result.grid_cols,
                "Estimated Leaves": int(result.plant_summary_df["Estimated Leaves"].sum()),
                "Total Canopy Area (px)": int(result.plant_summary_df["Canopy Area (px)"].sum()),
                "Total Canopy Area (cm^2)": round(float(result.plant_summary_df["Canopy Area (cm^2)"].fillna(0).sum()), 4),
                "Plants With Canopy": int((result.plant_summary_df["Canopy Area (px)"] > 0).sum()),
                "Tray Long Side (px)": result.tray_long_side_px,
                "Tray Long Side (cm)": result.tray_long_side_cm,
                "Pixels Per Cm": result.pixels_per_cm,
                "Mm Per Pixel": result.mm_per_pixel,
                "Segmentation": result.segmentation_source,
            }
        )

    image_summary_df = pd.DataFrame(image_rows)
    plant_summary_df = pd.concat(plant_frames, ignore_index=True) if plant_frames else pd.DataFrame()
    leaf_detail_df = pd.concat(leaf_frames, ignore_index=True) if leaf_frames else pd.DataFrame()

    return {
        "records": records,
        "image_summary_df": image_summary_df,
        "plant_summary_df": plant_summary_df,
        "leaf_detail_df": leaf_detail_df,
    }


def _write_batch_outputs(output_dir_str: str, batch_payload: dict[str, Any]) -> list[Path]:
    output_dir = Path(output_dir_str).expanduser()
    overlays_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    written_paths = [
        output_dir / "image_summary.csv",
        output_dir / "plant_summary.csv",
        output_dir / "leaf_details.csv",
    ]
    batch_payload["image_summary_df"].to_csv(written_paths[0], index=False)
    batch_payload["plant_summary_df"].to_csv(written_paths[1], index=False)
    batch_payload["leaf_detail_df"].to_csv(written_paths[2], index=False)

    for record in batch_payload["records"]:
        overlay_path = overlays_dir / f"{record['export_stem']}_overlay.png"
        Image.fromarray(record["result"].overlay_rgb).save(overlay_path)
        written_paths.append(overlay_path)

    return written_paths


def _render_record_detail(record: dict[str, Any]) -> None:
    result = record["result"]
    original_preview = _preview_from_bytes(record["bytes"], MAX_PREVIEW_EDGE)
    overlay_preview = _resize_for_preview(result.overlay_rgb, MAX_PREVIEW_EDGE)

    col_original, col_segmented, col_table = st.columns([1.2, 1.2, 0.9])

    with col_original:
        st.image(original_preview, caption=f"Original image: {record['name']}", use_container_width=True)

    with col_segmented:
        st.image(overlay_preview, caption="Segmented overlay", use_container_width=True)

    with col_table:
        st.subheader("Quick Summary")
        st.dataframe(result.counts_df, hide_index=True, use_container_width=True)
        st.metric("Total estimated leaves", int(result.plant_summary_df["Estimated Leaves"].sum()))
        st.metric("Total canopy area (cm^2)", round(float(result.plant_summary_df["Canopy Area (cm^2)"].fillna(0).sum()), 2))
        st.metric("Pixels / cm", result.pixels_per_cm if result.pixels_per_cm is not None else "n/a")
        st.caption(
            f"Layout: {result.tray_profile_name} | Segmentation: {result.segmentation_source} | "
            f"Tray long side: {result.tray_long_side_cm:.1f} cm = {result.tray_long_side_px:.1f} px"
        )

    export_col_1, export_col_2 = st.columns(2)
    with export_col_1:
        st.download_button(
            "Download Plant Summary CSV",
            data=_csv_bytes(result.plant_summary_df),
            file_name=f"{record['export_stem']}_plant_summary.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"single_plant_{record['export_stem']}",
        )
    with export_col_2:
        st.download_button(
            "Download Per-Leaf CSV",
            data=_csv_bytes(result.leaf_detail_df),
            file_name=f"{record['export_stem']}_leaf_details.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"single_leaf_{record['export_stem']}",
        )

    st.subheader("Plant Summary Traits")
    st.dataframe(result.plant_summary_df, hide_index=True, use_container_width=True)

    st.subheader("Per-Leaf Traits")
    st.dataframe(result.leaf_detail_df, hide_index=True, use_container_width=True, height=420)


def main() -> None:
    logo_data_uri = _image_data_uri(str(LOGO_PATH))
    title_col, logo_col = st.columns([0.92, 0.08])
    with title_col:
        st.title("Plant Tray Phenotyping Dashboard")
        st.caption(
            "Run potato, soybean, or Arabidopsis tray images, extract 2D plant and leaf traits, and export combined CSVs. "
            "Physical measurements are normalized from the detected tray long side."
        )
    with logo_col:
        if logo_data_uri is not None:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end; padding-top:0.35rem;">
                    <img src="{logo_data_uri}" alt="NPEC logo" style="width:64px; height:auto;" />
                </div>
                """,
                unsafe_allow_html=True,
            )

    settings_col_1, settings_col_2 = st.columns([1.3, 0.8])
    tray_profile_key = settings_col_1.selectbox(
        "Tray layout",
        options=[key for key, _ in TRAY_PROFILE_OPTIONS],
        index=next(
            (idx for idx, (key, _) in enumerate(TRAY_PROFILE_OPTIONS) if key == DEFAULT_TRAY_PROFILE_KEY),
            0,
        ),
        format_func=lambda key: TRAY_PROFILE_LABELS.get(key, key),
    )
    tray_long_side_cm = settings_col_2.number_input(
        "Tray long side (cm)",
        min_value=1.0,
        value=DEFAULT_TRAY_LONG_SIDE_CM,
        step=0.5,
    )
    if tray_profile_key == DEFAULT_TRAY_PROFILE_KEY:
        st.caption("Auto layout chooses between a 4-plant 2x2 tray and a 20-plant 4x5 Arabidopsis tray from the canopy pattern.")

    uploaded_files = st.file_uploader(
        "Tray images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )
    file_items: list[dict[str, Any]] = []
    for uploaded_file in uploaded_files or []:
        file_items.append({"name": uploaded_file.name, "bytes": uploaded_file.getvalue(), "source_path": ""})
    if not file_items:
        st.info("Upload one or more top-down tray images to analyze them together.")
        return

    with st.spinner(f"Analyzing {len(file_items)} image(s)..."):
        batch_payload = _build_batch_payload(
            file_items=file_items,
            tray_profile_key=tray_profile_key,
            tray_long_side_cm=float(tray_long_side_cm),
        )

    image_summary_df = batch_payload["image_summary_df"]
    plant_summary_df = batch_payload["plant_summary_df"]
    leaf_detail_df = batch_payload["leaf_detail_df"]

    summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
    with summary_col_1:
        st.metric("Images", int(len(batch_payload["records"])))
    with summary_col_2:
        st.metric("Total estimated leaves", int(image_summary_df["Estimated Leaves"].sum()))
    with summary_col_3:
        st.metric("Total canopy area (cm^2)", round(float(image_summary_df["Total Canopy Area (cm^2)"].sum()), 2))

    batch_export_col_1, batch_export_col_2, batch_export_col_3 = st.columns(3)
    with batch_export_col_1:
        st.download_button(
            "Download Image Summary CSV",
            data=_csv_bytes(image_summary_df),
            file_name="image_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with batch_export_col_2:
        st.download_button(
            "Download Batch Plant CSV",
            data=_csv_bytes(plant_summary_df),
            file_name="batch_plant_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with batch_export_col_3:
        st.download_button(
            "Download Batch Leaf CSV",
            data=_csv_bytes(leaf_detail_df),
            file_name="batch_leaf_details.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.subheader("Batch Image Summary")
    st.dataframe(image_summary_df, hide_index=True, use_container_width=True)

    selected_name = st.selectbox(
        "Image detail view",
        options=[record["name"] for record in batch_payload["records"]],
        index=0,
    )
    selected_record = next(record for record in batch_payload["records"] if record["name"] == selected_name)
    _render_record_detail(selected_record)

    with st.expander("Batch Plant Summary Table", expanded=(len(batch_payload["records"]) == 1)):
        st.dataframe(plant_summary_df, hide_index=True, use_container_width=True)

    with st.expander("Batch Per-Leaf Table", expanded=False):
        st.dataframe(leaf_detail_df, hide_index=True, use_container_width=True, height=420)

    st.caption("Upload mode supports one image or batches of many images directly in the browser.")
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; font-size:0.9rem; color:rgba(250,250,250,0.65);'>{FOOTER_TEXT}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
