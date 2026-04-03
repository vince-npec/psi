from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

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
ZIP_IMAGE_TYPES = sorted(ext.lstrip(".") for ext in SUPPORTED_IMAGE_EXTENSIONS)


def _open_rgb_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    return image.convert("RGB")


@st.cache_data(show_spinner=False, max_entries=32)
def _analyze_upload(image_bytes: bytes, tray_profile_key: str, tray_long_side_cm: float):
    image = _open_rgb_image(image_bytes)
    return analyze_tray_image(
        np.array(image),
        tray_profile_key=tray_profile_key,
        tray_long_side_cm=float(tray_long_side_cm),
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _preview_from_bytes(image_bytes: bytes, max_edge: int) -> np.ndarray:
    image = _open_rgb_image(image_bytes)
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


def _should_skip_zip_member(member_name: str) -> bool:
    normalized_path = Path(member_name.replace("\\", "/"))
    parts = normalized_path.parts
    if any(part == "__MACOSX" for part in parts):
        return True
    return normalized_path.name.startswith("._")


@st.cache_data(show_spinner=False, max_entries=32)
def _discover_zip_image_members(archive_bytes: bytes) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if _should_skip_zip_member(info.filename):
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            members.append(
                {
                    "member_name": info.filename,
                    "name": Path(info.filename).name,
                    "size_bytes": int(info.file_size),
                }
            )
    members.sort(key=lambda item: str(item["member_name"]).lower())
    return members


@st.cache_data(show_spinner=False, max_entries=2048)
def _extract_zip_member_bytes(archive_bytes: bytes, member_name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        return zf.read(member_name)


def _get_item_bytes(item: dict[str, Any]) -> bytes:
    if item["kind"] == "direct":
        return item["bytes"]
    return _extract_zip_member_bytes(item["archive_bytes"], str(item["member_name"]))


def _item_display_path(item: dict[str, Any]) -> str:
    source_path = str(item.get("source_path", "") or "").strip()
    if source_path:
        return source_path
    return str(item["name"])


def _record_label(record: dict[str, Any]) -> str:
    source_path = str(record.get("source_path", "") or "").strip()
    if source_path:
        return source_path
    return str(record["name"])


def _results_bundle_name(file_items: list[dict[str, Any]]) -> str:
    archive_names = sorted({str(item["archive_name"]) for item in file_items if item.get("kind") == "zip_member"})
    if archive_names and len(archive_names) == 1 and all(item.get("kind") == "zip_member" for item in file_items):
        return f"{Path(archive_names[0]).stem}_analysis_results.zip"
    if len(file_items) == 1:
        return f"{Path(str(file_items[0]['name'])).stem}_analysis_results.zip"
    return "plant_tray_analysis_results.zip"


def _safe_zip_path(path_str: str) -> str:
    parts = [part for part in Path(path_str.replace("\\", "/")).parts if part not in {"", ".", ".."}]
    return "/".join(parts) or "image"


def _png_bytes_from_array(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    height, width = label_map.shape[:2]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    label_ids = [int(value) for value in np.unique(label_map) if int(value) > 0]
    for idx, label_id in enumerate(label_ids):
        hue = int(round((180.0 * idx) / max(1, len(label_ids))))
        hsv_color = np.uint8([[[hue % 180, 210, 255]]])
        rgb[label_map == label_id] = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
    return rgb


def _build_mask_assets(result) -> dict[str, np.ndarray]:
    image_h, image_w = result.original_rgb.shape[:2]
    canopy_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    plant_label_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    leaf_label_mask = np.zeros((image_h, image_w), dtype=np.uint16)
    next_leaf_offset = 0

    for plant_idx, plant_result in enumerate(result.plant_results, start=1):
        x0, y0, x1, y1 = plant_result.bbox
        crop_canopy = plant_result.mask > 0
        canopy_mask[y0:y1, x0:x1][crop_canopy] = 255
        plant_label_mask[y0:y1, x0:x1][crop_canopy] = plant_idx

        crop_leaf_map = plant_result.leaf_label_map.astype(np.uint16)
        if np.any(crop_leaf_map > 0):
            crop_target = leaf_label_mask[y0:y1, x0:x1]
            active = crop_leaf_map > 0
            crop_target[active] = crop_leaf_map[active] + next_leaf_offset
            next_leaf_offset += int(np.max(crop_leaf_map))

    return {
        "canopy_mask": canopy_mask,
        "plant_label_mask": plant_label_mask,
        "leaf_label_mask": leaf_label_mask,
        "plant_color_mask": _colorize_label_map(plant_label_mask),
        "leaf_color_mask": _colorize_label_map(leaf_label_mask),
    }


def _build_results_bundle_bytes(batch_payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("image_summary.csv", _csv_bytes(batch_payload["image_summary_df"]))
        zf.writestr("plant_summary.csv", _csv_bytes(batch_payload["plant_summary_df"]))
        zf.writestr("leaf_details.csv", _csv_bytes(batch_payload["leaf_detail_df"]))

        for record in batch_payload["records"]:
            item = record["item"]
            image_bytes = _get_item_bytes(item)
            result = record["result"]
            export_stem = str(record["export_stem"])
            source_path = str(record.get("source_path", "") or record["name"])

            if item["kind"] == "zip_member":
                archive_stem = Path(str(item["archive_name"])).stem or "archive"
                original_rel = f"originals/{archive_stem}/{_safe_zip_path(str(item['member_name']))}"
            else:
                original_rel = f"originals/{_safe_zip_path(str(record['name']))}"
            zf.writestr(original_rel, image_bytes)

            zf.writestr(f"overlays/{export_stem}_overlay.png", _png_bytes_from_array(result.overlay_rgb))

            mask_assets = _build_mask_assets(result)
            zf.writestr(f"masks/{export_stem}_canopy_mask.png", _png_bytes_from_array(mask_assets["canopy_mask"]))
            zf.writestr(f"masks/{export_stem}_plant_labels.png", _png_bytes_from_array(mask_assets["plant_label_mask"]))
            zf.writestr(f"masks/{export_stem}_leaf_labels.png", _png_bytes_from_array(mask_assets["leaf_label_mask"]))
            zf.writestr(f"masks/{export_stem}_plant_color_mask.png", _png_bytes_from_array(mask_assets["plant_color_mask"]))
            zf.writestr(f"masks/{export_stem}_leaf_color_mask.png", _png_bytes_from_array(mask_assets["leaf_color_mask"]))

            zf.writestr(
                f"per_image_csv/{export_stem}_plant_summary.csv",
                _csv_bytes(result.plant_summary_df),
            )
            zf.writestr(
                f"per_image_csv/{export_stem}_leaf_details.csv",
                _csv_bytes(result.leaf_detail_df),
            )
            zf.writestr(
                f"manifests/{export_stem}.txt",
                "\n".join(
                    [
                        f"Image: {record['name']}",
                        f"Source: {source_path}",
                        f"Tray Profile: {result.tray_profile_name}",
                        f"Tray Long Side (cm): {result.tray_long_side_cm}",
                        f"Tray Long Side (px): {result.tray_long_side_px}",
                    ]
                )
                + "\n",
            )

    return buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=8)
def _build_batch_payload(
    file_items: list[dict[str, Any]],
    tray_profile_key: str,
    tray_long_side_cm: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    plant_frames = []
    leaf_frames = []
    skipped_items: list[dict[str, str]] = []
    export_stems = _unique_export_stems([str(item["name"]) for item in file_items])

    for item, export_stem in zip(file_items, export_stems):
        image_label = _item_display_path(item)
        try:
            image_bytes = _get_item_bytes(item)
            result = _analyze_upload(image_bytes, tray_profile_key, float(tray_long_side_cm))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            skipped_items.append(
                {
                    "Image": str(item["name"]),
                    "Source Path": image_label,
                    "Reason": f"Skipped invalid image data ({type(exc).__name__}).",
                }
            )
            continue
        except Exception as exc:
            skipped_items.append(
                {
                    "Image": str(item["name"]),
                    "Source Path": image_label,
                    "Reason": f"Processing failed ({type(exc).__name__}).",
                }
            )
            continue
        record = {
            "name": item["name"],
            "export_stem": export_stem,
            "item": item,
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
    skipped_df = pd.DataFrame(skipped_items)

    return {
        "records": records,
        "image_summary_df": image_summary_df,
        "plant_summary_df": plant_summary_df,
        "leaf_detail_df": leaf_detail_df,
        "skipped_df": skipped_df,
        "results_zip_bytes": _build_results_bundle_bytes(
            {
                "records": records,
                "image_summary_df": image_summary_df,
                "plant_summary_df": plant_summary_df,
                "leaf_detail_df": leaf_detail_df,
            }
        )
        if records
        else b"",
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
    original_preview = _preview_from_bytes(_get_item_bytes(record["item"]), MAX_PREVIEW_EDGE)
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
    uploaded_archives = st.file_uploader(
        "Tray image archives (.zip)",
        type=["zip"],
        accept_multiple_files=True,
        help=f"Upload one or more zip files containing tray images. Supported image formats inside the zip are {', '.join(ext.upper() for ext in ZIP_IMAGE_TYPES)}.",
    )
    file_items: list[dict[str, Any]] = []
    for uploaded_file in uploaded_files or []:
        file_items.append({"name": uploaded_file.name, "bytes": uploaded_file.getvalue(), "source_path": "", "kind": "direct"})
    zip_image_count = 0
    zip_archive_count = 0
    for uploaded_archive in uploaded_archives or []:
        archive_bytes = uploaded_archive.getvalue()
        try:
            archive_members = _discover_zip_image_members(archive_bytes)
        except zipfile.BadZipFile:
            st.warning(f"`{uploaded_archive.name}` is not a readable zip archive and was skipped.")
            continue
        if not archive_members:
            st.warning(f"`{uploaded_archive.name}` did not contain any supported tray images.")
            continue
        zip_archive_count += 1
        zip_image_count += len(archive_members)
        for member in archive_members:
            file_items.append(
                {
                    "name": member["name"],
                    "kind": "zip_member",
                    "archive_name": uploaded_archive.name,
                    "archive_bytes": archive_bytes,
                    "member_name": member["member_name"],
                    "source_path": f"{uploaded_archive.name}::{member['member_name']}",
                }
            )
    if zip_archive_count > 0:
        st.caption(f"Discovered {zip_image_count} image(s) across {zip_archive_count} uploaded zip archive(s).")
    if not file_items:
        st.info("Upload one or more top-down tray images or a zip archive of images to analyze them together.")
        return

    with st.spinner(f"Analyzing {len(file_items)} image(s)..."):
        batch_payload = _build_batch_payload(
            file_items=file_items,
            tray_profile_key=tray_profile_key,
            tray_long_side_cm=float(tray_long_side_cm),
        )
    results_bundle_name = _results_bundle_name(file_items)

    image_summary_df = batch_payload["image_summary_df"]
    plant_summary_df = batch_payload["plant_summary_df"]
    leaf_detail_df = batch_payload["leaf_detail_df"]
    skipped_df = batch_payload["skipped_df"]

    if not skipped_df.empty:
        st.warning(f"Skipped {len(skipped_df)} file(s) that could not be decoded or processed.")
        with st.expander("Skipped files", expanded=False):
            st.dataframe(skipped_df, hide_index=True, use_container_width=True)

    if not batch_payload["records"]:
        st.error("No valid tray images were available to analyze from the uploaded files.")
        return

    summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
    with summary_col_1:
        st.metric("Images", int(len(batch_payload["records"])))
    with summary_col_2:
        st.metric("Total estimated leaves", int(image_summary_df["Estimated Leaves"].sum()))
    with summary_col_3:
        st.metric("Total canopy area (cm^2)", round(float(image_summary_df["Total Canopy Area (cm^2)"].sum()), 2))
    with summary_col_1:
        if zip_archive_count > 0:
            st.caption(f"Zip inputs: {zip_archive_count}")

    batch_export_col_1, batch_export_col_2, batch_export_col_3, batch_export_col_4 = st.columns(4)
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
    with batch_export_col_4:
        st.download_button(
            "Download Full Results ZIP",
            data=batch_payload["results_zip_bytes"],
            file_name=results_bundle_name,
            mime="application/zip",
            use_container_width=True,
        )

    st.caption("The results zip contains combined CSVs, per-image CSVs, original images, overlays, and mask outputs for every processed image.")

    st.subheader("Batch Image Summary")
    st.dataframe(image_summary_df, hide_index=True, use_container_width=True)

    record_labels = [_record_label(record) for record in batch_payload["records"]]
    selected_name = st.selectbox(
        "Image detail view",
        options=record_labels,
        index=0,
    )
    selected_record = next(record for record in batch_payload["records"] if _record_label(record) == selected_name)
    _render_record_detail(selected_record)

    with st.expander("Batch Plant Summary Table", expanded=(len(batch_payload["records"]) == 1)):
        st.dataframe(plant_summary_df, hide_index=True, use_container_width=True)

    with st.expander("Batch Per-Leaf Table", expanded=False):
        st.dataframe(leaf_detail_df, hide_index=True, use_container_width=True, height=420)

    st.caption("Upload mode supports single images, large batches of images, and zip archives directly in the browser.")
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; font-size:0.9rem; color:rgba(250,250,250,0.65);'>{FOOTER_TEXT}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
