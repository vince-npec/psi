from __future__ import annotations

import base64
import hashlib
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
    from . import tray_analyzer
except ImportError:  # pragma: no cover - direct script execution fallback
    import tray_analyzer

tray_analyzer.MAX_PLANT_ANALYSIS_DIM = max(int(getattr(tray_analyzer, "MAX_PLANT_ANALYSIS_DIM", 0) or 0), 4000)
analyze_tray_image = tray_analyzer.analyze_tray_image
TRAY_PROFILE_OPTIONS = tray_analyzer.get_tray_profile_options()
TRAY_PROFILE_LABELS = {key: label for key, label in TRAY_PROFILE_OPTIONS}
CONTAINER_MODE_OPTIONS = tray_analyzer.get_container_mode_options()
CONTAINER_MODE_LABELS = {key: label for key, label in CONTAINER_MODE_OPTIONS}
DEFAULT_TRAY_PROFILE_KEY = getattr(tray_analyzer, "AUTO_TRAY_PROFILE_KEY", TRAY_PROFILE_OPTIONS[0][0])
CUSTOM_TRAY_PROFILE_KEY = getattr(tray_analyzer, "CUSTOM_TRAY_PROFILE_KEY", "custom")
SEEDLINGS_PROFILE_KEY = getattr(tray_analyzer, "SEEDLINGS_PROFILE_KEY", "seedlings")
DEFAULT_CONTAINER_MODE = getattr(tray_analyzer, "CONTAINER_MODE_AUTO", "auto")
DEFAULT_TRAY_LONG_SIDE_CM = float(getattr(tray_analyzer, "TRAY_LONG_SIDE_CM", 33.0))
ANALYSIS_KIND_SEEDLINGS = getattr(tray_analyzer, "ANALYSIS_KIND_SEEDLINGS", "seedlings")
FULL_IMAGE_CONTAINER_MODE = getattr(tray_analyzer, "CONTAINER_MODE_FULL_IMAGE", "full_image")

SEEDLING_WORKFLOW_SINGLE = "single"
SEEDLING_WORKFLOW_LINKED = "linked_multiview"


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


@st.cache_data(show_spinner=False, max_entries=12)
def _analyze_upload(
    image_bytes: bytes,
    tray_profile_key: str,
    tray_long_side_cm: float,
    pixels_per_cm_override: float | None = None,
    custom_grid_rows: int | None = None,
    custom_grid_cols: int | None = None,
    custom_outer_pad_ratio: float | None = None,
    custom_site_pad_ratio: float | None = None,
    container_mode: str = DEFAULT_CONTAINER_MODE,
    circle_center_x_shift_ratio: float = 0.0,
    circle_center_y_shift_ratio: float = 0.0,
    circle_radius_scale: float = 1.0,
):
    image = _open_rgb_image(image_bytes)
    return analyze_tray_image(
        np.array(image),
        tray_profile_key=tray_profile_key,
        tray_long_side_cm=float(tray_long_side_cm),
        pixels_per_cm_override=pixels_per_cm_override,
        custom_grid_rows=custom_grid_rows,
        custom_grid_cols=custom_grid_cols,
        custom_outer_pad_ratio=custom_outer_pad_ratio,
        custom_site_pad_ratio=custom_site_pad_ratio,
        container_mode=container_mode,
        circle_center_x_shift_ratio=circle_center_x_shift_ratio,
        circle_center_y_shift_ratio=circle_center_y_shift_ratio,
        circle_radius_scale=circle_radius_scale,
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


def _numeric_series_sum(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    if series.notna().sum() == 0:
        return None
    return float(series.fillna(0).sum())


def _numeric_series_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    if series.notna().sum() == 0:
        return None
    return float(series.dropna().mean())


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


@st.cache_data(show_spinner=False, max_entries=128)
def _extract_zip_member_bytes(archive_bytes: bytes, member_name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        return zf.read(member_name)


def _get_item_bytes(item: dict[str, Any]) -> bytes:
    if item["kind"] == "direct":
        return item["bytes"]
    return _extract_zip_member_bytes(item["archive_bytes"], str(item["member_name"]))


def _coerce_tray_long_side_cm(value: Any, fallback: float) -> float:
    try:
        tray_long_side_cm = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(tray_long_side_cm) or tray_long_side_cm <= 0:
        return float(fallback)
    return tray_long_side_cm


def _coerce_optional_pixels_per_cm(value: Any) -> float | None:
    if value is None:
        return None
    try:
        pixels_per_cm = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(pixels_per_cm) or pixels_per_cm <= 0:
        return None
    return pixels_per_cm


def _coerce_circle_shift_pct(value: Any) -> float:
    try:
        shift_pct = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(shift_pct):
        return 0.0
    return float(min(75.0, max(-75.0, shift_pct)))


def _coerce_circle_size_pct(value: Any) -> float:
    try:
        size_pct = float(value)
    except (TypeError, ValueError):
        return 100.0
    if not np.isfinite(size_pct):
        return 100.0
    return float(min(300.0, max(20.0, size_pct)))


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


def _build_scale_editor_df(file_items: list[dict[str, Any]], default_tray_long_side_cm: float) -> pd.DataFrame:
    tray_overrides = st.session_state.setdefault("per_image_tray_long_side_cm", {})
    pixels_per_cm_overrides = st.session_state.setdefault("per_image_pixels_per_cm_override", {})
    active_keys: list[str] = []
    rows: list[dict[str, Any]] = []

    for idx, item in enumerate(file_items):
        item_key = f"{idx}::{_item_display_path(item)}"
        active_keys.append(item_key)
        resolved_scale = _coerce_tray_long_side_cm(tray_overrides.get(item_key, default_tray_long_side_cm), default_tray_long_side_cm)
        rows.append(
            {
                "_Item Key": item_key,
                "Image": str(item["name"]),
                "Source Path": _item_display_path(item),
                "Tray Long Side (cm)": resolved_scale,
                "Pixels Per Cm Override": _coerce_optional_pixels_per_cm(pixels_per_cm_overrides.get(item_key)),
            }
        )

    for stale_key in [key for key in list(tray_overrides) if key not in active_keys]:
        tray_overrides.pop(stale_key, None)
    for stale_key in [key for key in list(pixels_per_cm_overrides) if key not in active_keys]:
        pixels_per_cm_overrides.pop(stale_key, None)

    return pd.DataFrame(rows)


def _build_circle_adjustment_editor_df(file_items: list[dict[str, Any]]) -> pd.DataFrame:
    x_shift_overrides = st.session_state.setdefault("per_image_circle_center_x_shift_pct", {})
    y_shift_overrides = st.session_state.setdefault("per_image_circle_center_y_shift_pct", {})
    radius_scale_overrides = st.session_state.setdefault("per_image_circle_radius_scale_pct", {})
    active_keys: list[str] = []
    rows: list[dict[str, Any]] = []

    for idx, item in enumerate(file_items):
        item_key = f"{idx}::{_item_display_path(item)}"
        active_keys.append(item_key)
        rows.append(
            {
                "_Item Key": item_key,
                "Image": str(item["name"]),
                "Source Path": _item_display_path(item),
                "Circle X Shift (%)": _coerce_circle_shift_pct(x_shift_overrides.get(item_key, 0.0)),
                "Circle Y Shift (%)": _coerce_circle_shift_pct(y_shift_overrides.get(item_key, 0.0)),
                "Circle Size (%)": _coerce_circle_size_pct(radius_scale_overrides.get(item_key, 100.0)),
            }
        )

    for stale_key in [key for key in list(x_shift_overrides) if key not in active_keys]:
        x_shift_overrides.pop(stale_key, None)
    for stale_key in [key for key in list(y_shift_overrides) if key not in active_keys]:
        y_shift_overrides.pop(stale_key, None)
    for stale_key in [key for key in list(radius_scale_overrides) if key not in active_keys]:
        radius_scale_overrides.pop(stale_key, None)

    return pd.DataFrame(rows)


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
    image_h, image_w = result.image_shape
    canopy_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    plant_label_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    leaf_label_mask = np.zeros((image_h, image_w), dtype=np.uint16)
    shoot_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    root_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    primary_root_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    lateral_root_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    root_skeleton_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    next_leaf_offset = 0

    for plant_idx, plant_result in enumerate(result.plant_results, start=1):
        x0, y0, x1, y1 = plant_result.bbox
        crop_canopy = plant_result.mask > 0
        canopy_mask[y0:y1, x0:x1][crop_canopy] = 255
        plant_label_mask[y0:y1, x0:x1][crop_canopy] = plant_idx
        if plant_result.shoot_mask is not None:
            shoot_mask[y0:y1, x0:x1][plant_result.shoot_mask > 0] = 255
        if plant_result.root_mask is not None:
            root_mask[y0:y1, x0:x1][plant_result.root_mask > 0] = 255
        if plant_result.primary_root_mask is not None:
            primary_root_mask[y0:y1, x0:x1][plant_result.primary_root_mask > 0] = 255
        if plant_result.lateral_root_mask is not None:
            lateral_root_mask[y0:y1, x0:x1][plant_result.lateral_root_mask > 0] = 255
        if plant_result.root_skeleton_mask is not None:
            root_skeleton_mask[y0:y1, x0:x1][plant_result.root_skeleton_mask > 0] = 255

        crop_leaf_map = plant_result.leaf_label_map.astype(np.uint16)
        if np.any(crop_leaf_map > 0):
            crop_target = leaf_label_mask[y0:y1, x0:x1]
            active = crop_leaf_map > 0
            crop_target[active] = crop_leaf_map[active] + next_leaf_offset
            next_leaf_offset += int(np.max(crop_leaf_map))

    mask_assets = {
        "canopy_mask": canopy_mask,
        "plant_label_mask": plant_label_mask,
        "leaf_label_mask": leaf_label_mask,
        "plant_color_mask": _colorize_label_map(plant_label_mask),
        "leaf_color_mask": _colorize_label_map(leaf_label_mask),
    }
    if result.analysis_kind == ANALYSIS_KIND_SEEDLINGS or np.any(root_mask > 0):
        mask_assets.update(
            {
                "shoot_mask": shoot_mask if np.any(shoot_mask > 0) else canopy_mask.copy(),
                "root_mask": root_mask,
                "primary_root_mask": primary_root_mask,
                "lateral_root_mask": lateral_root_mask,
                "root_skeleton_mask": root_skeleton_mask,
            }
        )
    return mask_assets


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
            if "shoot_mask" in mask_assets:
                zf.writestr(f"masks/{export_stem}_shoot_mask.png", _png_bytes_from_array(mask_assets["shoot_mask"]))
            if "root_mask" in mask_assets:
                zf.writestr(f"masks/{export_stem}_root_mask.png", _png_bytes_from_array(mask_assets["root_mask"]))
            if "primary_root_mask" in mask_assets:
                zf.writestr(
                    f"masks/{export_stem}_primary_root_mask.png",
                    _png_bytes_from_array(mask_assets["primary_root_mask"]),
                )
            if "lateral_root_mask" in mask_assets:
                zf.writestr(
                    f"masks/{export_stem}_lateral_root_mask.png",
                    _png_bytes_from_array(mask_assets["lateral_root_mask"]),
                )
            if "root_skeleton_mask" in mask_assets:
                zf.writestr(
                    f"masks/{export_stem}_root_skeleton_mask.png",
                    _png_bytes_from_array(mask_assets["root_skeleton_mask"]),
                )

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
                        f"Analysis Kind: {getattr(result, 'analysis_kind', '')}",
                        f"Container Source: {result.container_source}",
                        f"Circle X Shift (%): {round(float(result.circle_center_x_shift_ratio) * 100.0, 2)}",
                        f"Circle Y Shift (%): {round(float(result.circle_center_y_shift_ratio) * 100.0, 2)}",
                        f"Circle Size (%): {round(float(result.circle_radius_scale) * 100.0, 2)}",
                        f"Scale Source: {result.scale_source}",
                        f"Tray Long Side (cm): {result.tray_long_side_cm}",
                        f"Tray Long Side (px): {result.tray_long_side_px}",
                        f"Pixels Per Cm Override: {result.pixels_per_cm_override}",
                        f"Pixels Per Cm: {result.pixels_per_cm}",
                        f"Mm Per Pixel: {result.mm_per_pixel}",
                    ]
                )
                + "\n",
            )

    return buffer.getvalue()


def _build_batch_payload(
    file_items: list[dict[str, Any]],
    tray_profile_key: str,
    tray_long_side_values_cm: tuple[float, ...],
    pixels_per_cm_override_values: tuple[float | None, ...],
    custom_grid_rows: int | None = None,
    custom_grid_cols: int | None = None,
    custom_outer_pad_ratio: float | None = None,
    custom_site_pad_ratio: float | None = None,
    container_mode: str = DEFAULT_CONTAINER_MODE,
    circle_center_x_shift_ratios: tuple[float, ...] = (),
    circle_center_y_shift_ratios: tuple[float, ...] = (),
    circle_radius_scales: tuple[float, ...] = (),
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    plant_frames = []
    leaf_frames = []
    skipped_items: list[dict[str, str]] = []
    export_stems = _unique_export_stems([str(item["name"]) for item in file_items])
    resolved_tray_long_side_values_cm = list(tray_long_side_values_cm)
    if len(resolved_tray_long_side_values_cm) != len(file_items):
        resolved_tray_long_side_values_cm = [DEFAULT_TRAY_LONG_SIDE_CM] * len(file_items)
    resolved_pixels_per_cm_override_values = list(pixels_per_cm_override_values)
    if len(resolved_pixels_per_cm_override_values) != len(file_items):
        resolved_pixels_per_cm_override_values = [None] * len(file_items)
    resolved_circle_center_x_shift_ratios = list(circle_center_x_shift_ratios)
    if len(resolved_circle_center_x_shift_ratios) != len(file_items):
        resolved_circle_center_x_shift_ratios = [0.0] * len(file_items)
    resolved_circle_center_y_shift_ratios = list(circle_center_y_shift_ratios)
    if len(resolved_circle_center_y_shift_ratios) != len(file_items):
        resolved_circle_center_y_shift_ratios = [0.0] * len(file_items)
    resolved_circle_radius_scales = list(circle_radius_scales)
    if len(resolved_circle_radius_scales) != len(file_items):
        resolved_circle_radius_scales = [1.0] * len(file_items)

    for item, export_stem, tray_long_side_cm, pixels_per_cm_override, circle_center_x_shift_ratio, circle_center_y_shift_ratio, circle_radius_scale in zip(
        file_items,
        export_stems,
        resolved_tray_long_side_values_cm,
        resolved_pixels_per_cm_override_values,
        resolved_circle_center_x_shift_ratios,
        resolved_circle_center_y_shift_ratios,
        resolved_circle_radius_scales,
    ):
        image_label = _item_display_path(item)
        try:
            image_bytes = _get_item_bytes(item)
            result = _analyze_upload(
                image_bytes,
                tray_profile_key,
                _coerce_tray_long_side_cm(tray_long_side_cm, DEFAULT_TRAY_LONG_SIDE_CM),
                _coerce_optional_pixels_per_cm(pixels_per_cm_override),
                custom_grid_rows=custom_grid_rows,
                custom_grid_cols=custom_grid_cols,
                custom_outer_pad_ratio=custom_outer_pad_ratio,
                custom_site_pad_ratio=custom_site_pad_ratio,
                container_mode=container_mode,
                circle_center_x_shift_ratio=float(circle_center_x_shift_ratio),
                circle_center_y_shift_ratio=float(circle_center_y_shift_ratio),
                circle_radius_scale=float(circle_radius_scale),
            )
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

        image_row = {
            "Image": item["name"],
            "Source Path": item.get("source_path", ""),
            "Tray Profile": result.tray_profile_name,
            "Analysis Kind": getattr(result, "analysis_kind", ""),
            "Grid Rows": result.grid_rows,
            "Grid Columns": result.grid_cols,
            "Container Source": result.container_source,
            "Circle X Shift (%)": round(float(result.circle_center_x_shift_ratio) * 100.0, 2),
            "Circle Y Shift (%)": round(float(result.circle_center_y_shift_ratio) * 100.0, 2),
            "Circle Size (%)": round(float(result.circle_radius_scale) * 100.0, 2),
            "Scale Source": result.scale_source,
            "Estimated Leaves": int(result.plant_summary_df["Estimated Leaves"].sum()),
            "Total Canopy Area (px)": int(result.plant_summary_df["Canopy Area (px)"].sum()),
            "Total Canopy Area (cm^2)": round(float(result.plant_summary_df["Canopy Area (cm^2)"].fillna(0).sum()), 4),
            "Plants With Canopy": int((result.plant_summary_df["Canopy Area (px)"] > 0).sum()),
            "Tray Long Side (px)": result.tray_long_side_px,
            "Tray Long Side (cm)": result.tray_long_side_cm,
            "Pixels Per Cm": result.pixels_per_cm,
            "Pixels Per Cm Override": result.pixels_per_cm_override,
            "Mm Per Pixel": result.mm_per_pixel,
            "Segmentation": result.segmentation_source,
        }
        if "Total Root Length (cm)" in result.plant_summary_df.columns:
            image_row["Total Root Length (cm)"] = round(float(result.plant_summary_df["Total Root Length (cm)"].fillna(0).sum()), 4)
        if "Root Area (cm^2)" in result.plant_summary_df.columns:
            image_row["Total Root Area (cm^2)"] = round(float(result.plant_summary_df["Root Area (cm^2)"].fillna(0).sum()), 4)
        if "Lateral Root Count" in result.plant_summary_df.columns:
            image_row["Total Lateral Root Count"] = int(result.plant_summary_df["Lateral Root Count"].fillna(0).sum())
        image_rows.append(image_row)

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
    }


def _results_zip_signature(batch_payload: dict[str, Any], results_bundle_name: str) -> str:
    digest = hashlib.sha1()
    digest.update(results_bundle_name.encode("utf-8"))
    for key in ("image_summary_df", "plant_summary_df", "leaf_detail_df", "skipped_df"):
        csv_bytes = _csv_bytes(batch_payload[key]) if key in batch_payload and isinstance(batch_payload[key], pd.DataFrame) else b""
        digest.update(key.encode("utf-8"))
        digest.update(csv_bytes)
    for record in batch_payload.get("records", []):
        digest.update(str(record.get("name", "")).encode("utf-8"))
        digest.update(str(record.get("source_path", "")).encode("utf-8"))
        digest.update(str(record["result"].container_source).encode("utf-8"))
        digest.update(str(record["result"].scale_source).encode("utf-8"))
    return digest.hexdigest()


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


def _collect_uploaded_items(
    uploaded_files: list[Any] | None,
    uploaded_archives: list[Any] | None,
    image_label: str,
) -> tuple[list[dict[str, Any]], int, int]:
    file_items: list[dict[str, Any]] = []
    for uploaded_file in uploaded_files or []:
        file_items.append(
            {
                "name": uploaded_file.name,
                "bytes": uploaded_file.getvalue(),
                "source_path": "",
                "kind": "direct",
            }
        )

    zip_image_count = 0
    zip_archive_count = 0
    for uploaded_archive in uploaded_archives or []:
        archive_bytes = uploaded_archive.getvalue()
        try:
            archive_members = _discover_zip_image_members(archive_bytes)
        except zipfile.BadZipFile:
            st.warning(f"`{uploaded_archive.name}` is not a readable zip archive and was skipped from `{image_label}`.")
            continue
        if not archive_members:
            st.warning(f"`{uploaded_archive.name}` did not contain any supported images for `{image_label}`.")
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

    return file_items, zip_archive_count, zip_image_count


def _resolve_manual_scale_calibration(pixels_per_cm_value: float | None) -> tuple[float | None, float | None, str]:
    pixels_per_cm = _coerce_optional_pixels_per_cm(pixels_per_cm_value)
    if pixels_per_cm is None:
        return None, None, "Uncalibrated (manual pixels/cm not provided)"
    return pixels_per_cm, round(10.0 / float(pixels_per_cm), 6), "Manual pixels/cm override"


def _cluster_columns_1d(
    x_values: np.ndarray,
    cluster_count: int,
    initial_centers: np.ndarray | None = None,
) -> np.ndarray:
    if cluster_count <= 0:
        return np.zeros(0, dtype=np.float32)
    xs = np.asarray(x_values, dtype=np.float32).reshape(-1)
    if xs.size == 0:
        return np.zeros(cluster_count, dtype=np.float32)

    if initial_centers is not None and len(initial_centers) == cluster_count:
        centers = np.asarray(initial_centers, dtype=np.float32).reshape(-1)
    else:
        quantiles = np.linspace(0.08, 0.92, cluster_count)
        centers = np.quantile(xs, quantiles).astype(np.float32)

    min_x = float(xs.min())
    max_x = float(xs.max())
    centers = np.clip(centers, min_x, max_x)

    for _ in range(40):
        distances = np.abs(xs[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for cluster_idx in range(cluster_count):
            cluster_pixels = xs[labels == cluster_idx]
            if cluster_pixels.size > 0:
                new_centers[cluster_idx] = float(cluster_pixels.mean())
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return np.sort(centers.astype(np.float32))


def _extract_seedling_x_regions(
    mask_u8: np.ndarray,
    expected_seedling_count: int,
    y_min: int | None = None,
    y_max: int | None = None,
    initial_centers: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if mask_u8.size == 0 or expected_seedling_count <= 0:
        return [], np.zeros(0, dtype=np.float32)

    filtered_mask = mask_u8.copy()
    height, width = filtered_mask.shape[:2]
    if y_min is not None and y_min > 0:
        filtered_mask[: max(0, min(height, int(y_min))), :] = 0
    if y_max is not None and y_max < height - 1:
        filtered_mask[max(0, min(height, int(y_max) + 1)) :, :] = 0

    ys, xs = np.where(filtered_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [], np.zeros(expected_seedling_count, dtype=np.float32)

    centers = _cluster_columns_1d(xs, expected_seedling_count, initial_centers=initial_centers)
    if centers.size == 0:
        return [], np.zeros(expected_seedling_count, dtype=np.float32)

    distances = np.abs(xs[:, None].astype(np.float32) - centers[None, :])
    labels = np.argmin(distances, axis=1)
    min_cluster_pixels = max(120, int(xs.size * 0.018 / max(1, expected_seedling_count)))
    regions: list[dict[str, Any]] = []

    for cluster_idx in range(expected_seedling_count):
        selected = labels == cluster_idx
        if int(np.count_nonzero(selected)) < min_cluster_pixels:
            continue
        cluster_xs = xs[selected]
        cluster_ys = ys[selected]
        cluster_mask_u8 = np.zeros_like(filtered_mask, dtype=np.uint8)
        cluster_mask_u8[cluster_ys, cluster_xs] = 255
        x0 = int(cluster_xs.min())
        y0 = int(cluster_ys.min())
        x1 = int(cluster_xs.max()) + 1
        y1 = int(cluster_ys.max()) + 1
        anchor_band = cluster_xs[cluster_ys >= (cluster_ys.max() - 4)]
        anchor_x = int(np.median(anchor_band)) if anchor_band.size > 0 else int(round(float(cluster_xs.mean())))
        anchor_y = int(cluster_ys.max())
        regions.append(
            {
                "bbox": (x0, y0, x1, y1),
                "mask_u8": cluster_mask_u8,
                "anchor": (anchor_x, anchor_y),
                "center_x": float(cluster_xs.mean()),
                "center_y": float(cluster_ys.mean()),
                "site_index": int(cluster_idx + 1),
                "row_index": 0,
                "col_index": int(cluster_idx),
                "name": f"Seedling {cluster_idx + 1}",
                "position": tray_analyzer._format_grid_position(0, cluster_idx, 1, expected_seedling_count),
            }
        )

    center_ratios = (centers / float(max(1, width))).astype(np.float32)
    return regions, center_ratios


def _extract_seedling_component_regions(
    mask_u8: np.ndarray,
    expected_seedling_count: int,
    y_min: int | None = None,
    y_max: int | None = None,
    initial_centers: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if mask_u8.size == 0 or expected_seedling_count <= 0:
        return [], np.zeros(0, dtype=np.float32)

    filtered_mask = mask_u8.copy()
    height, width = filtered_mask.shape[:2]
    if y_min is not None and y_min > 0:
        filtered_mask[: max(0, min(height, int(y_min))), :] = 0
    if y_max is not None and y_max < height - 1:
        filtered_mask[max(0, min(height, int(y_max) + 1)) :, :] = 0

    binary = (filtered_mask > 0).astype(np.uint8)
    total_area_px = int(np.count_nonzero(binary))
    if total_area_px <= 0:
        return [], np.zeros(expected_seedling_count, dtype=np.float32)

    min_component_area_px = max(600, int(total_area_px / max(1, expected_seedling_count * 12)))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[dict[str, Any]] = []
    for label_idx in range(1, num_labels):
        area_px = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area_px < min_component_area_px:
            continue
        x0 = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_idx, cv2.CC_STAT_TOP])
        width_px = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height_px = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if width_px < 24 or height_px < 24:
            continue
        component_mask = (labels == label_idx)
        ys, xs = np.where(component_mask)
        if xs.size == 0 or ys.size == 0:
            continue
        anchor_band = xs[ys >= (ys.max() - 4)]
        anchor_x = int(np.median(anchor_band)) if anchor_band.size > 0 else int(round(float(centroids[label_idx, 0])))
        anchor_y = int(ys.max())
        mask_component_u8 = (component_mask.astype(np.uint8) * 255)
        components.append(
            {
                "bbox": (x0, y0, x0 + width_px, y0 + height_px),
                "mask_u8": mask_component_u8,
                "anchor": (anchor_x, anchor_y),
                "center_x": float(xs.mean()),
                "center_y": float(ys.mean()),
                "area_px": area_px,
            }
        )

    if not components:
        return [], np.zeros(expected_seedling_count, dtype=np.float32)

    if initial_centers is not None and len(initial_centers) == expected_seedling_count and len(components) >= expected_seedling_count:
        remaining = components.copy()
        selected: list[dict[str, Any]] = []
        for target_x in np.sort(initial_centers.astype(np.float32)):
            best_idx = min(
                range(len(remaining)),
                key=lambda idx: (
                    abs(float(remaining[idx]["center_x"]) - float(target_x)),
                    -int(remaining[idx]["area_px"]),
                ),
            )
            selected.append(remaining.pop(best_idx))
            if not remaining:
                break
        components = selected

    if len(components) != expected_seedling_count:
        return [], np.zeros(expected_seedling_count, dtype=np.float32)

    components.sort(key=lambda item: float(item["center_x"]))
    regions: list[dict[str, Any]] = []
    for component_idx, component in enumerate(components):
        component["site_index"] = int(component_idx + 1)
        component["row_index"] = 0
        component["col_index"] = int(component_idx)
        component["name"] = f"Seedling {component_idx + 1}"
        component["position"] = tray_analyzer._format_grid_position(0, component_idx, 1, expected_seedling_count)
        regions.append(component)

    center_ratios = (
        np.asarray([float(component["center_x"]) for component in regions], dtype=np.float32) / float(max(1, width))
    ).astype(np.float32)
    return regions, center_ratios


def _segment_linked_seedling_side_shoot_mask(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.size == 0:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    exg = (2 * rgb[:, :, 1]) - rgb[:, :, 0] - rgb[:, :, 2]
    hsv_mask = cv2.inRange(
        hsv,
        np.array([28, 60, 55], dtype=np.uint8),
        np.array([90, 255, 255], dtype=np.uint8),
    )
    exg_mask = ((exg > 22).astype(np.uint8) * 255)
    mask = cv2.bitwise_and(hsv_mask, exg_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)

    binary = (mask > 0).astype(np.uint8)
    total_area_px = int(np.count_nonzero(binary))
    if total_area_px <= 0:
        return np.zeros_like(mask)
    min_component_area_px = max(700, int(total_area_px * 0.035))
    filtered_mask = np.zeros_like(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for label_idx in range(1, num_labels):
        area_px = int(stats[label_idx, cv2.CC_STAT_AREA])
        width_px = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height_px = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if area_px < min_component_area_px or width_px < 50 or height_px < 60:
            continue
        filtered_mask[labels == label_idx] = 255
    return filtered_mask


@st.cache_data(show_spinner=False, max_entries=24)
def _analyze_linked_seedling_top_view(
    image_bytes: bytes,
    expected_seedling_count: int,
    pixels_per_cm_override: float | None,
) -> dict[str, Any]:
    image_rgb = np.array(_open_rgb_image(image_bytes))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    shoot_mask_u8 = tray_analyzer._segment_seedling_shoot_mask(image_bgr)
    regions, center_ratios = _extract_seedling_component_regions(
        shoot_mask_u8,
        expected_seedling_count=expected_seedling_count,
    )
    if len(regions) != expected_seedling_count:
        regions, center_ratios = _extract_seedling_x_regions(
            shoot_mask_u8,
            expected_seedling_count=expected_seedling_count,
        )
    pixels_per_cm, mm_per_pixel, scale_source = _resolve_manual_scale_calibration(pixels_per_cm_override)
    tray_profile = tray_analyzer.TrayProfile(
        key="seedling_top_linked",
        name="Seedling top view",
        rows=1,
        cols=max(1, expected_seedling_count),
    )

    overlay_bgr = image_bgr.copy()
    plant_results = []
    plant_rows: list[dict[str, Any]] = []
    leaf_rows: list[dict[str, Any]] = []

    for region_idx, region in enumerate(regions):
        x0, y0, x1, y1 = region["bbox"]
        pad_x = max(14, int(round((x1 - x0) * 0.08)))
        pad_y = max(14, int(round((y1 - y0) * 0.08)))
        crop_x0 = max(0, x0 - pad_x)
        crop_y0 = max(0, y0 - pad_y)
        crop_x1 = min(image_bgr.shape[1], x1 + pad_x)
        crop_y1 = min(image_bgr.shape[0], y1 + pad_y)
        crop_bgr = image_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
        if "mask_u8" in region:
            crop_mask_u8 = region["mask_u8"][crop_y0:crop_y1, crop_x0:crop_x1].copy()
        else:
            crop_mask_u8 = shoot_mask_u8[crop_y0:crop_y1, crop_x0:crop_x1].copy()

        leaf_label_map, leaf_count, canopy_area_px, min_leaf_area_px = tray_analyzer._estimate_leaf_instances(
            crop_mask_u8,
            crop_bgr,
            expected_groups=1,
        )
        site = tray_analyzer.PlantRegion(
            name=str(region["name"]),
            position=str(region["position"]),
            site_index=int(region["site_index"]),
            row_index=0,
            col_index=int(region["col_index"]),
            bbox=(x0, y0, x1, y1),
        )
        plant_traits, plant_leaf_rows = tray_analyzer._compute_trait_rows(
            crop_bgr=crop_bgr,
            mask_u8=crop_mask_u8,
            leaf_label_map=leaf_label_map,
            site=site,
            analysis_bbox=(crop_x0, crop_y0, crop_x1, crop_y1),
            min_leaf_area_px=min_leaf_area_px,
            tray_profile=tray_profile,
            container_source="Linked seedling top view",
            circle_center_x_shift_ratio=0.0,
            circle_center_y_shift_ratio=0.0,
            circle_radius_scale=1.0,
            tray_long_side_cm=0.0,
            tray_long_side_px=0.0,
            pixels_per_cm=pixels_per_cm,
            pixels_per_cm_override=pixels_per_cm,
            mm_per_pixel=mm_per_pixel,
            scale_source=scale_source,
        )
        plant_traits["Analysis Kind"] = "seedling_top_view"
        plant_traits["View"] = "Top view shoots"
        for leaf_row in plant_leaf_rows:
            leaf_row["Analysis Kind"] = "seedling_top_view"
            leaf_row["View"] = "Top view shoots"

        plant_result = tray_analyzer.PlantLeafResult(
            name=site.name,
            position=site.position,
            bbox=(crop_x0, crop_y0, crop_x1, crop_y1),
            site_bbox=(x0, y0, x1, y1),
            leaf_count=int(leaf_count),
            canopy_area_px=int(canopy_area_px),
            min_leaf_area_px=int(min_leaf_area_px),
            mask=crop_mask_u8,
            leaf_label_map=leaf_label_map,
            plant_traits=plant_traits,
            leaf_traits=plant_leaf_rows,
            shoot_mask=crop_mask_u8,
        )
        plant_results.append(plant_result)
        plant_rows.append(plant_traits)
        leaf_rows.extend(plant_leaf_rows)
        tray_analyzer._draw_region_overlay(
            overlay_bgr,
            plant_result,
            color=tray_analyzer._plant_color_bgr(region_idx, max(1, expected_seedling_count)),
        )

    plant_summary_df = pd.DataFrame(plant_rows)
    leaf_detail_df = pd.DataFrame(leaf_rows)
    counts_columns = ["Plant", "Position", "Estimated Leaves", "Canopy Area (cm^2)"]
    counts_df = plant_summary_df[[column for column in counts_columns if column in plant_summary_df.columns]].copy()

    return {
        "image_shape": tuple(int(v) for v in image_rgb.shape[:2]),
        "overlay_rgb": cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
        "plant_summary_df": plant_summary_df,
        "leaf_detail_df": leaf_detail_df,
        "counts_df": counts_df,
        "shoot_mask_u8": shoot_mask_u8,
        "pixels_per_cm": pixels_per_cm,
        "pixels_per_cm_override": pixels_per_cm,
        "mm_per_pixel": mm_per_pixel,
        "scale_source": scale_source,
        "center_ratios": tuple(float(value) for value in center_ratios.tolist()),
        "detected_seedling_count": int(len(plant_rows)),
    }


def _detect_side_view_surface_y(image_bgr: np.ndarray) -> int:
    if image_bgr.size == 0:
        return 0

    working_bgr, scale = tray_analyzer._downscale_for_analysis(image_bgr, 1600)
    gray = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    row_score = np.mean(np.abs(grad_y), axis=1)
    dark_threshold = float(np.percentile(gray, 45))
    dark_fraction = np.mean(gray < dark_threshold, axis=1)
    row_score = row_score * (0.55 + (0.45 * dark_fraction))
    search_start = int(row_score.size * 0.25)
    search_end = max(search_start + 1, int(row_score.size * 0.75))
    best_row = search_start + int(np.argmax(row_score[search_start:search_end]))
    if scale == 1.0:
        return int(best_row)
    return int(round(float(best_row) / float(scale)))


@st.cache_data(show_spinner=False, max_entries=24)
def _analyze_linked_seedling_side_view(
    image_bytes: bytes,
    expected_seedling_count: int,
    pixels_per_cm_override: float | None,
    initial_center_ratios: tuple[float, ...] = (),
) -> dict[str, Any]:
    image_rgb = np.array(_open_rgb_image(image_bytes))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    shoot_mask_u8 = _segment_linked_seedling_side_shoot_mask(image_bgr)
    surface_y = _detect_side_view_surface_y(image_bgr)
    expected_centers = None
    if initial_center_ratios and len(initial_center_ratios) == expected_seedling_count:
        expected_centers = np.asarray(initial_center_ratios, dtype=np.float32) * float(image_bgr.shape[1])
    regions, center_ratios = _extract_seedling_component_regions(
        shoot_mask_u8,
        expected_seedling_count=expected_seedling_count,
        y_min=int(surface_y * 0.7),
        y_max=min(image_bgr.shape[0] - 1, surface_y + max(40, int(round(image_bgr.shape[0] * 0.03)))),
        initial_centers=expected_centers,
    )
    if len(regions) != expected_seedling_count:
        regions, center_ratios = _extract_seedling_x_regions(
            shoot_mask_u8,
            expected_seedling_count=expected_seedling_count,
            y_min=int(surface_y * 0.7),
            y_max=min(image_bgr.shape[0] - 1, surface_y + max(40, int(round(image_bgr.shape[0] * 0.03)))),
            initial_centers=expected_centers,
        )
    pixels_per_cm, mm_per_pixel, scale_source = _resolve_manual_scale_calibration(pixels_per_cm_override)

    overlay_bgr = image_bgr.copy()
    cv2.line(
        overlay_bgr,
        (0, int(surface_y)),
        (overlay_bgr.shape[1] - 1, int(surface_y)),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    summary_rows: list[dict[str, Any]] = []
    for region_idx, region in enumerate(regions):
        region_mask_u8 = region["mask_u8"]
        ys, xs = np.where(region_mask_u8 > 0)
        if xs.size == 0 or ys.size == 0:
            continue

        top_y = int(region["bbox"][1])
        x0, _, x1, _ = region["bbox"]
        visible_height_px = max(0.0, float(surface_y - top_y))
        visible_width_px = float(max(0, x1 - x0))
        visible_area_px = float(xs.size)
        center_x = int(round(float(xs.mean())))
        mask_bool = region_mask_u8 > 0
        color_stats = tray_analyzer._masked_color_stats(image_bgr, mask_bool)

        color = tray_analyzer._plant_color_bgr(region_idx, max(1, expected_seedling_count))
        overlay_layer = np.zeros_like(overlay_bgr)
        overlay_layer[:] = color
        overlay_bgr[mask_bool] = cv2.addWeighted(overlay_bgr[mask_bool], 0.72, overlay_layer[mask_bool], 0.28, 0)
        cv2.rectangle(overlay_bgr, (int(x0), int(top_y)), (int(x1), int(surface_y)), color, 2)
        cv2.line(overlay_bgr, (center_x, int(surface_y)), (center_x, int(top_y)), color, 3, cv2.LINE_AA)
        cv2.circle(overlay_bgr, (center_x, int(top_y)), 6, color, -1)
        cv2.circle(overlay_bgr, (center_x, int(surface_y)), 6, color, -1)
        height_label = (
            f"{tray_analyzer._length_px_to_cm(visible_height_px, pixels_per_cm):.2f} cm"
            if pixels_per_cm is not None
            else f"{int(round(visible_height_px))} px"
        )
        caption = f"{region['name']}: {height_label}"
        caption_x = max(12, int(x0))
        caption_y = max(26, int(top_y) - 12)
        (caption_w, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(overlay_bgr, (caption_x - 8, caption_y - 18), (caption_x + caption_w + 10, caption_y + 8), (18, 18, 18), -1)
        cv2.putText(overlay_bgr, caption, (caption_x, caption_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

        summary_rows.append(
            {
                "Plant": str(region["name"]),
                "Position": str(region["position"]),
                "Seedling Index": int(region["site_index"]),
                "Grid Row": 1,
                "Grid Column": int(region["col_index"] + 1),
                "View": "Side view height",
                "Scale Source": scale_source,
                "Pixels Per Cm": pixels_per_cm,
                "Pixels Per Cm Override": pixels_per_cm,
                "Mm Per Pixel": mm_per_pixel,
                "Surface Y": int(surface_y),
                "Top Y": int(top_y),
                "Center X": int(center_x),
                "Visible Shoot Area (px)": int(round(visible_area_px)),
                "Visible Shoot Area (cm^2)": tray_analyzer._area_px_to_cm2(visible_area_px, pixels_per_cm),
                "Visible Shoot Width (px)": int(round(visible_width_px)),
                "Visible Shoot Width (cm)": tray_analyzer._length_px_to_cm(visible_width_px, pixels_per_cm),
                "Visible Shoot Height (px)": tray_analyzer._round_or_none(visible_height_px, 2),
                "Visible Shoot Height (cm)": tray_analyzer._length_px_to_cm(visible_height_px, pixels_per_cm),
                "Mean R": color_stats["mean_r"],
                "Mean G": color_stats["mean_g"],
                "Mean B": color_stats["mean_b"],
                "Mean H": color_stats["mean_h"],
                "Mean S": color_stats["mean_s"],
                "Mean V": color_stats["mean_v"],
                "ExG Mean": color_stats["exg_mean"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    counts_columns = ["Plant", "Position", "Visible Shoot Height (cm)"]
    counts_df = summary_df[[column for column in counts_columns if column in summary_df.columns]].copy()

    return {
        "image_shape": tuple(int(v) for v in image_rgb.shape[:2]),
        "overlay_rgb": cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
        "summary_df": summary_df,
        "counts_df": counts_df,
        "shoot_mask_u8": shoot_mask_u8,
        "surface_y": int(surface_y),
        "pixels_per_cm": pixels_per_cm,
        "pixels_per_cm_override": pixels_per_cm,
        "mm_per_pixel": mm_per_pixel,
        "scale_source": scale_source,
        "center_ratios": tuple(float(value) for value in center_ratios.tolist()),
        "detected_seedling_count": int(len(summary_rows)),
    }


def _build_linked_seedling_batch_payload(
    top_items: list[dict[str, Any]],
    side_items: list[dict[str, Any]],
    flat_items: list[dict[str, Any]],
    expected_seedling_count: int,
    top_pixels_per_cm_override: float | None,
    side_pixels_per_cm_override: float | None,
    flat_pixels_per_cm_override: float | None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    experiment_rows: list[dict[str, Any]] = []
    seedling_frames = []
    top_leaf_frames = []
    flat_leaf_frames = []
    skipped_items: list[dict[str, str]] = []

    experiment_count = min(len(top_items), len(side_items), len(flat_items))
    for experiment_idx in range(experiment_count):
        top_item = top_items[experiment_idx]
        side_item = side_items[experiment_idx]
        flat_item = flat_items[experiment_idx]
        experiment_name = f"Experiment {experiment_idx + 1}"

        try:
            top_result = _analyze_linked_seedling_top_view(
                _get_item_bytes(top_item),
                expected_seedling_count=expected_seedling_count,
                pixels_per_cm_override=top_pixels_per_cm_override,
            )
            side_result = _analyze_linked_seedling_side_view(
                _get_item_bytes(side_item),
                expected_seedling_count=expected_seedling_count,
                pixels_per_cm_override=side_pixels_per_cm_override,
                initial_center_ratios=tuple(float(value) for value in top_result["center_ratios"]),
            )
            flat_result = _analyze_upload(
                _get_item_bytes(flat_item),
                SEEDLINGS_PROFILE_KEY,
                DEFAULT_TRAY_LONG_SIDE_CM,
                _coerce_optional_pixels_per_cm(flat_pixels_per_cm_override),
                container_mode=FULL_IMAGE_CONTAINER_MODE,
            )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            skipped_items.append(
                {
                    "Experiment": experiment_name,
                    "Reason": f"Skipped invalid image data ({type(exc).__name__}).",
                    "Top Image": _item_display_path(top_item),
                    "Side Image": _item_display_path(side_item),
                    "Flat Image": _item_display_path(flat_item),
                }
            )
            continue
        except Exception as exc:
            skipped_items.append(
                {
                    "Experiment": experiment_name,
                    "Reason": f"Processing failed ({type(exc).__name__}).",
                    "Top Image": _item_display_path(top_item),
                    "Side Image": _item_display_path(side_item),
                    "Flat Image": _item_display_path(flat_item),
                }
            )
            continue

        base_seedlings = pd.DataFrame(
            {
                "Seedling Index": list(range(1, int(expected_seedling_count) + 1)),
                "Seedling": [f"Seedling {idx}" for idx in range(1, int(expected_seedling_count) + 1)],
                "Expected Position": [
                    tray_analyzer._format_grid_position(0, idx - 1, 1, int(expected_seedling_count))
                    for idx in range(1, int(expected_seedling_count) + 1)
                ],
            }
        )

        top_merge_df = pd.DataFrame()
        if not top_result["plant_summary_df"].empty:
            top_merge_df = (
                top_result["plant_summary_df"][
                    [
                        column
                        for column in [
                            "Site Index",
                            "Estimated Leaves",
                            "Canopy Area (px)",
                            "Canopy Area (cm^2)",
                            "Canopy BBox Width (px)",
                            "Canopy BBox Width (cm)",
                            "Canopy BBox Height (px)",
                            "Canopy BBox Height (cm)",
                            "Canopy Angle (deg)",
                            "Mean R",
                            "Mean G",
                            "Mean B",
                            "Mean H",
                            "Mean S",
                            "Mean V",
                            "ExG Mean",
                        ]
                        if column in top_result["plant_summary_df"].columns
                    ]
                ]
                .rename(
                    columns={
                        "Site Index": "Seedling Index",
                        "Estimated Leaves": "Top Estimated Leaves",
                        "Canopy Area (px)": "Top Shoot Area (px)",
                        "Canopy Area (cm^2)": "Top Shoot Area (cm^2)",
                        "Canopy BBox Width (px)": "Top Shoot Width (px)",
                        "Canopy BBox Width (cm)": "Top Shoot Width (cm)",
                        "Canopy BBox Height (px)": "Top Shoot Height Span (px)",
                        "Canopy BBox Height (cm)": "Top Shoot Height Span (cm)",
                        "Canopy Angle (deg)": "Top Shoot Angle (deg)",
                        "Mean R": "Top Mean R",
                        "Mean G": "Top Mean G",
                        "Mean B": "Top Mean B",
                        "Mean H": "Top Mean H",
                        "Mean S": "Top Mean S",
                        "Mean V": "Top Mean V",
                        "ExG Mean": "Top ExG Mean",
                    }
                )
            )

        side_merge_df = pd.DataFrame()
        if not side_result["summary_df"].empty:
            side_merge_df = side_result["summary_df"][
                [
                    column
                    for column in [
                        "Seedling Index",
                        "Visible Shoot Area (px)",
                        "Visible Shoot Area (cm^2)",
                        "Visible Shoot Width (px)",
                        "Visible Shoot Width (cm)",
                        "Visible Shoot Height (px)",
                        "Visible Shoot Height (cm)",
                        "Surface Y",
                        "Top Y",
                    ]
                    if column in side_result["summary_df"].columns
                ]
            ].copy()

        flat_merge_df = pd.DataFrame()
        if not flat_result.plant_summary_df.empty:
            flat_merge_df = (
                flat_result.plant_summary_df[
                    [
                        column
                        for column in [
                            "Site Index",
                            "Estimated Leaves",
                            "Shoot Area (px)",
                            "Shoot Area (cm^2)",
                            "Shoot Perimeter (px)",
                            "Shoot Perimeter (cm)",
                            "Total Root Length (px)",
                            "Total Root Length (cm)",
                            "Primary Root Length (px)",
                            "Primary Root Length (cm)",
                            "Lateral Root Length (px)",
                            "Lateral Root Length (cm)",
                            "Lateral Root Count",
                            "Root Area (px)",
                            "Root Area (cm^2)",
                            "Primary Root Area (px)",
                            "Primary Root Area (cm^2)",
                            "Lateral Root Area (px)",
                            "Lateral Root Area (cm^2)",
                            "Root Endpoint Count",
                            "Root:Shoot Area Ratio",
                            "Primary:Total Root Length Ratio",
                        ]
                        if column in flat_result.plant_summary_df.columns
                    ]
                ]
                .rename(
                    columns={
                        "Site Index": "Seedling Index",
                        "Estimated Leaves": "Flat Estimated Leaves",
                        "Shoot Area (px)": "Flat Shoot Area (px)",
                        "Shoot Area (cm^2)": "Flat Shoot Area (cm^2)",
                        "Shoot Perimeter (px)": "Flat Shoot Perimeter (px)",
                        "Shoot Perimeter (cm)": "Flat Shoot Perimeter (cm)",
                    }
                )
            )

        combined_df = base_seedlings.copy()
        if not top_merge_df.empty:
            combined_df = combined_df.merge(top_merge_df, on="Seedling Index", how="left")
        if not side_merge_df.empty:
            combined_df = combined_df.merge(side_merge_df, on="Seedling Index", how="left")
        if not flat_merge_df.empty:
            combined_df = combined_df.merge(flat_merge_df, on="Seedling Index", how="left")
        combined_df.insert(0, "Experiment", experiment_name)
        combined_df["Top Source"] = _item_display_path(top_item)
        combined_df["Side Source"] = _item_display_path(side_item)
        combined_df["Flat Source"] = _item_display_path(flat_item)

        top_leaf_df = top_result["leaf_detail_df"].copy()
        if not top_leaf_df.empty:
            top_leaf_df.insert(0, "Experiment", experiment_name)
            top_leaf_df.insert(1, "Source Path", _item_display_path(top_item))
            top_leaf_frames.append(top_leaf_df)

        flat_leaf_df = flat_result.leaf_detail_df.copy()
        if not flat_leaf_df.empty:
            flat_leaf_df.insert(0, "Experiment", experiment_name)
            flat_leaf_df.insert(1, "Source Path", _item_display_path(flat_item))
            flat_leaf_frames.append(flat_leaf_df)

        seedling_frames.append(combined_df)
        experiment_rows.append(
            {
                "Experiment": experiment_name,
                "Top Image": str(top_item["name"]),
                "Side Image": str(side_item["name"]),
                "Flat Image": str(flat_item["name"]),
                "Top Source": _item_display_path(top_item),
                "Side Source": _item_display_path(side_item),
                "Flat Source": _item_display_path(flat_item),
                "Seedlings Requested": int(expected_seedling_count),
                "Top Seedlings Detected": int(top_result["detected_seedling_count"]),
                "Side Seedlings Detected": int(side_result["detected_seedling_count"]),
                "Flat Seedlings Detected": int(len(flat_result.plant_summary_df)),
                "Top Pixels Per Cm": top_result["pixels_per_cm"],
                "Side Pixels Per Cm": side_result["pixels_per_cm"],
                "Flat Pixels Per Cm": flat_result.pixels_per_cm,
                "Top Shoot Area Sum (cm^2)": tray_analyzer._round_or_none(_numeric_series_sum(combined_df, "Top Shoot Area (cm^2)"), 4),
                "Mean Visible Shoot Height (cm)": tray_analyzer._round_or_none(_numeric_series_mean(combined_df, "Visible Shoot Height (cm)"), 4),
                "Total Root Length (cm)": tray_analyzer._round_or_none(_numeric_series_sum(combined_df, "Total Root Length (cm)"), 4),
                "Total Lateral Root Count": int(round(_numeric_series_sum(combined_df, "Lateral Root Count") or 0.0)),
            }
        )
        records.append(
            {
                "experiment_name": experiment_name,
                "top_item": top_item,
                "side_item": side_item,
                "flat_item": flat_item,
                "top_result": top_result,
                "side_result": side_result,
                "flat_result": flat_result,
                "combined_df": combined_df,
            }
        )

    experiment_summary_df = pd.DataFrame(experiment_rows)
    seedling_summary_df = pd.concat(seedling_frames, ignore_index=True) if seedling_frames else pd.DataFrame()
    top_leaf_detail_df = pd.concat(top_leaf_frames, ignore_index=True) if top_leaf_frames else pd.DataFrame()
    flat_leaf_detail_df = pd.concat(flat_leaf_frames, ignore_index=True) if flat_leaf_frames else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped_items)

    return {
        "records": records,
        "experiment_summary_df": experiment_summary_df,
        "seedling_summary_df": seedling_summary_df,
        "top_leaf_detail_df": top_leaf_detail_df,
        "flat_leaf_detail_df": flat_leaf_detail_df,
        "skipped_df": skipped_df,
    }


def _linked_seedling_results_zip_signature(batch_payload: dict[str, Any]) -> str:
    digest = hashlib.sha1()
    for key in ("experiment_summary_df", "seedling_summary_df", "top_leaf_detail_df", "flat_leaf_detail_df", "skipped_df"):
        df = batch_payload.get(key)
        digest.update(key.encode("utf-8"))
        digest.update(_csv_bytes(df) if isinstance(df, pd.DataFrame) else b"")
    for record in batch_payload.get("records", []):
        digest.update(str(record.get("experiment_name", "")).encode("utf-8"))
        digest.update(str(record["top_item"].get("source_path", "")).encode("utf-8"))
        digest.update(str(record["side_item"].get("source_path", "")).encode("utf-8"))
        digest.update(str(record["flat_item"].get("source_path", "")).encode("utf-8"))
    return digest.hexdigest()


def _build_linked_seedling_results_bundle_bytes(batch_payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("experiment_summary.csv", _csv_bytes(batch_payload["experiment_summary_df"]))
        zf.writestr("seedling_multiview_summary.csv", _csv_bytes(batch_payload["seedling_summary_df"]))
        zf.writestr("top_view_leaf_details.csv", _csv_bytes(batch_payload["top_leaf_detail_df"]))
        zf.writestr("flat_view_leaf_details.csv", _csv_bytes(batch_payload["flat_leaf_detail_df"]))
        if not batch_payload["skipped_df"].empty:
            zf.writestr("skipped_items.csv", _csv_bytes(batch_payload["skipped_df"]))

        for record in batch_payload["records"]:
            experiment_slug = str(record["experiment_name"]).lower().replace(" ", "_")
            top_item = record["top_item"]
            side_item = record["side_item"]
            flat_item = record["flat_item"]

            zf.writestr(
                f"originals/{experiment_slug}/top_{_safe_zip_path(str(top_item['name']))}",
                _get_item_bytes(top_item),
            )
            zf.writestr(
                f"originals/{experiment_slug}/side_{_safe_zip_path(str(side_item['name']))}",
                _get_item_bytes(side_item),
            )
            zf.writestr(
                f"originals/{experiment_slug}/flat_{_safe_zip_path(str(flat_item['name']))}",
                _get_item_bytes(flat_item),
            )

            zf.writestr(
                f"overlays/{experiment_slug}_top_overlay.png",
                _png_bytes_from_array(record["top_result"]["overlay_rgb"]),
            )
            zf.writestr(
                f"overlays/{experiment_slug}_side_overlay.png",
                _png_bytes_from_array(record["side_result"]["overlay_rgb"]),
            )
            zf.writestr(
                f"overlays/{experiment_slug}_flat_overlay.png",
                _png_bytes_from_array(record["flat_result"].overlay_rgb),
            )

            zf.writestr(
                f"masks/{experiment_slug}_top_shoot_mask.png",
                _png_bytes_from_array(record["top_result"]["shoot_mask_u8"]),
            )
            zf.writestr(
                f"masks/{experiment_slug}_side_shoot_mask.png",
                _png_bytes_from_array(record["side_result"]["shoot_mask_u8"]),
            )
            flat_mask_assets = _build_mask_assets(record["flat_result"])
            zf.writestr(
                f"masks/{experiment_slug}_flat_shoot_mask.png",
                _png_bytes_from_array(flat_mask_assets.get("shoot_mask", flat_mask_assets["canopy_mask"])),
            )
            zf.writestr(
                f"masks/{experiment_slug}_flat_root_mask.png",
                _png_bytes_from_array(flat_mask_assets.get("root_mask", np.zeros(record["flat_result"].image_shape, dtype=np.uint8))),
            )
            zf.writestr(
                f"masks/{experiment_slug}_flat_primary_root_mask.png",
                _png_bytes_from_array(flat_mask_assets.get("primary_root_mask", np.zeros(record["flat_result"].image_shape, dtype=np.uint8))),
            )
            zf.writestr(
                f"masks/{experiment_slug}_flat_lateral_root_mask.png",
                _png_bytes_from_array(flat_mask_assets.get("lateral_root_mask", np.zeros(record["flat_result"].image_shape, dtype=np.uint8))),
            )

            zf.writestr(
                f"per_experiment_csv/{experiment_slug}_seedling_summary.csv",
                _csv_bytes(record["combined_df"]),
            )

    return buffer.getvalue()


def _render_record_detail(record: dict[str, Any]) -> None:
    result = record["result"]
    is_seedling_analysis = getattr(result, "analysis_kind", "") == ANALYSIS_KIND_SEEDLINGS
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
        if is_seedling_analysis and "Total Root Length (cm)" in result.plant_summary_df.columns:
            st.metric(
                "Total root length (cm)",
                round(float(result.plant_summary_df["Total Root Length (cm)"].fillna(0).sum()), 2),
            )
        else:
            st.metric("Total canopy area (cm^2)", round(float(result.plant_summary_df["Canopy Area (cm^2)"].fillna(0).sum()), 2))
        if is_seedling_analysis and "Lateral Root Count" in result.plant_summary_df.columns:
            st.metric("Total lateral roots", int(result.plant_summary_df["Lateral Root Count"].fillna(0).sum()))
        else:
            st.metric("Pixels / cm", result.pixels_per_cm if result.pixels_per_cm is not None else "n/a")
        st.caption(
            f"Layout: {result.tray_profile_name} | Container: {result.container_source} | Scale: {result.scale_source} | Segmentation: {result.segmentation_source} | "
            f"Tray long side: {result.tray_long_side_cm:.1f} cm = {result.tray_long_side_px:.1f} px"
        )
        if "circular pot" in str(result.container_source).lower():
            st.caption(
                f"Circle adjustment: x {round(float(result.circle_center_x_shift_ratio) * 100.0, 1)}% | "
                f"y {round(float(result.circle_center_y_shift_ratio) * 100.0, 1)}% | "
                f"size {round(float(result.circle_radius_scale) * 100.0, 1)}%"
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

    st.subheader("Seedling Summary Traits" if is_seedling_analysis else "Plant Summary Traits")
    st.dataframe(result.plant_summary_df, hide_index=True, use_container_width=True)

    st.subheader("Per-Shoot-Leaf Traits" if is_seedling_analysis else "Per-Leaf Traits")
    st.dataframe(result.leaf_detail_df, hide_index=True, use_container_width=True, height=420)


def _render_linked_seedling_experiment_detail(record: dict[str, Any]) -> None:
    top_original = _preview_from_bytes(_get_item_bytes(record["top_item"]), MAX_PREVIEW_EDGE)
    side_original = _preview_from_bytes(_get_item_bytes(record["side_item"]), MAX_PREVIEW_EDGE)
    flat_original = _preview_from_bytes(_get_item_bytes(record["flat_item"]), MAX_PREVIEW_EDGE)
    top_overlay = _resize_for_preview(record["top_result"]["overlay_rgb"], MAX_PREVIEW_EDGE)
    side_overlay = _resize_for_preview(record["side_result"]["overlay_rgb"], MAX_PREVIEW_EDGE)
    flat_overlay = _resize_for_preview(record["flat_result"].overlay_rgb, MAX_PREVIEW_EDGE)

    overlay_tab, original_tab = st.tabs(["Overlays", "Originals"])
    with overlay_tab:
        overlay_cols = st.columns(3)
        overlay_cols[0].image(top_overlay, caption=f"Top view overlay: {record['top_item']['name']}", use_container_width=True)
        overlay_cols[1].image(side_overlay, caption=f"Side view overlay: {record['side_item']['name']}", use_container_width=True)
        overlay_cols[2].image(flat_overlay, caption=f"Flat roots+shoots overlay: {record['flat_item']['name']}", use_container_width=True)
    with original_tab:
        original_cols = st.columns(3)
        original_cols[0].image(top_original, caption=f"Top view original: {record['top_item']['name']}", use_container_width=True)
        original_cols[1].image(side_original, caption=f"Side view original: {record['side_item']['name']}", use_container_width=True)
        original_cols[2].image(flat_original, caption=f"Flat roots+shoots original: {record['flat_item']['name']}", use_container_width=True)

    summary_cols = st.columns(4)
    combined_df = record["combined_df"]
    summary_cols[0].metric("Seedlings linked", int(combined_df["Seedling Index"].nunique()))
    top_area_sum_cm2 = _numeric_series_sum(combined_df, "Top Shoot Area (cm^2)")
    mean_visible_height_cm = _numeric_series_mean(combined_df, "Visible Shoot Height (cm)")
    total_root_length_cm = _numeric_series_sum(combined_df, "Total Root Length (cm)")
    summary_cols[1].metric("Top shoot area (cm^2)", "n/a" if top_area_sum_cm2 is None else round(top_area_sum_cm2, 2))
    summary_cols[2].metric("Mean visible height (cm)", "n/a" if mean_visible_height_cm is None else round(mean_visible_height_cm, 2))
    summary_cols[3].metric("Total root length (cm)", "n/a" if total_root_length_cm is None else round(total_root_length_cm, 2))

    st.subheader("Linked Seedling Summary")
    st.dataframe(combined_df, hide_index=True, use_container_width=True)

    with st.expander("Top-View Leaf Detail Table", expanded=False):
        st.dataframe(record["top_result"]["leaf_detail_df"], hide_index=True, use_container_width=True, height=360)

    with st.expander("Flat Root + Shoot Leaf Detail Table", expanded=False):
        st.dataframe(record["flat_result"].leaf_detail_df, hide_index=True, use_container_width=True, height=360)


def _render_linked_seedling_workflow() -> None:
    st.subheader("Linked Seedling Experiment")
    st.caption(
        "Upload the `top view shoots`, `side view shoot height`, and `flat roots + shoots` images separately. "
        "Experiments are paired by upload order and seedlings are linked left-to-right across the three modalities."
    )

    settings_cols = st.columns([0.9, 0.7, 0.7, 0.7])
    expected_seedling_count = int(
        settings_cols[0].number_input(
            "Seedlings per experiment",
            min_value=1,
            max_value=64,
            value=4,
            step=1,
        )
    )
    top_pixels_per_cm = float(
        settings_cols[1].number_input(
            "Top view pixels / cm",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Optional manual calibration for the top-view images. Use `0` to leave top-view physical units uncalibrated.",
        )
    )
    side_pixels_per_cm = float(
        settings_cols[2].number_input(
            "Side view pixels / cm",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Optional manual calibration for visible shoot height in the side-view images. Use `0` to keep heights in pixels only.",
        )
    )
    flat_pixels_per_cm = float(
        settings_cols[3].number_input(
            "Flat roots pixels / cm",
            min_value=0.0,
            value=100.0,
            step=0.1,
            help="Manual calibration for the flat roots+shoots images. This is usually the most important scale for root traits.",
        )
    )

    top_uploads_col, side_uploads_col, flat_uploads_col = st.columns(3)
    with top_uploads_col:
        top_files = st.file_uploader(
            "Top-view shoot images",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="linked_seedling_top_files",
        )
        top_archives = st.file_uploader(
            "Top-view zip archives",
            type=["zip"],
            accept_multiple_files=True,
            key="linked_seedling_top_archives",
        )
    with side_uploads_col:
        side_files = st.file_uploader(
            "Side-view height images",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="linked_seedling_side_files",
        )
        side_archives = st.file_uploader(
            "Side-view zip archives",
            type=["zip"],
            accept_multiple_files=True,
            key="linked_seedling_side_archives",
        )
    with flat_uploads_col:
        flat_files = st.file_uploader(
            "Flat roots + shoots images",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="linked_seedling_flat_files",
        )
        flat_archives = st.file_uploader(
            "Flat roots + shoots zip archives",
            type=["zip"],
            accept_multiple_files=True,
            key="linked_seedling_flat_archives",
        )

    top_items, top_zip_count, top_zip_images = _collect_uploaded_items(top_files, top_archives, "top-view shoot images")
    side_items, side_zip_count, side_zip_images = _collect_uploaded_items(side_files, side_archives, "side-view height images")
    flat_items, flat_zip_count, flat_zip_images = _collect_uploaded_items(flat_files, flat_archives, "flat roots + shoots images")

    if top_zip_count or side_zip_count or flat_zip_count:
        st.caption(
            f"Zip inputs discovered: top {top_zip_images} image(s), side {side_zip_images} image(s), flat {flat_zip_images} image(s)."
        )

    if not top_items or not side_items or not flat_items:
        st.info(
            "Upload at least one image for each modality: top-view shoots, side-view height, and flat roots + shoots. "
            "You can upload files directly or zip archives for any lane."
        )
        return

    experiment_count = min(len(top_items), len(side_items), len(flat_items))
    if len(top_items) != len(side_items) or len(top_items) != len(flat_items):
        st.warning(
            "The three modality lanes do not contain the same number of images. "
            f"The workflow will analyze the first {experiment_count} paired experiment(s) by upload order."
        )

    with st.spinner(f"Analyzing {experiment_count} linked seedling experiment(s)..."):
        batch_payload = _build_linked_seedling_batch_payload(
            top_items=top_items[:experiment_count],
            side_items=side_items[:experiment_count],
            flat_items=flat_items[:experiment_count],
            expected_seedling_count=expected_seedling_count,
            top_pixels_per_cm_override=_coerce_optional_pixels_per_cm(top_pixels_per_cm),
            side_pixels_per_cm_override=_coerce_optional_pixels_per_cm(side_pixels_per_cm),
            flat_pixels_per_cm_override=_coerce_optional_pixels_per_cm(flat_pixels_per_cm),
        )

    results_zip_signature = _linked_seedling_results_zip_signature(batch_payload)
    if st.session_state.get("prepared_linked_seedling_zip_signature") != results_zip_signature:
        st.session_state.pop("prepared_linked_seedling_zip_bytes", None)
        st.session_state["prepared_linked_seedling_zip_signature"] = results_zip_signature

    skipped_df = batch_payload["skipped_df"]
    if not skipped_df.empty:
        st.warning(f"Skipped {len(skipped_df)} linked experiment(s) that could not be decoded or processed.")
        with st.expander("Skipped linked experiments", expanded=False):
            st.dataframe(skipped_df, hide_index=True, use_container_width=True)

    if not batch_payload["records"]:
        st.error("No linked seedling experiments could be analyzed from the uploaded images.")
        return

    experiment_summary_df = batch_payload["experiment_summary_df"]
    seedling_summary_df = batch_payload["seedling_summary_df"]
    top_leaf_detail_df = batch_payload["top_leaf_detail_df"]
    flat_leaf_detail_df = batch_payload["flat_leaf_detail_df"]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Experiments", int(len(batch_payload["records"])))
    summary_cols[1].metric("Seedlings", int(seedling_summary_df["Seedling Index"].count()))
    visible_height_sum_cm = _numeric_series_sum(seedling_summary_df, "Visible Shoot Height (cm)")
    root_length_sum_cm = _numeric_series_sum(seedling_summary_df, "Total Root Length (cm)")
    summary_cols[2].metric("Visible height sum (cm)", "n/a" if visible_height_sum_cm is None else round(visible_height_sum_cm, 2))
    summary_cols[3].metric("Root length sum (cm)", "n/a" if root_length_sum_cm is None else round(root_length_sum_cm, 2))

    export_cols = st.columns(5)
    export_cols[0].download_button(
        "Download Experiment CSV",
        data=_csv_bytes(experiment_summary_df),
        file_name="experiment_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "Download Seedling CSV",
        data=_csv_bytes(seedling_summary_df),
        file_name="seedling_multiview_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[2].download_button(
        "Download Top Leaf CSV",
        data=_csv_bytes(top_leaf_detail_df),
        file_name="top_view_leaf_details.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[3].download_button(
        "Download Flat Leaf CSV",
        data=_csv_bytes(flat_leaf_detail_df),
        file_name="flat_view_leaf_details.csv",
        mime="text/csv",
        use_container_width=True,
    )
    prepared_zip_bytes = st.session_state.get("prepared_linked_seedling_zip_bytes")
    if prepared_zip_bytes is None:
        if export_cols[4].button("Prepare Linked ZIP", use_container_width=True):
            with st.spinner("Preparing linked experiment ZIP..."):
                st.session_state["prepared_linked_seedling_zip_bytes"] = _build_linked_seedling_results_bundle_bytes(batch_payload)
            st.rerun()
    else:
        export_cols[4].download_button(
            "Download Linked ZIP",
            data=prepared_zip_bytes,
            file_name="seedling_multiview_results.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.caption(
        "The linked results zip contains experiment-level CSVs, original images, overlays for the three modalities, "
        "top-view masks, and flat root/shoot masks."
    )

    st.subheader("Experiment Summary")
    st.dataframe(experiment_summary_df, hide_index=True, use_container_width=True)

    experiment_labels = [record["experiment_name"] for record in batch_payload["records"]]
    selected_experiment = st.selectbox("Experiment detail view", options=experiment_labels, index=0)
    selected_record = next(record for record in batch_payload["records"] if record["experiment_name"] == selected_experiment)
    _render_linked_seedling_experiment_detail(selected_record)

    with st.expander("Combined Seedling Table", expanded=(len(batch_payload["records"]) == 1)):
        st.dataframe(seedling_summary_df, hide_index=True, use_container_width=True, height=420)

    with st.expander("Top-View Leaf Detail Table", expanded=False):
        st.dataframe(top_leaf_detail_df, hide_index=True, use_container_width=True, height=360)

    with st.expander("Flat Root + Shoot Leaf Detail Table", expanded=False):
        st.dataframe(flat_leaf_detail_df, hide_index=True, use_container_width=True, height=360)


def main() -> None:
    logo_data_uri = _image_data_uri(str(LOGO_PATH))
    title_col, logo_col = st.columns([0.92, 0.08])
    with title_col:
        st.title("Plant Tray Phenotyping Dashboard")
        st.caption(
            "Run tray, chamber, or seedling images, extract shoot and root traits, and export combined CSVs. "
            "Physical measurements are normalized from the detected tray long side or your manual pixels/cm calibration."
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

    settings_col_1, settings_col_2, settings_col_3 = st.columns([1.15, 0.65, 0.7])
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
    container_mode = settings_col_3.selectbox(
        "Container geometry",
        options=[key for key, _ in CONTAINER_MODE_OPTIONS],
        index=next(
            (idx for idx, (key, _) in enumerate(CONTAINER_MODE_OPTIONS) if key == DEFAULT_CONTAINER_MODE),
            0,
        ),
        format_func=lambda key: CONTAINER_MODE_LABELS.get(key, key),
    )
    if tray_profile_key == DEFAULT_TRAY_PROFILE_KEY:
        st.caption("Auto layout chooses between a 4-plant 2x2 tray and a 20-plant 4x5 Arabidopsis tray from the canopy pattern.")
    if tray_profile_key == SEEDLINGS_PROFILE_KEY:
        st.caption(
            "Seedlings mode separates green shoots from roots and reports primary-root and lateral-root traits per seedling. "
            "Use `Full image` for flat scans on dark backgrounds and `Pixels Per Cm Override` when there is no tray scale in the image."
        )
    if tray_profile_key == CUSTOM_TRAY_PROFILE_KEY:
        custom_col_1, custom_col_2, custom_col_3, custom_col_4 = st.columns(4)
        custom_grid_rows = int(
            custom_col_1.number_input(
                "Custom grid rows",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
            )
        )
        custom_grid_cols = int(
            custom_col_2.number_input(
                "Custom grid columns",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
            )
        )
        custom_outer_pad_pct = float(
            custom_col_3.slider(
                "Outer margin (%)",
                min_value=0,
                max_value=25,
                value=5,
                step=1,
            )
        )
        custom_site_pad_pct = float(
            custom_col_4.slider(
                "Pot / site padding (%)",
                min_value=0,
                max_value=30,
                value=6,
                step=1,
            )
        )
        st.caption(
            "Universal mode works for arbitrary grids such as `1x1`, `1x2`, `2x1`, `3x4`, or `5x5`. "
            "Use `Circular pot / chamber` when the growing area is round and you want wall reflections ignored, "
            "and use `Full image` for open top-down tray or chamber photos where the whole field should be analyzed."
        )
    else:
        custom_grid_rows = None
        custom_grid_cols = None
        custom_outer_pad_pct = None
        custom_site_pad_pct = None

    if tray_profile_key == SEEDLINGS_PROFILE_KEY:
        seedling_workflow_mode = st.selectbox(
            "Seedling workflow",
            options=[SEEDLING_WORKFLOW_SINGLE, SEEDLING_WORKFLOW_LINKED],
            index=0,
            format_func=lambda value: {
                SEEDLING_WORKFLOW_SINGLE: "Single image or batch (current seedlings mode)",
                SEEDLING_WORKFLOW_LINKED: "Linked experiment: top view + side view + flat roots",
            }.get(value, value),
        )
        if seedling_workflow_mode == SEEDLING_WORKFLOW_LINKED:
            _render_linked_seedling_workflow()
            st.markdown("---")
            st.markdown(
                f"<div style='text-align:center; font-size:0.9rem; color:rgba(250,250,250,0.65);'>{FOOTER_TEXT}</div>",
                unsafe_allow_html=True,
            )
            return

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
        st.info("Upload one or more tray or seedling images, or a zip archive of images, to analyze them together.")
        return

    st.subheader("Per-Image Scale")
    st.caption(
        "`33 cm` is the default tray long side. You can override it per image below. "
        "If `Pixels Per Cm Override` is filled, it takes priority for that image and is useful when no blue tray is visible."
    )
    scale_editor_df = _build_scale_editor_df(file_items, float(tray_long_side_cm))
    scale_apply_col, scale_clear_col, scale_note_col = st.columns([0.22, 0.22, 0.56])
    with scale_apply_col:
        if st.button(f"Apply {float(tray_long_side_cm):.2f} cm to all images", use_container_width=True):
            st.session_state["per_image_tray_long_side_cm"] = {
                str(row["_Item Key"]): float(tray_long_side_cm)
                for _, row in scale_editor_df.iterrows()
            }
            st.rerun()
    with scale_clear_col:
        if st.button("Clear pixels/cm overrides", use_container_width=True):
            st.session_state["per_image_pixels_per_cm_override"] = {}
            st.rerun()
    with scale_note_col:
        st.caption("Use the global tray value above as your starting point, then edit only the images that need a different tray size or a direct pixels/cm calibration.")

    edited_scale_df = st.data_editor(
        scale_editor_df,
        hide_index=True,
        use_container_width=True,
        key="per_image_scale_editor",
        disabled=["_Item Key", "Image", "Source Path"],
        column_config={
            "_Item Key": None,
            "Tray Long Side (cm)": st.column_config.NumberColumn(
                "Tray Long Side (cm)",
                min_value=1.0,
                step=0.1,
                format="%.2f",
                help="Override the physical tray long side used to convert pixels into cm for this image.",
            ),
            "Pixels Per Cm Override": st.column_config.NumberColumn(
                "Pixels Per Cm Override",
                min_value=0.0001,
                step=0.1,
                format="%.4f",
                help="Optional direct calibration. When provided, this overrides tray-based scale detection for this image.",
            ),
        },
    )
    st.session_state["per_image_tray_long_side_cm"] = {
        str(row["_Item Key"]): _coerce_tray_long_side_cm(row["Tray Long Side (cm)"], float(tray_long_side_cm))
        for _, row in edited_scale_df.iterrows()
    }
    st.session_state["per_image_pixels_per_cm_override"] = {
        str(row["_Item Key"]): _coerce_optional_pixels_per_cm(row.get("Pixels Per Cm Override"))
        for _, row in edited_scale_df.iterrows()
        if _coerce_optional_pixels_per_cm(row.get("Pixels Per Cm Override")) is not None
    }
    tray_long_side_values_cm = tuple(
        _coerce_tray_long_side_cm(row["Tray Long Side (cm)"], float(tray_long_side_cm))
        for _, row in edited_scale_df.iterrows()
    )
    pixels_per_cm_override_values = tuple(
        _coerce_optional_pixels_per_cm(row.get("Pixels Per Cm Override"))
        for _, row in edited_scale_df.iterrows()
    )

    st.subheader("Per-Image Circular Mask Adjustment")
    st.caption(
        "These controls let you move or enlarge the circular container mask per image. "
        "They matter for `Circular pot / chamber`, and are ignored for `Rectangle tray` or `Full image`."
    )
    circle_editor_df = _build_circle_adjustment_editor_df(file_items)
    circle_reset_col, circle_note_col = st.columns([0.22, 0.78])
    with circle_reset_col:
        if st.button("Reset circle adjustments", use_container_width=True):
            st.session_state["per_image_circle_center_x_shift_pct"] = {}
            st.session_state["per_image_circle_center_y_shift_pct"] = {}
            st.session_state["per_image_circle_radius_scale_pct"] = {}
            st.rerun()
    with circle_note_col:
        st.caption("Positive X moves right, positive Y moves down, and size above `100%` makes the circle larger.")

    edited_circle_df = st.data_editor(
        circle_editor_df,
        hide_index=True,
        use_container_width=True,
        key="per_image_circle_editor",
        disabled=["_Item Key", "Image", "Source Path"],
        column_config={
            "_Item Key": None,
            "Circle X Shift (%)": st.column_config.NumberColumn(
                "Circle X Shift (%)",
                min_value=-75.0,
                max_value=75.0,
                step=1.0,
                format="%.1f",
                help="Shift the detected circle horizontally as a percent of image width.",
            ),
            "Circle Y Shift (%)": st.column_config.NumberColumn(
                "Circle Y Shift (%)",
                min_value=-75.0,
                max_value=75.0,
                step=1.0,
                format="%.1f",
                help="Shift the detected circle vertically as a percent of image height.",
            ),
            "Circle Size (%)": st.column_config.NumberColumn(
                "Circle Size (%)",
                min_value=20.0,
                max_value=300.0,
                step=1.0,
                format="%.1f",
                help="Scale the detected circle radius. `100%` keeps the original size.",
            ),
        },
    )
    st.session_state["per_image_circle_center_x_shift_pct"] = {
        str(row["_Item Key"]): _coerce_circle_shift_pct(row.get("Circle X Shift (%)"))
        for _, row in edited_circle_df.iterrows()
    }
    st.session_state["per_image_circle_center_y_shift_pct"] = {
        str(row["_Item Key"]): _coerce_circle_shift_pct(row.get("Circle Y Shift (%)"))
        for _, row in edited_circle_df.iterrows()
    }
    st.session_state["per_image_circle_radius_scale_pct"] = {
        str(row["_Item Key"]): _coerce_circle_size_pct(row.get("Circle Size (%)"))
        for _, row in edited_circle_df.iterrows()
    }
    circle_center_x_shift_ratios = tuple(
        _coerce_circle_shift_pct(row.get("Circle X Shift (%)")) / 100.0
        for _, row in edited_circle_df.iterrows()
    )
    circle_center_y_shift_ratios = tuple(
        _coerce_circle_shift_pct(row.get("Circle Y Shift (%)")) / 100.0
        for _, row in edited_circle_df.iterrows()
    )
    circle_radius_scales = tuple(
        _coerce_circle_size_pct(row.get("Circle Size (%)")) / 100.0
        for _, row in edited_circle_df.iterrows()
    )

    with st.spinner(f"Analyzing {len(file_items)} image(s)..."):
        batch_payload = _build_batch_payload(
            file_items=file_items,
            tray_profile_key=tray_profile_key,
            tray_long_side_values_cm=tray_long_side_values_cm,
            pixels_per_cm_override_values=pixels_per_cm_override_values,
            custom_grid_rows=custom_grid_rows,
            custom_grid_cols=custom_grid_cols,
            custom_outer_pad_ratio=(None if custom_outer_pad_pct is None else float(custom_outer_pad_pct) / 100.0),
            custom_site_pad_ratio=(None if custom_site_pad_pct is None else float(custom_site_pad_pct) / 100.0),
            container_mode=container_mode,
            circle_center_x_shift_ratios=circle_center_x_shift_ratios,
            circle_center_y_shift_ratios=circle_center_y_shift_ratios,
            circle_radius_scales=circle_radius_scales,
        )
    results_bundle_name = _results_bundle_name(file_items)
    results_zip_signature = _results_zip_signature(batch_payload, results_bundle_name)
    if st.session_state.get("prepared_results_zip_signature") != results_zip_signature:
        st.session_state.pop("prepared_results_zip_bytes", None)
        st.session_state["prepared_results_zip_signature"] = results_zip_signature

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

    has_root_summary = "Total Root Length (cm)" in image_summary_df.columns
    summary_columns = st.columns(4 if has_root_summary else 3)
    summary_col_1, summary_col_2, summary_col_3 = summary_columns[:3]
    with summary_col_1:
        st.metric("Images", int(len(batch_payload["records"])))
    with summary_col_2:
        st.metric("Total estimated leaves", int(image_summary_df["Estimated Leaves"].sum()))
    with summary_col_3:
        st.metric("Total canopy area (cm^2)", round(float(image_summary_df["Total Canopy Area (cm^2)"].sum()), 2))
    if has_root_summary:
        with summary_columns[3]:
            st.metric("Total root length (cm)", round(float(image_summary_df["Total Root Length (cm)"].fillna(0).sum()), 2))
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
        prepared_results_zip_bytes = st.session_state.get("prepared_results_zip_bytes")
        if prepared_results_zip_bytes is None:
            if st.button("Prepare Full Results ZIP", use_container_width=True):
                with st.spinner("Preparing full results ZIP..."):
                    st.session_state["prepared_results_zip_bytes"] = _build_results_bundle_bytes(batch_payload)
                st.rerun()
        else:
            st.download_button(
                "Download Full Results ZIP",
                data=prepared_results_zip_bytes,
                file_name=results_bundle_name,
                mime="application/zip",
                use_container_width=True,
            )

    st.caption(
        "The full results zip contains combined CSVs, per-image CSVs, original images, overlays, and mask outputs. "
        "It is prepared on demand to keep the web app lighter."
    )

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
