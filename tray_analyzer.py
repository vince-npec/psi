from __future__ import annotations

from dataclasses import dataclass
from math import pi

import cv2
import numpy as np
import pandas as pd

try:
    from plantcv import plantcv as pcv
except Exception:
    pcv = None


BASE_PLANT_COLORS_BGR = [
    (56, 189, 248),
    (16, 185, 129),
    (244, 114, 182),
    (250, 204, 21),
]

AUTO_TRAY_PROFILE_KEY = "auto"
GRID_2X2_PROFILE_KEY = "grid_2x2"
GRID_4X5_PROFILE_KEY = "grid_4x5"
CUSTOM_TRAY_PROFILE_KEY = "custom"
CONTAINER_MODE_AUTO = "auto"
CONTAINER_MODE_RECTANGLE = "rectangle"
CONTAINER_MODE_CIRCLE = "circle"
CONTAINER_MODE_FULL_IMAGE = "full_image"
TRAY_LONG_SIDE_CM = 33.0
MAX_TRAY_ANALYSIS_DIM = 1400
MAX_PLANT_ANALYSIS_DIM = 3000


@dataclass(frozen=True)
class TrayProfile:
    key: str
    name: str
    rows: int
    cols: int
    outer_pad_ratio_x: float = 0.05
    outer_pad_ratio_y: float = 0.05
    site_pad_ratio_x: float = 0.06
    site_pad_ratio_y: float = 0.06


TRAY_PROFILES: dict[str, TrayProfile] = {
    GRID_2X2_PROFILE_KEY: TrayProfile(
        key=GRID_2X2_PROFILE_KEY,
        name="Potato / Soybean (2x2)",
        rows=2,
        cols=2,
    ),
    GRID_4X5_PROFILE_KEY: TrayProfile(
        key=GRID_4X5_PROFILE_KEY,
        name="Arabidopsis (4x5)",
        rows=4,
        cols=5,
        outer_pad_ratio_x=0.04,
        outer_pad_ratio_y=0.04,
        site_pad_ratio_x=0.08,
        site_pad_ratio_y=0.08,
    ),
}


@dataclass(frozen=True)
class PlantRegion:
    name: str
    position: str
    site_index: int
    row_index: int
    col_index: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PlantLeafResult:
    name: str
    position: str
    bbox: tuple[int, int, int, int]
    site_bbox: tuple[int, int, int, int]
    leaf_count: int
    canopy_area_px: int
    min_leaf_area_px: int
    mask: np.ndarray
    leaf_label_map: np.ndarray
    plant_traits: dict[str, object]
    leaf_traits: list[dict[str, object]]


@dataclass(frozen=True)
class TrayAnalysisResult:
    image_shape: tuple[int, int]
    overlay_rgb: np.ndarray
    counts_df: pd.DataFrame
    plant_summary_df: pd.DataFrame
    leaf_detail_df: pd.DataFrame
    tray_bbox: tuple[int, int, int, int]
    tray_profile_key: str
    tray_profile_name: str
    grid_rows: int
    grid_cols: int
    container_source: str
    circle_center_x_shift_ratio: float
    circle_center_y_shift_ratio: float
    circle_radius_scale: float
    tray_long_side_px: float
    tray_long_side_cm: float
    pixels_per_cm: float | None
    pixels_per_cm_override: float | None
    mm_per_pixel: float | None
    scale_source: str
    plant_results: list[PlantLeafResult]
    segmentation_source: str


def get_tray_profile_options() -> list[tuple[str, str]]:
    return [
        (AUTO_TRAY_PROFILE_KEY, "Auto (2x2 or 4x5)"),
        (GRID_2X2_PROFILE_KEY, TRAY_PROFILES[GRID_2X2_PROFILE_KEY].name),
        (GRID_4X5_PROFILE_KEY, TRAY_PROFILES[GRID_4X5_PROFILE_KEY].name),
        (CUSTOM_TRAY_PROFILE_KEY, "Universal / Custom Grid"),
    ]


def get_container_mode_options() -> list[tuple[str, str]]:
    return [
        (CONTAINER_MODE_AUTO, "Auto detect"),
        (CONTAINER_MODE_RECTANGLE, "Rectangle tray"),
        (CONTAINER_MODE_CIRCLE, "Circular pot / chamber"),
        (CONTAINER_MODE_FULL_IMAGE, "Full image"),
    ]


def analyze_tray_image(
    image_rgb: np.ndarray,
    tray_profile_key: str = AUTO_TRAY_PROFILE_KEY,
    tray_long_side_cm: float = TRAY_LONG_SIDE_CM,
    pixels_per_cm_override: float | None = None,
    custom_grid_rows: int | None = None,
    custom_grid_cols: int | None = None,
    custom_outer_pad_ratio: float | None = None,
    custom_site_pad_ratio: float | None = None,
    container_mode: str = CONTAINER_MODE_AUTO,
    circular_container_inset_ratio: float = 0.04,
    circle_center_x_shift_ratio: float = 0.0,
    circle_center_y_shift_ratio: float = 0.0,
    circle_radius_scale: float = 1.0,
) -> TrayAnalysisResult:
    """Analyze a top-down tray image with adaptive ownership assignment across the full tray."""
    image_rgb = _normalize_rgb_image(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    circle_center_x_shift_ratio = _clip_ratio(circle_center_x_shift_ratio, 0.0, minimum=-0.75, maximum=0.75)
    circle_center_y_shift_ratio = _clip_ratio(circle_center_y_shift_ratio, 0.0, minimum=-0.75, maximum=0.75)
    circle_radius_scale = _clip_ratio(circle_radius_scale, 1.0, minimum=0.2, maximum=3.0)
    tray_bbox, tray_long_side_px, container_mask_u8, container_source = _detect_container_geometry(
        image_bgr,
        container_mode=container_mode,
        circular_container_inset_ratio=circular_container_inset_ratio,
        circle_center_x_shift_ratio=circle_center_x_shift_ratio,
        circle_center_y_shift_ratio=circle_center_y_shift_ratio,
        circle_radius_scale=circle_radius_scale,
    )
    tray_x0, tray_y0, tray_x1, tray_y1 = tray_bbox
    tray_crop_bgr = image_bgr[tray_y0:tray_y1, tray_x0:tray_x1]
    tray_mask = _segment_leaf_mask(tray_crop_bgr)
    tray_mask = cv2.bitwise_and(tray_mask, container_mask_u8[tray_y0:tray_y1, tray_x0:tray_x1])
    tray_profile = _resolve_tray_profile(
        tray_profile_key,
        tray_mask,
        custom_grid_rows=custom_grid_rows,
        custom_grid_cols=custom_grid_cols,
        custom_outer_pad_ratio=custom_outer_pad_ratio,
        custom_site_pad_ratio=custom_site_pad_ratio,
    )
    pixels_per_cm_override = _normalize_optional_positive_float(pixels_per_cm_override)
    pixels_per_cm, mm_per_pixel, scale_source = _resolve_scale_calibration(
        tray_long_side_px=tray_long_side_px,
        tray_long_side_cm=tray_long_side_cm,
        pixels_per_cm_override=pixels_per_cm_override,
    )
    plant_sites = _build_plant_regions(tray_bbox, image_bgr.shape[:2], tray_profile)
    ownership_masks = _assign_canopy_pixels_to_sites(tray_mask, plant_sites, tray_bbox)

    overlay_bgr = image_bgr.copy()
    plant_results: list[PlantLeafResult] = []
    plant_rows: list[dict[str, object]] = []
    leaf_rows: list[dict[str, object]] = []
    segmentation_source = "PlantCV color threshold" if pcv is not None else "OpenCV HSV/LAB threshold"

    for idx, (site, owned_mask) in enumerate(zip(plant_sites, ownership_masks)):
        analysis_bbox, owned_mask_u8 = _build_adaptive_crop_from_mask(
            ownership_mask_u8=owned_mask,
            site_bbox=site.bbox,
            tray_bbox=tray_bbox,
        )
        crop_x0, crop_y0, crop_x1, crop_y1 = analysis_bbox
        crop_bgr = image_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
        crop_raw_mask_u8 = _segment_leaf_mask(crop_bgr)
        plant_mask_u8 = cv2.bitwise_and(crop_raw_mask_u8, owned_mask_u8)
        plant_mask_u8 = cv2.bitwise_and(plant_mask_u8, container_mask_u8[crop_y0:crop_y1, crop_x0:crop_x1])
        plant_mask_u8 = _suppress_spurious_small_masks(
            plant_mask_u8=plant_mask_u8,
            analysis_bbox=analysis_bbox,
            site_bbox=site.bbox,
        )
        plant_label_map, leaf_count, canopy_area_px, min_leaf_area_px = _estimate_leaf_instances(
            plant_mask_u8,
            crop_bgr,
            expected_groups=1,
        )
        plant_traits, plant_leaf_rows = _compute_trait_rows(
            crop_bgr=crop_bgr,
            mask_u8=plant_mask_u8,
            leaf_label_map=plant_label_map,
            site=site,
            analysis_bbox=analysis_bbox,
            min_leaf_area_px=min_leaf_area_px,
            tray_profile=tray_profile,
            container_source=container_source,
            circle_center_x_shift_ratio=circle_center_x_shift_ratio,
            circle_center_y_shift_ratio=circle_center_y_shift_ratio,
            circle_radius_scale=circle_radius_scale,
            tray_long_side_cm=tray_long_side_cm,
            tray_long_side_px=tray_long_side_px,
            pixels_per_cm=pixels_per_cm,
            pixels_per_cm_override=pixels_per_cm_override,
            mm_per_pixel=mm_per_pixel,
            scale_source=scale_source,
        )

        plant_result = PlantLeafResult(
            name=site.name,
            position=site.position,
            bbox=analysis_bbox,
            site_bbox=site.bbox,
            leaf_count=int(leaf_count),
            canopy_area_px=int(canopy_area_px),
            min_leaf_area_px=int(min_leaf_area_px),
            mask=plant_mask_u8,
            leaf_label_map=plant_label_map,
            plant_traits=plant_traits,
            leaf_traits=plant_leaf_rows,
        )
        plant_results.append(plant_result)
        plant_rows.append(plant_traits)
        leaf_rows.extend(plant_leaf_rows)
        _draw_region_overlay(overlay_bgr, plant_result, color=_plant_color_bgr(idx, len(plant_sites)))

    _draw_container_outline(overlay_bgr, tray_bbox, container_mask_u8)
    plant_summary_df = pd.DataFrame(plant_rows)
    leaf_detail_df = pd.DataFrame(leaf_rows)
    counts_df = plant_summary_df[["Plant", "Position", "Estimated Leaves"]].copy()

    return TrayAnalysisResult(
        image_shape=tuple(int(v) for v in image_rgb.shape[:2]),
        overlay_rgb=cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
        counts_df=counts_df,
        plant_summary_df=plant_summary_df,
        leaf_detail_df=leaf_detail_df,
        tray_bbox=tray_bbox,
        tray_profile_key=tray_profile.key,
        tray_profile_name=tray_profile.name,
        grid_rows=tray_profile.rows,
        grid_cols=tray_profile.cols,
        container_source=container_source,
        circle_center_x_shift_ratio=circle_center_x_shift_ratio,
        circle_center_y_shift_ratio=circle_center_y_shift_ratio,
        circle_radius_scale=circle_radius_scale,
        tray_long_side_px=float(tray_long_side_px),
        tray_long_side_cm=float(tray_long_side_cm),
        pixels_per_cm=pixels_per_cm,
        pixels_per_cm_override=pixels_per_cm_override,
        mm_per_pixel=mm_per_pixel,
        scale_source=scale_source,
        plant_results=plant_results,
        segmentation_source=segmentation_source,
    )


def _normalize_rgb_image(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 2:
        return np.stack([image_rgb] * 3, axis=-1)
    if image_rgb.ndim != 3:
        raise ValueError("Expected a 2D grayscale image or a 3-channel RGB image.")
    if image_rgb.shape[2] == 4:
        return image_rgb[:, :, :3]
    if image_rgb.shape[2] != 3:
        raise ValueError("Expected a 3-channel RGB image.")
    return image_rgb


def _detect_container_geometry(
    image_bgr: np.ndarray,
    container_mode: str = CONTAINER_MODE_AUTO,
    circular_container_inset_ratio: float = 0.04,
    circle_center_x_shift_ratio: float = 0.0,
    circle_center_y_shift_ratio: float = 0.0,
    circle_radius_scale: float = 1.0,
) -> tuple[tuple[int, int, int, int], float, np.ndarray, str]:
    """Detect a rectangular tray, circular chamber, or fall back to the full image."""
    if container_mode in {CONTAINER_MODE_AUTO, CONTAINER_MODE_RECTANGLE}:
        rectangle_candidate = _detect_rectangular_tray_geometry(image_bgr)
        if rectangle_candidate is not None:
            return rectangle_candidate

    if container_mode in {CONTAINER_MODE_AUTO, CONTAINER_MODE_CIRCLE}:
        circle_candidate = _detect_circular_container_geometry(
            image_bgr,
            circular_container_inset_ratio=circular_container_inset_ratio,
            circle_center_x_shift_ratio=circle_center_x_shift_ratio,
            circle_center_y_shift_ratio=circle_center_y_shift_ratio,
            circle_radius_scale=circle_radius_scale,
            allow_default_circle=(container_mode == CONTAINER_MODE_CIRCLE),
        )
        if circle_candidate is not None:
            return circle_candidate

    height, width = image_bgr.shape[:2]
    full_mask = np.full((height, width), 255, dtype=np.uint8)
    return (0, 0, width, height), float(max(width, height)), full_mask, "Full image fallback"


def _detect_rectangular_tray_geometry(
    image_bgr: np.ndarray,
) -> tuple[tuple[int, int, int, int], float, np.ndarray, str] | None:
    """Find the blue tray carrier and estimate its long side in pixels."""
    height, width = image_bgr.shape[:2]
    working_bgr, scale = _downscale_for_analysis(image_bgr, MAX_TRAY_ANALYSIS_DIM)
    tray_mask = _segment_blue_tray_mask(working_bgr)
    tray_candidate = _select_tray_candidate(tray_mask)
    if tray_candidate is None:
        return None

    contour, (x, y, w, h) = tray_candidate
    rect = cv2.minAreaRect(contour)
    rect_w, rect_h = rect[1]
    long_side_px = float(max(rect_w, rect_h, w, h))
    full_mask = np.zeros((height, width), dtype=np.uint8)
    if scale != 1.0:
        inv = 1.0 / scale
        x = int(round(x * inv))
        y = int(round(y * inv))
        w = int(round(w * inv))
        h = int(round(h * inv))
        long_side_px = float(long_side_px * inv)
        contour = np.round(contour.astype(np.float32) * inv).astype(np.int32)
    else:
        contour = contour.astype(np.int32)
    if w < 0.4 * width or h < 0.4 * height:
        return None

    bbox = (max(0, x), max(0, y), min(width, x + w), min(height, y + h))
    cv2.drawContours(full_mask, [contour], -1, 255, thickness=-1)
    if not np.any(full_mask > 0):
        full_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 255
    return bbox, max(1.0, float(long_side_px)), full_mask, "Detected blue tray"


def _detect_circular_container_geometry(
    image_bgr: np.ndarray,
    circular_container_inset_ratio: float = 0.04,
    circle_center_x_shift_ratio: float = 0.0,
    circle_center_y_shift_ratio: float = 0.0,
    circle_radius_scale: float = 1.0,
    allow_default_circle: bool = False,
) -> tuple[tuple[int, int, int, int], float, np.ndarray, str] | None:
    height, width = image_bgr.shape[:2]
    working_bgr, scale = _downscale_for_analysis(image_bgr, MAX_TRAY_ANALYSIS_DIM)
    gray = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 9)
    image_h, image_w = blur.shape[:2]
    min_dim = min(image_h, image_w)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(50, min_dim // 4),
        param1=120,
        param2=45,
        minRadius=max(40, int(min_dim * 0.22)),
        maxRadius=max(60, int(min_dim * 0.49)),
    )
    if circles is None:
        if not allow_default_circle:
            return None
        center_x = image_w // 2
        center_y = image_h // 2
        radius = int(round(min_dim * 0.42))
        source_label = "Default circular pot"
    else:
        candidates = np.round(circles[0]).astype(np.int32)
        best_candidate: tuple[int, int, int] | None = None
        best_score = -1e9
        for center_x, center_y, radius in candidates.tolist():
            if radius <= 0:
                continue
            center_distance = float(np.hypot(center_x - (image_w / 2.0), center_y - (image_h / 2.0))) / max(1.0, min_dim)
            radius_ratio = float(radius) / float(max(1, min_dim))
            score = (radius_ratio * 3.0) - center_distance
            if score > best_score:
                best_score = score
                best_candidate = (int(center_x), int(center_y), int(radius))

        if best_candidate is None:
            return None

        center_x, center_y, radius = best_candidate
        source_label = "Detected circular pot"
    inset_ratio = _clip_ratio(circular_container_inset_ratio, 0.04, minimum=0.0, maximum=0.2)
    effective_radius = max(12, int(round(radius * (1.0 - inset_ratio))))
    if scale != 1.0:
        inv = 1.0 / scale
        center_x = int(round(center_x * inv))
        center_y = int(round(center_y * inv))
        radius = int(round(radius * inv))
        effective_radius = int(round(effective_radius * inv))

    center_x = int(round(center_x + (width * circle_center_x_shift_ratio)))
    center_y = int(round(center_y + (height * circle_center_y_shift_ratio)))
    effective_radius = max(12, int(round(effective_radius * circle_radius_scale)))
    center_x = int(min(width - 1, max(0, center_x)))
    center_y = int(min(height - 1, max(0, center_y)))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), max(1, effective_radius), 255, thickness=-1)
    bbox = (
        max(0, center_x - effective_radius),
        max(0, center_y - effective_radius),
        min(width, center_x + effective_radius),
        min(height, center_y + effective_radius),
    )
    if bbox[2] - bbox[0] < 30 or bbox[3] - bbox[1] < 30:
        return None

    adjusted = (
        abs(circle_center_x_shift_ratio) > 1e-6
        or abs(circle_center_y_shift_ratio) > 1e-6
        or abs(circle_radius_scale - 1.0) > 1e-6
    )
    source = f"{source_label} + user adjustment" if adjusted else source_label
    return bbox, max(1.0, float(effective_radius * 2.0)), mask, source


def _segment_blue_tray_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    blue = image_bgr[:, :, 0].astype(np.int16)
    green = image_bgr[:, :, 1].astype(np.int16)
    red = image_bgr[:, :, 2].astype(np.int16)

    hue_mask = cv2.inRange(
        hsv,
        np.array([88, 45, 35], dtype=np.uint8),
        np.array([130, 255, 255], dtype=np.uint8),
    )
    dominance_mask = ((blue - np.maximum(green, red)) > 18).astype(np.uint8) * 255
    tray_mask = cv2.bitwise_and(hue_mask, dominance_mask)
    tray_mask = cv2.morphologyEx(
        tray_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    tray_mask = cv2.morphologyEx(
        tray_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    return tray_mask


def _select_tray_candidate(mask_u8: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    binary = (mask_u8 > 0).astype(np.uint8)
    if not np.any(binary):
        return None

    image_h, image_w = binary.shape[:2]
    image_area = float(max(1, image_h * image_w))
    image_diag = float(np.hypot(image_w, image_h))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    best_candidate: tuple[np.ndarray, tuple[int, int, int, int]] | None = None
    best_score = -1e9

    for min_area_ratio in (0.18, 0.08):
        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            area_ratio = float(area) / image_area
            if area_ratio < min_area_ratio:
                continue
            if w < 0.35 * image_w or h < 0.35 * image_h:
                continue

            border_touches = int(x <= 1) + int(y <= 1) + int(x + w >= image_w - 1) + int(y + h >= image_h - 1)
            rectangularity = float(area) / float(max(1, w * h))
            center_x = float(x) + (float(w) / 2.0)
            center_y = float(y) + (float(h) / 2.0)
            center_distance = float(np.hypot(center_x - (image_w / 2.0), center_y - (image_h / 2.0))) / max(1.0, image_diag)

            component_mask = ((labels == idx).astype(np.uint8) * 255)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(contour))
            if contour_area <= 0:
                continue

            score = (
                (area_ratio * 3.0)
                + (rectangularity * 2.0)
                - (border_touches * 0.6)
                - (center_distance * 0.8)
            )
            if score <= best_score:
                continue

            best_score = score
            best_candidate = (contour, (int(x), int(y), int(w), int(h)))

        if best_candidate is not None:
            break

    return best_candidate


def _tray_scale_from_long_side(tray_long_side_px: float, tray_long_side_cm: float) -> tuple[float | None, float | None]:
    if tray_long_side_px <= 0 or tray_long_side_cm <= 0:
        return None, None
    pixels_per_cm = float(tray_long_side_px) / float(tray_long_side_cm)
    mm_per_pixel = float(tray_long_side_cm * 10.0) / float(tray_long_side_px)
    return _round_or_none(pixels_per_cm, 4), _round_or_none(mm_per_pixel, 4)


def _scale_from_pixels_per_cm(pixels_per_cm: float | None) -> tuple[float | None, float | None]:
    if pixels_per_cm is None or pixels_per_cm <= 0:
        return None, None
    mm_per_pixel = 10.0 / float(pixels_per_cm)
    return _round_or_none(pixels_per_cm, 4), _round_or_none(mm_per_pixel, 4)


def _resolve_scale_calibration(
    tray_long_side_px: float,
    tray_long_side_cm: float,
    pixels_per_cm_override: float | None,
) -> tuple[float | None, float | None, str]:
    if pixels_per_cm_override is not None and pixels_per_cm_override > 0:
        pixels_per_cm, mm_per_pixel = _scale_from_pixels_per_cm(pixels_per_cm_override)
        return pixels_per_cm, mm_per_pixel, "User pixels/cm override"

    pixels_per_cm, mm_per_pixel = _tray_scale_from_long_side(tray_long_side_px, tray_long_side_cm)
    return pixels_per_cm, mm_per_pixel, "Detected tray long side"


def _resolve_tray_profile(
    requested_key: str,
    tray_mask_u8: np.ndarray,
    custom_grid_rows: int | None = None,
    custom_grid_cols: int | None = None,
    custom_outer_pad_ratio: float | None = None,
    custom_site_pad_ratio: float | None = None,
) -> TrayProfile:
    if requested_key == CUSTOM_TRAY_PROFILE_KEY:
        return _build_custom_tray_profile(
            custom_grid_rows=custom_grid_rows,
            custom_grid_cols=custom_grid_cols,
            custom_outer_pad_ratio=custom_outer_pad_ratio,
            custom_site_pad_ratio=custom_site_pad_ratio,
        )

    if requested_key in TRAY_PROFILES:
        return TRAY_PROFILES[requested_key]

    group_count = _estimate_plant_group_count(tray_mask_u8)
    if group_count >= 12:
        return TRAY_PROFILES[GRID_4X5_PROFILE_KEY]
    return TRAY_PROFILES[GRID_2X2_PROFILE_KEY]


def _build_custom_tray_profile(
    custom_grid_rows: int | None = None,
    custom_grid_cols: int | None = None,
    custom_outer_pad_ratio: float | None = None,
    custom_site_pad_ratio: float | None = None,
) -> TrayProfile:
    rows = max(1, int(custom_grid_rows or 1))
    cols = max(1, int(custom_grid_cols or 1))
    total_sites = rows * cols
    default_outer_pad_ratio = 0.04 if total_sites >= 12 else 0.05
    default_site_pad_ratio = 0.08 if total_sites >= 12 else 0.06
    outer_pad_ratio = _clip_ratio(custom_outer_pad_ratio, default_outer_pad_ratio, minimum=0.0, maximum=0.25)
    site_pad_ratio = _clip_ratio(custom_site_pad_ratio, default_site_pad_ratio, minimum=0.0, maximum=0.3)

    return TrayProfile(
        key=CUSTOM_TRAY_PROFILE_KEY,
        name=f"Universal ({rows}x{cols})",
        rows=rows,
        cols=cols,
        outer_pad_ratio_x=outer_pad_ratio,
        outer_pad_ratio_y=outer_pad_ratio,
        site_pad_ratio_x=site_pad_ratio,
        site_pad_ratio_y=site_pad_ratio,
    )


def _estimate_plant_group_count(mask_u8: np.ndarray) -> int:
    binary = (mask_u8 > 0).astype(np.uint8)
    if not np.any(binary):
        return 0

    min_group_area_px = max(250, int(mask_u8.size * 0.00015))
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return int(sum(1 for idx in range(1, num_labels) if int(stats[idx, cv2.CC_STAT_AREA]) >= min_group_area_px))


def _build_plant_regions(
    tray_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    tray_profile: TrayProfile,
) -> list[PlantRegion]:
    """Build plant ownership sites from tray geometry and the selected grid profile."""
    x0, y0, x1, y1 = tray_bbox
    image_h, image_w = image_shape

    tray_w = max(1, x1 - x0)
    tray_h = max(1, y1 - y0)
    outer_pad_x = int(tray_w * tray_profile.outer_pad_ratio_x)
    outer_pad_y = int(tray_h * tray_profile.outer_pad_ratio_y)

    inner_x0 = min(max(0, x0 + outer_pad_x), image_w - 1)
    inner_y0 = min(max(0, y0 + outer_pad_y), image_h - 1)
    inner_x1 = max(inner_x0 + 1, min(image_w, x1 - outer_pad_x))
    inner_y1 = max(inner_y0 + 1, min(image_h, y1 - outer_pad_y))

    x_edges = np.linspace(inner_x0, inner_x1, tray_profile.cols + 1)
    y_edges = np.linspace(inner_y0, inner_y1, tray_profile.rows + 1)

    plant_regions: list[PlantRegion] = []
    site_index = 0
    for row_idx in range(tray_profile.rows):
        for col_idx in range(tray_profile.cols):
            cell_x0 = int(round(float(x_edges[col_idx])))
            cell_x1 = int(round(float(x_edges[col_idx + 1])))
            cell_y0 = int(round(float(y_edges[row_idx])))
            cell_y1 = int(round(float(y_edges[row_idx + 1])))
            pad_x = int((cell_x1 - cell_x0) * tray_profile.site_pad_ratio_x)
            pad_y = int((cell_y1 - cell_y0) * tray_profile.site_pad_ratio_y)
            site_index += 1
            name = f"Plant {site_index}"
            position = _format_grid_position(row_idx, col_idx, tray_profile.rows, tray_profile.cols)
            bbox = (
                max(0, cell_x0 + pad_x),
                max(0, cell_y0 + pad_y),
                min(image_w, cell_x1 - pad_x),
                min(image_h, cell_y1 - pad_y),
            )
            plant_regions.append(
                PlantRegion(
                    name=name,
                    position=position,
                    site_index=site_index,
                    row_index=row_idx,
                    col_index=col_idx,
                    bbox=bbox,
                )
            )

    return plant_regions


def _format_grid_position(row_idx: int, col_idx: int, rows: int, cols: int) -> str:
    if rows == 2 and cols == 2:
        mapping = {
            (0, 0): "Top-left",
            (0, 1): "Top-right",
            (1, 0): "Bottom-left",
            (1, 1): "Bottom-right",
        }
        return mapping[(row_idx, col_idx)]
    return f"Row {row_idx + 1}, Col {col_idx + 1}"


def _plant_color_bgr(index: int, total: int) -> tuple[int, int, int]:
    if total <= len(BASE_PLANT_COLORS_BGR):
        return BASE_PLANT_COLORS_BGR[index % len(BASE_PLANT_COLORS_BGR)]

    hue = int(round((180.0 * index) / max(1, total)))
    hsv_color = np.uint8([[[hue % 180, 200, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])


def _segment_leaf_mask(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros(crop_bgr.shape[:2], dtype=np.uint8)

    if pcv is not None:
        try:
            return _segment_with_plantcv(crop_bgr)
        except Exception:
            pass

    return _segment_with_opencv(crop_bgr)


def _segment_with_opencv(crop_bgr: np.ndarray) -> np.ndarray:
    working_bgr, scale = _downscale_for_analysis(crop_bgr, MAX_PLANT_ANALYSIS_DIM)
    hsv = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2LAB)

    hsv_mask = cv2.inRange(
        hsv,
        np.array([30, 35, 20], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    lab_mask = cv2.inRange(lab[:, :, 1], 0, 124)
    mask = cv2.bitwise_and(hsv_mask, lab_mask)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=2,
    )
    if scale != 1.0:
        mask = cv2.resize(mask, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _segment_with_plantcv(crop_bgr: np.ndarray) -> np.ndarray:
    working_bgr, scale = _downscale_for_analysis(crop_bgr, MAX_PLANT_ANALYSIS_DIM)
    crop_rgb = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2RGB)
    hsv_mask, _ = _pcv_custom_range(
        crop_rgb,
        lower_thresh=[30, 35, 20],
        upper_thresh=[95, 255, 255],
        channel="HSV",
    )
    lab_mask, _ = _pcv_custom_range(
        crop_rgb,
        lower_thresh=[0, 0, 0],
        upper_thresh=[255, 124, 255],
        channel="LAB",
    )
    mask = cv2.bitwise_and(hsv_mask, lab_mask)

    try:
        mask = pcv.fill(bin_img=mask, size=max(64, mask.size // 400))
    except TypeError:
        mask = pcv.fill(mask, max(64, mask.size // 400))

    try:
        mask = pcv.fill_holes(mask)
    except Exception:
        pass

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=2,
    )
    if scale != 1.0:
        mask = cv2.resize(mask, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _pcv_custom_range(image_rgb: np.ndarray, lower_thresh: list[int], upper_thresh: list[int], channel: str):
    try:
        return pcv.threshold.custom_range(
            img=image_rgb,
            lower_thresh=lower_thresh,
            upper_thresh=upper_thresh,
            channel=channel,
        )
    except TypeError:
        return pcv.threshold.custom_range(
            rgb_img=image_rgb,
            lower_thresh=lower_thresh,
            upper_thresh=upper_thresh,
            channel=channel,
        )


def _estimate_leaf_instances(
    mask_u8: np.ndarray,
    crop_bgr: np.ndarray,
    expected_groups: int = 1,
) -> tuple[np.ndarray, int, int, int]:
    working_bgr, scale = _downscale_for_analysis(crop_bgr, MAX_PLANT_ANALYSIS_DIM)
    if scale != 1.0:
        working_mask = cv2.resize(
            mask_u8,
            (working_bgr.shape[1], working_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        working_mask = mask_u8

    binary = (working_mask > 0).astype(np.uint8) * 255
    canopy_area_px = int(np.count_nonzero(mask_u8))
    working_canopy_area_px = int(np.count_nonzero(binary))
    empty_cell_cutoff = max(400, int(binary.size * 0.004))
    if canopy_area_px < empty_cell_cutoff:
        return _restore_label_map(np.zeros(binary.shape, dtype=np.int32), mask_u8.shape, scale), 0, canopy_area_px, empty_cell_cutoff

    groups = max(1, int(expected_groups))
    min_leaf_area_px = max(180, int((working_canopy_area_px * 0.012) / groups))
    kernel = np.ones((3, 3), dtype=np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    distance = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    max_distance = float(np.nanmax(distance)) if distance.size else 0.0
    if not np.isfinite(max_distance) or max_distance <= 0:
        return _restore_fallback_result(_component_fallback(clean, min_leaf_area_px, canopy_area_px), mask_u8.shape, scale)

    sure_fg = ((distance >= max(2.0, 0.22 * max_distance)).astype(np.uint8) * 255)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(clean, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    marker_count, markers = cv2.connectedComponents((sure_fg > 0).astype(np.uint8))
    if marker_count <= 1:
        return _restore_fallback_result(_component_fallback(clean, min_leaf_area_px, canopy_area_px), mask_u8.shape, scale)

    markers = markers + 1
    markers[unknown == 255] = 0

    try:
        markers = cv2.watershed(working_bgr.copy(), markers)
    except Exception:
        return _restore_fallback_result(_component_fallback(clean, min_leaf_area_px, canopy_area_px), mask_u8.shape, scale)

    label_map = np.zeros(clean.shape, dtype=np.int32)
    next_label = 1
    for raw_label in sorted(int(value) for value in np.unique(markers) if int(value) > 1):
        region = (markers == raw_label) & (clean > 0)
        if int(np.count_nonzero(region)) < min_leaf_area_px:
            continue
        label_map[region] = next_label
        next_label += 1

    leaf_count = int(np.max(label_map))
    if leaf_count == 0:
        return _restore_fallback_result(_component_fallback(clean, min_leaf_area_px, canopy_area_px), mask_u8.shape, scale)

    return _restore_label_map(label_map, mask_u8.shape, scale), leaf_count, canopy_area_px, min_leaf_area_px


def _assign_canopy_pixels_to_sites(
    tray_mask_u8: np.ndarray,
    plant_sites: list[PlantRegion],
    tray_bbox: tuple[int, int, int, int],
) -> list[np.ndarray]:
    tray_x0, tray_y0, _, _ = tray_bbox
    canopy = tray_mask_u8 > 0
    if not np.any(canopy):
        return [np.zeros(tray_mask_u8.shape, dtype=np.uint8) for _ in plant_sites]

    site_centers = [
        (
            float((site.bbox[0] + site.bbox[2]) / 2.0),
            float((site.bbox[1] + site.bbox[3]) / 2.0),
        )
        for site in plant_sites
    ]
    ys, xs = np.where(canopy)
    x_coords = xs.astype(np.float32) + float(tray_x0)
    y_coords = ys.astype(np.float32) + float(tray_y0)
    anchors = np.asarray(site_centers, dtype=np.float32)
    dist_sq = (
        (x_coords[:, None] - anchors[None, :, 0]) ** 2
        + (y_coords[:, None] - anchors[None, :, 1]) ** 2
    )
    owners = np.argmin(dist_sq, axis=1)

    ownership_masks = [np.zeros(tray_mask_u8.shape, dtype=np.uint8) for _ in plant_sites]
    for site_idx in range(len(plant_sites)):
        selected = owners == site_idx
        if not np.any(selected):
            continue
        ownership_masks[site_idx][ys[selected], xs[selected]] = 255

    return ownership_masks


def _build_adaptive_crop_from_mask(
    ownership_mask_u8: np.ndarray,
    site_bbox: tuple[int, int, int, int],
    tray_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    tray_x0, tray_y0, _, _ = tray_bbox
    site_x0, site_y0, site_x1, site_y1 = site_bbox

    ys, xs = np.where(ownership_mask_u8 > 0)
    if xs.size == 0 or ys.size == 0:
        empty_h = max(1, site_y1 - site_y0)
        empty_w = max(1, site_x1 - site_x0)
        return site_bbox, np.zeros((empty_h, empty_w), dtype=np.uint8)

    tray_h, tray_w = ownership_mask_u8.shape[:2]
    pad = max(16, int(round(0.015 * max(tray_w, tray_h))))
    local_x0 = max(0, int(xs.min()) - pad)
    local_y0 = max(0, int(ys.min()) - pad)
    local_x1 = min(tray_w, int(xs.max()) + 1 + pad)
    local_y1 = min(tray_h, int(ys.max()) + 1 + pad)
    analysis_bbox = (
        tray_x0 + local_x0,
        tray_y0 + local_y0,
        tray_x0 + local_x1,
        tray_y0 + local_y1,
    )
    return analysis_bbox, ownership_mask_u8[local_y0:local_y1, local_x0:local_x1].copy()


def _component_fallback(clean_mask: np.ndarray, min_leaf_area_px: int, canopy_area_px: int) -> tuple[np.ndarray, int, int, int]:
    label_map = np.zeros(clean_mask.shape, dtype=np.int32)
    count = 0
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((clean_mask > 0).astype(np.uint8), connectivity=8)
    for idx in range(1, component_count):
        if int(stats[idx, cv2.CC_STAT_AREA]) < min_leaf_area_px:
            continue
        count += 1
        label_map[labels == idx] = count
    return label_map, count, canopy_area_px, min_leaf_area_px


def _restore_fallback_result(
    result: tuple[np.ndarray, int, int, int],
    original_shape: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, int, int, int]:
    label_map, leaf_count, canopy_area_px, min_leaf_area_px = result
    return _restore_label_map(label_map, original_shape, scale), leaf_count, canopy_area_px, min_leaf_area_px


def _restore_label_map(label_map: np.ndarray, original_shape: tuple[int, int], scale: float) -> np.ndarray:
    if scale == 1.0 or label_map.shape[:2] == original_shape[:2]:
        return label_map
    return cv2.resize(
        label_map.astype(np.int32),
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)


def _downscale_for_analysis(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Return a smaller working image plus the scale factor back to the original size."""
    if image.ndim < 2:
        return image, 1.0

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image, 1.0

    scale = max_dim / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def _suppress_spurious_small_masks(
    plant_mask_u8: np.ndarray,
    analysis_bbox: tuple[int, int, int, int],
    site_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    mask_bool = plant_mask_u8 > 0
    if not np.any(mask_bool):
        return plant_mask_u8

    mask_stats = _shape_stats_from_mask(mask_bool)
    area_px = int(mask_stats["area_px"])
    if area_px <= 0:
        return np.zeros_like(plant_mask_u8)

    site_x0, site_y0, site_x1, site_y1 = site_bbox
    site_area_px = max(1, (site_x1 - site_x0) * (site_y1 - site_y0))
    site_center_x = float((site_x0 + site_x1) / 2.0)
    site_center_y = float((site_y0 + site_y1) / 2.0)
    site_diag_px = float(((site_x1 - site_x0) ** 2 + (site_y1 - site_y0) ** 2) ** 0.5)

    bbox_x0, bbox_y0, _, _ = analysis_bbox
    canopy_centroid_x = float(bbox_x0) + float(mask_stats["centroid_x"])
    canopy_centroid_y = float(bbox_y0) + float(mask_stats["centroid_y"])
    centroid_distance_px = float(
        ((canopy_centroid_x - site_center_x) ** 2 + (canopy_centroid_y - site_center_y) ** 2) ** 0.5
    )

    small_area_threshold = max(1800, int(site_area_px * 0.0015))
    far_distance_threshold = 0.28 * site_diag_px
    if area_px < small_area_threshold and centroid_distance_px > far_distance_threshold:
        return np.zeros_like(plant_mask_u8)

    return plant_mask_u8


def _compute_trait_rows(
    crop_bgr: np.ndarray,
    mask_u8: np.ndarray,
    leaf_label_map: np.ndarray,
    site: PlantRegion,
    analysis_bbox: tuple[int, int, int, int],
    min_leaf_area_px: int,
    tray_profile: TrayProfile,
    container_source: str,
    circle_center_x_shift_ratio: float,
    circle_center_y_shift_ratio: float,
    circle_radius_scale: float,
    tray_long_side_cm: float,
    tray_long_side_px: float,
    pixels_per_cm: float | None,
    pixels_per_cm_override: float | None,
    mm_per_pixel: float | None,
    scale_source: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    binary = mask_u8 > 0
    canopy_stats = _shape_stats_from_mask(binary)
    canopy_color = _masked_color_stats(crop_bgr, binary)
    segment_count = _count_connected_segments(binary)
    x0, y0, x1, y1 = analysis_bbox
    site_x0, site_y0, site_x1, site_y1 = site.bbox
    site_center_x = float((site_x0 + site_x1) / 2.0)
    site_center_y = float((site_y0 + site_y1) / 2.0)
    analysis_area_px = int(mask_u8.size)
    coverage_pct = 100.0 * float(canopy_stats["area_px"]) / float(analysis_area_px) if analysis_area_px > 0 else 0.0

    leaf_rows: list[dict[str, object]] = []
    leaf_areas: list[float] = []
    leaf_angles: list[float] = []

    for leaf_id in [int(value) for value in np.unique(leaf_label_map) if int(value) > 0]:
        leaf_mask = leaf_label_map == leaf_id
        leaf_stats = _shape_stats_from_mask(leaf_mask)
        leaf_color = _masked_color_stats(crop_bgr, leaf_mask)
        if leaf_stats["area_px"] <= 0:
            continue

        leaf_areas.append(float(leaf_stats["area_px"]))
        if leaf_stats["orientation_deg"] is not None:
            leaf_angles.append(float(leaf_stats["orientation_deg"]))

        global_centroid_x = x0 + float(leaf_stats["centroid_x"])
        global_centroid_y = y0 + float(leaf_stats["centroid_y"])
        ownership_distance_px = float(
            ((global_centroid_x - site_center_x) ** 2 + (global_centroid_y - site_center_y) ** 2) ** 0.5
        )
        leaf_rows.append(
            {
                "Plant": site.name,
                "Position": site.position,
                "Site Index": int(site.site_index),
                "Grid Row": int(site.row_index + 1),
                "Grid Column": int(site.col_index + 1),
                "Tray Profile": tray_profile.name,
                "Grid Rows": int(tray_profile.rows),
                "Grid Columns": int(tray_profile.cols),
                "Container Source": container_source,
                "Circle Center X Shift (%)": _round_or_none(circle_center_x_shift_ratio * 100.0, 2),
                "Circle Center Y Shift (%)": _round_or_none(circle_center_y_shift_ratio * 100.0, 2),
                "Circle Radius Scale (%)": _round_or_none(circle_radius_scale * 100.0, 2),
                "Scale Source": scale_source,
                "Pixels Per Cm Override": pixels_per_cm_override,
                "Leaf ID": int(leaf_id),
                "Ownership Distance (px)": _round_or_none(ownership_distance_px, 2),
                "Ownership Distance (cm)": _length_px_to_cm(ownership_distance_px, pixels_per_cm),
                "Area (px)": int(leaf_stats["area_px"]),
                "Area (cm^2)": _area_px_to_cm2(leaf_stats["area_px"], pixels_per_cm),
                "Area Of Canopy (%)": _round_or_none(
                    100.0 * float(leaf_stats["area_px"]) / float(canopy_stats["area_px"]),
                    2,
                )
                if canopy_stats["area_px"] > 0
                else 0.0,
                "Perimeter (px)": leaf_stats["perimeter_px"],
                "Perimeter (cm)": _length_px_to_cm(leaf_stats["perimeter_px"], pixels_per_cm),
                "Convex Hull Area (px)": leaf_stats["convex_hull_area_px"],
                "Convex Hull Area (cm^2)": _area_px_to_cm2(leaf_stats["convex_hull_area_px"], pixels_per_cm),
                "Solidity": leaf_stats["solidity"],
                "Circularity": leaf_stats["circularity"],
                "Extent": leaf_stats["extent"],
                "Orientation (deg)": leaf_stats["orientation_deg"],
                "Major Axis (px)": leaf_stats["major_axis_px"],
                "Major Axis (cm)": _length_px_to_cm(leaf_stats["major_axis_px"], pixels_per_cm),
                "Minor Axis (px)": leaf_stats["minor_axis_px"],
                "Minor Axis (cm)": _length_px_to_cm(leaf_stats["minor_axis_px"], pixels_per_cm),
                "Aspect Ratio": leaf_stats["aspect_ratio"],
                "Equivalent Diameter (px)": leaf_stats["equivalent_diameter_px"],
                "Equivalent Diameter (cm)": _length_px_to_cm(leaf_stats["equivalent_diameter_px"], pixels_per_cm),
                "Centroid X": _round_or_none(global_centroid_x, 2),
                "Centroid Y": _round_or_none(global_centroid_y, 2),
                "BBox X0": int(x0 + leaf_stats["bbox_x0"]),
                "BBox Y0": int(y0 + leaf_stats["bbox_y0"]),
                "BBox X1": int(x0 + leaf_stats["bbox_x1"]),
                "BBox Y1": int(y0 + leaf_stats["bbox_y1"]),
                "BBox Width (px)": int(leaf_stats["bbox_width_px"]),
                "BBox Width (cm)": _length_px_to_cm(leaf_stats["bbox_width_px"], pixels_per_cm),
                "BBox Height (px)": int(leaf_stats["bbox_height_px"]),
                "BBox Height (cm)": _length_px_to_cm(leaf_stats["bbox_height_px"], pixels_per_cm),
                "Mean R": leaf_color["mean_r"],
                "Mean G": leaf_color["mean_g"],
                "Mean B": leaf_color["mean_b"],
                "Mean H": leaf_color["mean_h"],
                "Mean S": leaf_color["mean_s"],
                "Mean V": leaf_color["mean_v"],
                "ExG Mean": leaf_color["exg_mean"],
            }
        )

    labeled_leaf_area_px = int(sum(leaf_areas))
    plant_row = {
        "Plant": site.name,
        "Position": site.position,
        "Site Index": int(site.site_index),
        "Grid Row": int(site.row_index + 1),
        "Grid Column": int(site.col_index + 1),
        "Tray Profile": tray_profile.name,
        "Grid Rows": int(tray_profile.rows),
        "Grid Columns": int(tray_profile.cols),
        "Container Source": container_source,
        "Circle Center X Shift (%)": _round_or_none(circle_center_x_shift_ratio * 100.0, 2),
        "Circle Center Y Shift (%)": _round_or_none(circle_center_y_shift_ratio * 100.0, 2),
        "Circle Radius Scale (%)": _round_or_none(circle_radius_scale * 100.0, 2),
        "Scale Source": scale_source,
        "Tray Long Side (px)": _round_or_none(tray_long_side_px, 2),
        "Tray Long Side (cm)": float(tray_long_side_cm),
        "Pixels Per Cm": pixels_per_cm,
        "Pixels Per Cm Override": pixels_per_cm_override,
        "Mm Per Pixel": mm_per_pixel,
        "Estimated Leaves": int(len(leaf_rows)),
        "Canopy Area (px)": int(canopy_stats["area_px"]),
        "Canopy Area (cm^2)": _area_px_to_cm2(canopy_stats["area_px"], pixels_per_cm),
        "Convex Hull Area (px)": canopy_stats["convex_hull_area_px"],
        "Convex Hull Area (cm^2)": _area_px_to_cm2(canopy_stats["convex_hull_area_px"], pixels_per_cm),
        "Solidity": canopy_stats["solidity"],
        "Perimeter (px)": canopy_stats["perimeter_px"],
        "Perimeter (cm)": _length_px_to_cm(canopy_stats["perimeter_px"], pixels_per_cm),
        "Coverage (%)": _round_or_none(coverage_pct, 2),
        "Segment Count": int(segment_count),
        "Canopy Angle (deg)": canopy_stats["orientation_deg"],
        "Canopy Major Axis (px)": canopy_stats["major_axis_px"],
        "Canopy Major Axis (cm)": _length_px_to_cm(canopy_stats["major_axis_px"], pixels_per_cm),
        "Canopy Minor Axis (px)": canopy_stats["minor_axis_px"],
        "Canopy Minor Axis (cm)": _length_px_to_cm(canopy_stats["minor_axis_px"], pixels_per_cm),
        "Canopy Aspect Ratio": canopy_stats["aspect_ratio"],
        "Canopy Equivalent Diameter (px)": canopy_stats["equivalent_diameter_px"],
        "Canopy Equivalent Diameter (cm)": _length_px_to_cm(canopy_stats["equivalent_diameter_px"], pixels_per_cm),
        "Canopy Centroid X": _round_or_none(x0 + canopy_stats["centroid_x"], 2) if canopy_stats["centroid_x"] is not None else None,
        "Canopy Centroid Y": _round_or_none(y0 + canopy_stats["centroid_y"], 2) if canopy_stats["centroid_y"] is not None else None,
        "Canopy BBox X0": int(x0 + canopy_stats["bbox_x0"]),
        "Canopy BBox Y0": int(y0 + canopy_stats["bbox_y0"]),
        "Canopy BBox X1": int(x0 + canopy_stats["bbox_x1"]),
        "Canopy BBox Y1": int(y0 + canopy_stats["bbox_y1"]),
        "Canopy BBox Width (px)": int(canopy_stats["bbox_width_px"]),
        "Canopy BBox Width (cm)": _length_px_to_cm(canopy_stats["bbox_width_px"], pixels_per_cm),
        "Canopy BBox Height (px)": int(canopy_stats["bbox_height_px"]),
        "Canopy BBox Height (cm)": _length_px_to_cm(canopy_stats["bbox_height_px"], pixels_per_cm),
        "Site X0": int(site_x0),
        "Site Y0": int(site_y0),
        "Site X1": int(site_x1),
        "Site Y1": int(site_y1),
        "Site Center X": _round_or_none(site_center_x, 2),
        "Site Center Y": _round_or_none(site_center_y, 2),
        "Adaptive Crop X0": int(x0),
        "Adaptive Crop Y0": int(y0),
        "Adaptive Crop X1": int(x1),
        "Adaptive Crop Y1": int(y1),
        "Min Leaf Area Threshold (px)": int(min_leaf_area_px),
        "Min Leaf Area Threshold (cm^2)": _area_px_to_cm2(min_leaf_area_px, pixels_per_cm),
        "Leaf Area Sum (px)": int(labeled_leaf_area_px),
        "Leaf Area Sum (cm^2)": _area_px_to_cm2(labeled_leaf_area_px, pixels_per_cm),
        "Mean Leaf Area (px)": _round_or_none(np.mean(leaf_areas), 2) if leaf_areas else 0.0,
        "Mean Leaf Area (cm^2)": _area_px_to_cm2(np.mean(leaf_areas), pixels_per_cm) if leaf_areas else 0.0,
        "Median Leaf Area (px)": _round_or_none(np.median(leaf_areas), 2) if leaf_areas else 0.0,
        "Median Leaf Area (cm^2)": _area_px_to_cm2(np.median(leaf_areas), pixels_per_cm) if leaf_areas else 0.0,
        "Largest Leaf Area (px)": int(max(leaf_areas)) if leaf_areas else 0,
        "Largest Leaf Area (cm^2)": _area_px_to_cm2(max(leaf_areas), pixels_per_cm) if leaf_areas else 0.0,
        "Mean Leaf Angle (deg)": _round_or_none(np.mean(leaf_angles), 2) if leaf_angles else None,
        "Mean R": canopy_color["mean_r"],
        "Mean G": canopy_color["mean_g"],
        "Mean B": canopy_color["mean_b"],
        "Mean H": canopy_color["mean_h"],
        "Mean S": canopy_color["mean_s"],
        "Mean V": canopy_color["mean_v"],
        "ExG Mean": canopy_color["exg_mean"],
    }
    return plant_row, leaf_rows


def _normalize_optional_positive_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _clip_ratio(
    value: float | int | None,
    default: float,
    minimum: float = 0.0,
    maximum: float = 0.3,
) -> float:
    if value is None:
        return float(default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(min(maximum, max(minimum, numeric)))


def _count_connected_segments(mask_bool: np.ndarray, min_area_px: int = 50) -> int:
    binary = mask_bool.astype(np.uint8)
    if not np.any(binary):
        return 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return int(sum(1 for idx in range(1, num_labels) if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area_px))


def _shape_stats_from_mask(mask_bool: np.ndarray) -> dict[str, object]:
    if mask_bool is None or not np.any(mask_bool):
        return {
            "area_px": 0,
            "perimeter_px": 0.0,
            "convex_hull_area_px": 0.0,
            "solidity": None,
            "circularity": None,
            "extent": None,
            "equivalent_diameter_px": 0.0,
            "orientation_deg": None,
            "major_axis_px": 0.0,
            "minor_axis_px": 0.0,
            "aspect_ratio": None,
            "centroid_x": None,
            "centroid_y": None,
            "bbox_x0": 0,
            "bbox_y0": 0,
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_width_px": 0,
            "bbox_height_px": 0,
        }

    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ys, xs = np.where(mask_bool)
    area_px = int(xs.size)
    bbox_x0 = int(xs.min())
    bbox_y0 = int(ys.min())
    bbox_x1 = int(xs.max()) + 1
    bbox_y1 = int(ys.max()) + 1
    bbox_width_px = int(bbox_x1 - bbox_x0)
    bbox_height_px = int(bbox_y1 - bbox_y0)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())

    perimeter_px = 0.0
    convex_hull_area_px = 0.0
    if contours:
        perimeter_px = float(sum(cv2.arcLength(contour, True) for contour in contours))
        pts = np.vstack(contours)
        if len(pts) >= 3:
            hull = cv2.convexHull(pts)
            convex_hull_area_px = float(cv2.contourArea(hull))

    extent = float(area_px) / float(max(1, bbox_width_px * bbox_height_px))
    equivalent_diameter_px = float((4.0 * area_px / pi) ** 0.5) if area_px > 0 else 0.0
    orientation_deg, major_axis_px, minor_axis_px = _orientation_from_mask(mask_bool, contours)

    return {
        "area_px": int(area_px),
        "perimeter_px": _round_or_none(perimeter_px, 2),
        "convex_hull_area_px": _round_or_none(convex_hull_area_px, 2),
        "solidity": _safe_ratio(area_px, convex_hull_area_px, digits=4),
        "circularity": _round_or_none((4.0 * pi * area_px) / (perimeter_px * perimeter_px), 4) if perimeter_px > 0 else None,
        "extent": _round_or_none(extent, 4),
        "equivalent_diameter_px": _round_or_none(equivalent_diameter_px, 2),
        "orientation_deg": _round_or_none(orientation_deg, 2) if orientation_deg is not None else None,
        "major_axis_px": _round_or_none(major_axis_px, 2),
        "minor_axis_px": _round_or_none(minor_axis_px, 2),
        "aspect_ratio": _safe_ratio(major_axis_px, minor_axis_px, digits=4),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "bbox_x0": int(bbox_x0),
        "bbox_y0": int(bbox_y0),
        "bbox_x1": int(bbox_x1),
        "bbox_y1": int(bbox_y1),
        "bbox_width_px": int(bbox_width_px),
        "bbox_height_px": int(bbox_height_px),
    }


def _orientation_from_mask(mask_bool: np.ndarray, contours: list[np.ndarray]) -> tuple[float | None, float, float]:
    contour = max(contours, key=cv2.contourArea) if contours else None
    if contour is not None and len(contour) >= 5:
        (_, _), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        major_axis_px = float(max(axis_a, axis_b))
        minor_axis_px = float(min(axis_a, axis_b))
        if axis_a < axis_b:
            angle += 90.0
        return _normalize_angle(float(angle)), major_axis_px, minor_axis_px

    ys, xs = np.where(mask_bool)
    if xs.size < 2:
        return None, 0.0, 0.0

    coords = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major_vec = eigvecs[:, 0]
    angle = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    major_axis_px = float(4.0 * np.sqrt(max(eigvals[0], 0.0)))
    minor_axis_px = float(4.0 * np.sqrt(max(eigvals[1], 0.0))) if eigvals.size > 1 else 0.0
    return _normalize_angle(angle), major_axis_px, minor_axis_px


def _normalize_angle(angle_deg: float) -> float:
    return ((float(angle_deg) + 90.0) % 180.0) - 90.0


def _masked_color_stats(image_bgr: np.ndarray, mask_bool: np.ndarray) -> dict[str, float | None]:
    if image_bgr.size == 0 or mask_bool is None or not np.any(mask_bool):
        return {
            "mean_r": None,
            "mean_g": None,
            "mean_b": None,
            "mean_h": None,
            "mean_s": None,
            "mean_v": None,
            "exg_mean": None,
        }

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels_rgb = rgb[mask_bool]
    pixels_hsv = hsv[mask_bool]

    exg = (2.0 * pixels_rgb[:, 1].astype(np.float64)) - pixels_rgb[:, 0].astype(np.float64) - pixels_rgb[:, 2].astype(np.float64)
    return {
        "mean_r": _round_or_none(np.mean(pixels_rgb[:, 0]), 2),
        "mean_g": _round_or_none(np.mean(pixels_rgb[:, 1]), 2),
        "mean_b": _round_or_none(np.mean(pixels_rgb[:, 2]), 2),
        "mean_h": _round_or_none(np.mean(pixels_hsv[:, 0]), 2),
        "mean_s": _round_or_none(np.mean(pixels_hsv[:, 1]), 2),
        "mean_v": _round_or_none(np.mean(pixels_hsv[:, 2]), 2),
        "exg_mean": _round_or_none(np.mean(exg), 2),
    }


def _safe_ratio(numerator: float, denominator: float, digits: int = 4) -> float | None:
    if denominator is None or denominator == 0:
        return None
    return _round_or_none(float(numerator) / float(denominator), digits)


def _length_px_to_cm(length_px: float | np.floating | None, pixels_per_cm: float | None) -> float | None:
    if length_px is None or pixels_per_cm is None or pixels_per_cm <= 0:
        return None
    return _round_or_none(float(length_px) / float(pixels_per_cm), 4)


def _area_px_to_cm2(area_px: float | np.floating | None, pixels_per_cm: float | None) -> float | None:
    if area_px is None or pixels_per_cm is None or pixels_per_cm <= 0:
        return None
    return _round_or_none(float(area_px) / float(pixels_per_cm * pixels_per_cm), 4)


def _round_or_none(value: float | np.floating | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    if not np.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _draw_region_overlay(image_bgr: np.ndarray, plant_result: PlantLeafResult, color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = plant_result.bbox
    site_x0, site_y0, site_x1, site_y1 = plant_result.site_bbox
    mask = plant_result.mask
    label_map = plant_result.leaf_label_map
    crop = image_bgr[y0:y1, x0:x1]

    if mask.size > 0 and np.any(mask > 0):
        overlay_layer = np.zeros_like(crop)
        overlay_layer[:] = color
        idx = mask > 0
        crop[idx] = cv2.addWeighted(crop[idx], 0.72, overlay_layer[idx], 0.28, 0)

    for leaf_id in [int(value) for value in np.unique(label_map) if int(value) > 0]:
        leaf_mask = ((label_map == leaf_id).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(crop, contours, -1, color, 2)

    for leaf_row in plant_result.leaf_traits:
        centroid_x = leaf_row.get("Centroid X")
        centroid_y = leaf_row.get("Centroid Y")
        leaf_id = leaf_row.get("Leaf ID")
        if centroid_x is None or centroid_y is None or leaf_id is None:
            continue
        cv2.putText(
            image_bgr,
            str(int(leaf_id)),
            (int(round(float(centroid_x))), int(round(float(centroid_y)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    site_center_x = int(round((site_x0 + site_x1) / 2.0))
    site_center_y = int(round((site_y0 + site_y1) / 2.0))
    cv2.circle(image_bgr, (site_center_x, site_center_y), 6, color, -1)

    if np.any(mask > 0):
        caption_x = max(12, x0 + 12)
        caption_y = max(28, y0 + 28)
    else:
        caption_x = max(12, site_center_x - 58)
        caption_y = max(28, site_center_y - 14)

    caption = f"{plant_result.name}: {plant_result.leaf_count}"
    (caption_w, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.rectangle(
        image_bgr,
        (caption_x - 8, caption_y - 18),
        (caption_x + caption_w + 10, caption_y + 8),
        (18, 18, 18),
        -1,
    )
    cv2.putText(
        image_bgr,
        caption,
        (caption_x, caption_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_container_outline(
    image_bgr: np.ndarray,
    tray_bbox: tuple[int, int, int, int],
    container_mask_u8: np.ndarray,
) -> None:
    contours, _ = cv2.findContours((container_mask_u8 > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image_bgr, contours, -1, (255, 255, 255), 2)
        return

    x0, y0, x1, y1 = tray_bbox
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (255, 255, 255), 2)
