from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, pi

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
LETTUCE_2X2_PROFILE_KEY = "lettuce_2x2"
GRID_4X5_PROFILE_KEY = "grid_4x5"
GROWTH_CHAMBER_PROFILE_KEY = "growth_chamber"
CUSTOM_TRAY_PROFILE_KEY = "custom"
SEEDLINGS_PROFILE_KEY = "seedlings"
CONTAINER_MODE_AUTO = "auto"
CONTAINER_MODE_RECTANGLE = "rectangle"
CONTAINER_MODE_CIRCLE = "circle"
CONTAINER_MODE_FULL_IMAGE = "full_image"
ANALYSIS_KIND_FOLIAGE = "foliage"
ANALYSIS_KIND_SEEDLINGS = "seedlings"
ANALYSIS_KIND_GROWTH_CHAMBER = "growth_chamber"
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
    LETTUCE_2X2_PROFILE_KEY: TrayProfile(
        key=LETTUCE_2X2_PROFILE_KEY,
        name="Lettuce (2x2)",
        rows=2,
        cols=2,
        outer_pad_ratio_x=0.045,
        outer_pad_ratio_y=0.045,
        site_pad_ratio_x=0.035,
        site_pad_ratio_y=0.035,
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
    shoot_mask: np.ndarray | None = None
    root_mask: np.ndarray | None = None
    primary_root_mask: np.ndarray | None = None
    lateral_root_mask: np.ndarray | None = None
    root_skeleton_mask: np.ndarray | None = None


@dataclass(frozen=True)
class TrayAnalysisResult:
    image_shape: tuple[int, int]
    overlay_rgb: np.ndarray
    analysis_kind: str
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
        (LETTUCE_2X2_PROFILE_KEY, TRAY_PROFILES[LETTUCE_2X2_PROFILE_KEY].name),
        (GRID_4X5_PROFILE_KEY, TRAY_PROFILES[GRID_4X5_PROFILE_KEY].name),
        (GROWTH_CHAMBER_PROFILE_KEY, "Growth Chamber Level"),
        (SEEDLINGS_PROFILE_KEY, "Seedlings (roots + shoots)"),
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
    if tray_profile_key == GROWTH_CHAMBER_PROFILE_KEY:
        return _analyze_growth_chamber_image(
            image_rgb=image_rgb,
            image_bgr=image_bgr,
            tray_long_side_cm=tray_long_side_cm,
            pixels_per_cm_override=pixels_per_cm_override,
            container_mode=container_mode,
            circular_container_inset_ratio=circular_container_inset_ratio,
            circle_center_x_shift_ratio=circle_center_x_shift_ratio,
            circle_center_y_shift_ratio=circle_center_y_shift_ratio,
            circle_radius_scale=circle_radius_scale,
        )
    if tray_profile_key == SEEDLINGS_PROFILE_KEY:
        return _analyze_seedling_image(
            image_rgb=image_rgb,
            image_bgr=image_bgr,
            tray_long_side_cm=tray_long_side_cm,
            pixels_per_cm_override=pixels_per_cm_override,
            container_mode=container_mode,
            circular_container_inset_ratio=circular_container_inset_ratio,
            circle_center_x_shift_ratio=circle_center_x_shift_ratio,
            circle_center_y_shift_ratio=circle_center_y_shift_ratio,
            circle_radius_scale=circle_radius_scale,
        )
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
    segmentation_source = (
        "PlantCV/OpenCV vegetation threshold (green + anthocyanin)"
        if pcv is not None
        else "OpenCV vegetation threshold (green + anthocyanin)"
    )

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
        analysis_kind=ANALYSIS_KIND_FOLIAGE,
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


def _analyze_growth_chamber_image(
    image_rgb: np.ndarray,
    image_bgr: np.ndarray,
    tray_long_side_cm: float,
    pixels_per_cm_override: float | None,
    container_mode: str,
    circular_container_inset_ratio: float,
    circle_center_x_shift_ratio: float,
    circle_center_y_shift_ratio: float,
    circle_radius_scale: float,
) -> TrayAnalysisResult:
    tray_bbox, tray_long_side_px, container_mask_u8, container_source = _detect_container_geometry(
        image_bgr,
        container_mode=container_mode,
        circular_container_inset_ratio=circular_container_inset_ratio,
        circle_center_x_shift_ratio=circle_center_x_shift_ratio,
        circle_center_y_shift_ratio=circle_center_y_shift_ratio,
        circle_radius_scale=circle_radius_scale,
    )
    tray_x0, tray_y0, tray_x1, tray_y1 = tray_bbox
    chamber_crop_bgr = image_bgr[tray_y0:tray_y1, tray_x0:tray_x1]
    chamber_mask_u8 = _segment_growth_chamber_canopy_mask(chamber_crop_bgr)
    chamber_mask_u8 = cv2.bitwise_and(chamber_mask_u8, container_mask_u8[tray_y0:tray_y1, tray_x0:tray_x1])
    components, component_labels, grid_rows, grid_cols = _extract_growth_chamber_components(chamber_mask_u8)

    pixels_per_cm_override = _normalize_optional_positive_float(pixels_per_cm_override)
    pixels_per_cm, mm_per_pixel, scale_source = _resolve_scale_calibration(
        tray_long_side_px=tray_long_side_px,
        tray_long_side_cm=tray_long_side_cm,
        pixels_per_cm_override=pixels_per_cm_override,
    )

    chamber_profile = TrayProfile(
        key=GROWTH_CHAMBER_PROFILE_KEY,
        name="Growth Chamber Level",
        rows=max(1, grid_rows),
        cols=max(1, grid_cols),
        outer_pad_ratio_x=0.0,
        outer_pad_ratio_y=0.0,
        site_pad_ratio_x=0.0,
        site_pad_ratio_y=0.0,
    )

    overlay_bgr = image_bgr.copy()
    plant_results: list[PlantLeafResult] = []
    plant_rows: list[dict[str, object]] = []
    leaf_rows: list[dict[str, object]] = []

    for idx, component in enumerate(components):
        local_x0, local_y0, local_x1, local_y1 = component["crop_bbox_local"]
        analysis_bbox = (
            tray_x0 + local_x0,
            tray_y0 + local_y0,
            tray_x0 + local_x1,
            tray_y0 + local_y1,
        )
        component_bbox = (
            tray_x0 + component["bbox_local"][0],
            tray_y0 + component["bbox_local"][1],
            tray_x0 + component["bbox_local"][2],
            tray_y0 + component["bbox_local"][3],
        )
        crop_bgr = image_bgr[analysis_bbox[1] : analysis_bbox[3], analysis_bbox[0] : analysis_bbox[2]]
        component_mask_u8 = (
            (component_labels[local_y0:local_y1, local_x0:local_x1] == int(component["label_id"])).astype(np.uint8) * 255
        )
        leaf_label_map, leaf_count, canopy_area_px, min_leaf_area_px = _estimate_leaf_instances(
            component_mask_u8,
            crop_bgr,
            expected_groups=1,
        )
        site = PlantRegion(
            name=f"Plant {idx + 1}",
            position=_format_grid_position(int(component["row_index"]), int(component["col_index"]), chamber_profile.rows, chamber_profile.cols),
            site_index=int(idx + 1),
            row_index=int(component["row_index"]),
            col_index=int(component["col_index"]),
            bbox=component_bbox,
        )
        plant_traits, plant_leaf_rows = _compute_trait_rows(
            crop_bgr=crop_bgr,
            mask_u8=component_mask_u8,
            leaf_label_map=leaf_label_map,
            site=site,
            analysis_bbox=analysis_bbox,
            min_leaf_area_px=min_leaf_area_px,
            tray_profile=chamber_profile,
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
        plant_traits["Analysis Kind"] = ANALYSIS_KIND_GROWTH_CHAMBER
        for leaf_row in plant_leaf_rows:
            leaf_row["Analysis Kind"] = ANALYSIS_KIND_GROWTH_CHAMBER

        plant_result = PlantLeafResult(
            name=site.name,
            position=site.position,
            bbox=analysis_bbox,
            site_bbox=component_bbox,
            leaf_count=int(leaf_count),
            canopy_area_px=int(canopy_area_px),
            min_leaf_area_px=int(min_leaf_area_px),
            mask=component_mask_u8,
            leaf_label_map=leaf_label_map,
            plant_traits=plant_traits,
            leaf_traits=plant_leaf_rows,
        )
        plant_results.append(plant_result)
        plant_rows.append(plant_traits)
        leaf_rows.extend(plant_leaf_rows)
        _draw_chamber_region_overlay(overlay_bgr, plant_result, color=_plant_color_bgr(idx, len(components)))

    _draw_container_outline(overlay_bgr, tray_bbox, container_mask_u8)
    plant_summary_df = pd.DataFrame(plant_rows)
    leaf_detail_df = pd.DataFrame(leaf_rows)
    counts_df = plant_summary_df[[column for column in ["Plant", "Position", "Estimated Leaves", "Canopy Area (px)"] if column in plant_summary_df.columns]].copy()

    return TrayAnalysisResult(
        image_shape=tuple(int(v) for v in image_rgb.shape[:2]),
        overlay_rgb=cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
        analysis_kind=ANALYSIS_KIND_GROWTH_CHAMBER,
        counts_df=counts_df,
        plant_summary_df=plant_summary_df,
        leaf_detail_df=leaf_detail_df,
        tray_bbox=tray_bbox,
        tray_profile_key=GROWTH_CHAMBER_PROFILE_KEY,
        tray_profile_name=chamber_profile.name,
        grid_rows=chamber_profile.rows,
        grid_cols=chamber_profile.cols,
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
        segmentation_source="Growth chamber vegetation threshold",
    )


def _analyze_seedling_image(
    image_rgb: np.ndarray,
    image_bgr: np.ndarray,
    tray_long_side_cm: float,
    pixels_per_cm_override: float | None,
    container_mode: str,
    circular_container_inset_ratio: float,
    circle_center_x_shift_ratio: float,
    circle_center_y_shift_ratio: float,
    circle_radius_scale: float,
) -> TrayAnalysisResult:
    tray_bbox, tray_long_side_px, container_mask_u8, container_source = _detect_container_geometry(
        image_bgr,
        container_mode=container_mode,
        circular_container_inset_ratio=circular_container_inset_ratio,
        circle_center_x_shift_ratio=circle_center_x_shift_ratio,
        circle_center_y_shift_ratio=circle_center_y_shift_ratio,
        circle_radius_scale=circle_radius_scale,
    )
    pixels_per_cm_override = _normalize_optional_positive_float(pixels_per_cm_override)
    pixels_per_cm, mm_per_pixel, scale_source = _resolve_scale_calibration(
        tray_long_side_px=tray_long_side_px,
        tray_long_side_cm=tray_long_side_cm,
        pixels_per_cm_override=pixels_per_cm_override,
    )

    global_shoot_mask_u8 = _segment_seedling_shoot_mask(image_bgr)
    global_shoot_mask_u8 = cv2.bitwise_and(global_shoot_mask_u8, container_mask_u8)
    seedling_regions, grid_rows, grid_cols = _detect_seedling_regions(global_shoot_mask_u8)
    seedling_profile = TrayProfile(
        key=SEEDLINGS_PROFILE_KEY,
        name="Seedlings",
        rows=max(1, grid_rows),
        cols=max(1, grid_cols),
    )

    overlay_bgr = image_bgr.copy()
    plant_results: list[PlantLeafResult] = []
    plant_rows: list[dict[str, object]] = []
    leaf_rows: list[dict[str, object]] = []

    for idx, seedling in enumerate(seedling_regions):
        analysis_bbox = _build_seedling_analysis_bbox(
            seedling=seedling,
            seedling_regions=seedling_regions,
            image_shape=image_bgr.shape[:2],
            tray_bbox=tray_bbox,
        )
        crop_x0, crop_y0, crop_x1, crop_y1 = analysis_bbox
        crop_bgr = image_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
        crop_container_mask_u8 = container_mask_u8[crop_y0:crop_y1, crop_x0:crop_x1]
        crop_shoot_mask_u8 = global_shoot_mask_u8[crop_y0:crop_y1, crop_x0:crop_x1].copy()
        crop_shoot_mask_u8 = cv2.bitwise_and(crop_shoot_mask_u8, crop_container_mask_u8)

        anchor_global_x, anchor_global_y = seedling["anchor"]
        anchor_local = (
            int(anchor_global_x - crop_x0),
            int(anchor_global_y - crop_y0),
        )
        shoot_bbox_global = seedling["bbox"]
        shoot_bbox_local = (
            int(shoot_bbox_global[0] - crop_x0),
            int(shoot_bbox_global[1] - crop_y0),
            int(shoot_bbox_global[2] - crop_x0),
            int(shoot_bbox_global[3] - crop_y0),
        )

        root_support_mask_u8 = _segment_seedling_root_support_mask(
            crop_bgr=crop_bgr,
            crop_shoot_mask_u8=crop_shoot_mask_u8,
            crop_container_mask_u8=crop_container_mask_u8,
            anchor_local=anchor_local,
            shoot_bbox_local=shoot_bbox_local,
        )
        primary_root_score_map = _compute_seedling_primary_root_score_map(
            crop_bgr=crop_bgr,
            crop_shoot_mask_u8=crop_shoot_mask_u8,
            crop_container_mask_u8=crop_container_mask_u8,
            anchor_local=anchor_local,
            shoot_bbox_local=shoot_bbox_local,
        )
        root_parts = _classify_seedling_root_parts(
            root_support_mask_u8,
            anchor_local,
            primary_root_score_map=primary_root_score_map,
        )

        leaf_label_map, leaf_count, canopy_area_px, min_leaf_area_px = _estimate_leaf_instances(
            crop_shoot_mask_u8,
            crop_bgr,
            expected_groups=1,
        )

        site = PlantRegion(
            name=str(seedling["name"]),
            position=str(seedling["position"]),
            site_index=int(seedling["site_index"]),
            row_index=int(seedling["row_index"]),
            col_index=int(seedling["col_index"]),
            bbox=tuple(int(v) for v in seedling["bbox"]),
        )
        plant_traits, plant_leaf_rows = _compute_trait_rows(
            crop_bgr=crop_bgr,
            mask_u8=crop_shoot_mask_u8,
            leaf_label_map=leaf_label_map,
            site=site,
            analysis_bbox=analysis_bbox,
            min_leaf_area_px=min_leaf_area_px,
            tray_profile=seedling_profile,
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
        plant_traits.update(
            _seedling_root_trait_columns(
                root_parts=root_parts,
                pixels_per_cm=pixels_per_cm,
                shoot_area_px=float(plant_traits.get("Canopy Area (px)", 0) or 0.0),
                anchor_global=(anchor_global_x, anchor_global_y),
            )
        )
        plant_traits["Analysis Kind"] = ANALYSIS_KIND_SEEDLINGS
        plant_traits["Shoot Area (px)"] = plant_traits.get("Canopy Area (px)")
        plant_traits["Shoot Area (cm^2)"] = plant_traits.get("Canopy Area (cm^2)")
        plant_traits["Shoot Perimeter (px)"] = plant_traits.get("Perimeter (px)")
        plant_traits["Shoot Perimeter (cm)"] = plant_traits.get("Perimeter (cm)")
        plant_traits["Shoot Solidity"] = plant_traits.get("Solidity")
        plant_traits["Shoot Circularity"] = plant_traits.get("Circularity")
        plant_traits["Shoot Aspect Ratio"] = plant_traits.get("Aspect Ratio")

        for leaf_row in plant_leaf_rows:
            leaf_row["Analysis Kind"] = ANALYSIS_KIND_SEEDLINGS
            leaf_row["Organ"] = "Shoot leaf"

        plant_result = PlantLeafResult(
            name=site.name,
            position=site.position,
            bbox=analysis_bbox,
            site_bbox=site.bbox,
            leaf_count=int(leaf_count),
            canopy_area_px=int(canopy_area_px),
            min_leaf_area_px=int(min_leaf_area_px),
            mask=crop_shoot_mask_u8,
            leaf_label_map=leaf_label_map,
            plant_traits=plant_traits,
            leaf_traits=plant_leaf_rows,
            shoot_mask=crop_shoot_mask_u8,
            root_mask=root_parts["root_mask_u8"],
            primary_root_mask=root_parts["primary_root_mask_u8"],
            lateral_root_mask=root_parts["lateral_root_mask_u8"],
            root_skeleton_mask=root_parts["skeleton_mask_u8"],
        )
        plant_results.append(plant_result)
        plant_rows.append(plant_traits)
        leaf_rows.extend(plant_leaf_rows)
        _draw_seedling_overlay(overlay_bgr, plant_result, color=_plant_color_bgr(idx, len(seedling_regions)))

    _draw_container_outline(overlay_bgr, tray_bbox, container_mask_u8)
    plant_summary_df = pd.DataFrame(plant_rows)
    leaf_detail_df = pd.DataFrame(leaf_rows)
    count_columns = ["Plant", "Position", "Estimated Leaves", "Total Root Length (cm)", "Lateral Root Count"]
    counts_df = plant_summary_df[[column for column in count_columns if column in plant_summary_df.columns]].copy()

    return TrayAnalysisResult(
        image_shape=tuple(int(v) for v in image_rgb.shape[:2]),
        overlay_rgb=cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
        analysis_kind=ANALYSIS_KIND_SEEDLINGS,
        counts_df=counts_df,
        plant_summary_df=plant_summary_df,
        leaf_detail_df=leaf_detail_df,
        tray_bbox=tray_bbox,
        tray_profile_key=SEEDLINGS_PROFILE_KEY,
        tray_profile_name=seedling_profile.name,
        grid_rows=max(0, grid_rows),
        grid_cols=max(0, grid_cols),
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
        segmentation_source="Seedling heuristic (green shoot mask + anchored root skeleton)",
    )


def _segment_seedling_shoot_mask(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.size == 0:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    exg = (2 * rgb[:, :, 1]) - rgb[:, :, 0] - rgb[:, :, 2]
    hsv_mask = cv2.inRange(
        hsv,
        np.array([22, 25, 35], dtype=np.uint8),
        np.array([105, 255, 255], dtype=np.uint8),
    )
    exg_mask = ((exg > 8).astype(np.uint8) * 255)
    mask = cv2.bitwise_and(hsv_mask, exg_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2)
    return mask


def _detect_seedling_regions(shoot_mask_u8: np.ndarray) -> tuple[list[dict[str, object]], int, int]:
    binary = (shoot_mask_u8 > 0).astype(np.uint8)
    if not np.any(binary):
        return [], 0, 0

    total_green_area_px = int(np.count_nonzero(binary))
    min_component_area_px = max(300, int(total_green_area_px * 0.01))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[dict[str, object]] = []
    for label_idx in range(1, num_labels):
        area_px = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area_px < min_component_area_px:
            continue
        x0 = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if width < max(50, int(height * 0.35)):
            continue
        component_mask = labels == label_idx
        ys, xs = np.where(component_mask)
        if xs.size == 0 or ys.size == 0:
            continue
        anchor_band = xs[ys >= (ys.max() - 4)]
        anchor_x = int(np.median(anchor_band)) if anchor_band.size > 0 else int(round(float(centroids[label_idx, 0])))
        anchor_y = int(ys.max())
        components.append(
            {
                "bbox": (x0, y0, x0 + width, y0 + height),
                "centroid_x": float(centroids[label_idx, 0]),
                "centroid_y": float(centroids[label_idx, 1]),
                "anchor": (anchor_x, anchor_y),
                "height": height,
            }
        )

    if not components:
        return [], 0, 0

    components.sort(key=lambda item: (float(item["centroid_y"]), float(item["centroid_x"])))
    median_height = float(np.median([item["height"] for item in components])) if components else 0.0
    row_tolerance_px = max(45.0, median_height * 0.7)
    row_groups: list[dict[str, object]] = []

    for component in components:
        assigned = False
        for row_group in row_groups:
            if abs(float(component["centroid_y"]) - float(row_group["mean_y"])) <= row_tolerance_px:
                row_group["items"].append(component)
                row_group["mean_y"] = float(np.mean([float(item["centroid_y"]) for item in row_group["items"]]))
                assigned = True
                break
        if not assigned:
            row_groups.append({"mean_y": float(component["centroid_y"]), "items": [component]})

    row_groups.sort(key=lambda row_group: float(row_group["mean_y"]))
    grid_rows = len(row_groups)
    grid_cols = max(len(row_group["items"]) for row_group in row_groups)
    seedling_regions: list[dict[str, object]] = []
    site_index = 0
    for row_idx, row_group in enumerate(row_groups):
        row_items = sorted(row_group["items"], key=lambda item: float(item["centroid_x"]))
        for col_idx, component in enumerate(row_items):
            site_index += 1
            component["name"] = f"Seedling {site_index}"
            component["position"] = _format_grid_position(row_idx, col_idx, grid_rows, grid_cols)
            component["site_index"] = site_index
            component["row_index"] = row_idx
            component["col_index"] = col_idx
            seedling_regions.append(component)

    return seedling_regions, grid_rows, grid_cols


def _build_seedling_analysis_bbox(
    seedling: dict[str, object],
    seedling_regions: list[dict[str, object]],
    image_shape: tuple[int, int],
    tray_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    image_h, image_w = image_shape
    tray_x0, tray_y0, tray_x1, tray_y1 = tray_bbox
    shoot_bbox = tuple(int(v) for v in seedling["bbox"])
    x0, y0, x1, y1 = shoot_bbox
    shoot_w = max(1, x1 - x0)
    shoot_h = max(1, y1 - y0)
    pad_top = max(16, int(round(shoot_h * 0.12)))
    horizontal_margin_px = max(18, int(round(shoot_w * 0.16)))

    same_row_seedlings = sorted(
        [item for item in seedling_regions if int(item["row_index"]) == int(seedling["row_index"])],
        key=lambda item: float(item["centroid_x"]),
    )
    current_idx = next(
        (
            idx
            for idx, item in enumerate(same_row_seedlings)
            if int(item["site_index"]) == int(seedling["site_index"])
        ),
        0,
    )
    left_bound = tray_x0
    right_bound = tray_x1
    if current_idx > 0:
        prev_bbox = tuple(int(v) for v in same_row_seedlings[current_idx - 1]["bbox"])
        left_bound = max(tray_x0, int(round((prev_bbox[2] + x0) / 2.0)))
    if current_idx < len(same_row_seedlings) - 1:
        next_bbox = tuple(int(v) for v in same_row_seedlings[current_idx + 1]["bbox"])
        right_bound = min(tray_x1, int(round((x1 + next_bbox[0]) / 2.0)))
    return (
        max(tray_x0, left_bound - horizontal_margin_px),
        max(tray_y0, y0 - pad_top),
        min(tray_x1, right_bound + horizontal_margin_px),
        min(tray_y1, image_h),
    )


def _seedling_root_corridor_geometry(
    crop_shape: tuple[int, int],
    anchor_local: tuple[int, int],
    shoot_bbox_local: tuple[int, int, int, int],
    *,
    dark_background: bool,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, int, int]:
    crop_h, crop_w = crop_shape
    ax = int(np.clip(int(anchor_local[0]), 0, max(0, crop_w - 1)))
    ay = int(np.clip(int(anchor_local[1]), 0, max(0, crop_h - 1)))
    x0, _, x1, _ = shoot_bbox_local
    shoot_width_px = max(1, int(x1 - x0))
    if dark_background:
        top_half_width_px = max(int(round(shoot_width_px * 0.72)), int(round(crop_w * 0.16)), 56)
        bottom_half_width_px = max(int(round(crop_w * 0.03)), 8)
    else:
        top_half_width_px = max(int(round(shoot_width_px * 1.0)), int(round(crop_w * 0.28)), 72)
        bottom_half_width_px = max(int(round(crop_w * 0.06)), 12)

    corridor_start_y = max(0, ay - 8)
    ys = np.arange(crop_h, dtype=np.float32)[:, None]
    xs = np.arange(crop_w, dtype=np.float32)[None, :]
    remaining_height_px = max(1.0, float(crop_h - corridor_start_y - 1))
    down_fraction = np.clip((ys - float(corridor_start_y)) / remaining_height_px, 0.0, 1.0)
    half_width_px = float(top_half_width_px) - ((float(top_half_width_px) - float(bottom_half_width_px)) * down_fraction)
    tapered_corridor = (np.abs(xs - float(ax)) <= half_width_px) & (ys >= float(corridor_start_y))
    return ax, ay, xs, ys, half_width_px.astype(np.float32), top_half_width_px, bottom_half_width_px


def _seedling_has_dark_root_background(
    crop_bgr: np.ndarray,
    crop_shoot_mask_u8: np.ndarray,
    crop_container_mask_u8: np.ndarray,
    anchor_local: tuple[int, int],
) -> bool:
    if crop_bgr.size == 0:
        return False

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    crop_h = crop_bgr.shape[0]
    sample_mask = (crop_container_mask_u8 > 0) & (crop_shoot_mask_u8 == 0)
    sample_mask[: max(0, min(crop_h, int(anchor_local[1]) - 6)), :] = False
    if int(np.count_nonzero(sample_mask)) < 200:
        sample_mask = (crop_container_mask_u8 > 0) & (crop_shoot_mask_u8 == 0)
    if int(np.count_nonzero(sample_mask)) < 200:
        return False

    sat_values = hsv[:, :, 1][sample_mask]
    val_values = hsv[:, :, 2][sample_mask]
    median_sat = float(np.median(sat_values))
    median_val = float(np.median(val_values))
    bright_fraction = float(np.mean(val_values >= 150))
    return median_sat < 55.0 and median_val < 135.0 and bright_fraction < 0.24


def _compute_seedling_dark_root_score_map(
    crop_bgr: np.ndarray,
    crop_shoot_mask_u8: np.ndarray,
    crop_container_mask_u8: np.ndarray,
    anchor_local: tuple[int, int],
    shoot_bbox_local: tuple[int, int, int, int],
) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros(crop_bgr.shape[:2], dtype=np.float32)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 6)
    contrast = cv2.subtract(denoised, blurred).astype(np.float32)
    bright_tophat = cv2.morphologyEx(
        denoised,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    ).astype(np.float32)
    lab_b = lab[:, :, 2].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)

    ax, ay, xs, ys, half_width_px, _, _ = _seedling_root_corridor_geometry(
        crop_bgr.shape[:2],
        anchor_local,
        shoot_bbox_local,
        dark_background=True,
    )
    center_weight = 1.0 - np.clip(np.abs(xs - float(ax)) / np.maximum(half_width_px, 1.0), 0.0, 1.0)
    tapered_corridor = (np.abs(xs - float(ax)) <= half_width_px) & (ys >= float(max(0, ay - 8)))
    score_map = (
        np.maximum(contrast - 4.0, 0.0) * 2.0
        + np.maximum(bright_tophat - 5.0, 0.0) * 1.6
        + np.maximum(val - 112.0, 0.0) * 0.8
        + np.maximum(lab_b - 133.0, 0.0) * 0.8
        - np.maximum(sat - 115.0, 0.0) * 0.35
    )
    score_map[(crop_shoot_mask_u8 > 0) | (crop_container_mask_u8 == 0) | (val < 78.0)] = 0.0
    score_map[~tapered_corridor] = 0.0
    score_map *= (0.58 + (0.42 * center_weight))
    score_map = cv2.GaussianBlur(score_map.astype(np.float32), (0, 0), sigmaX=1.1, sigmaY=2.6)
    score_map[score_map < 0.0] = 0.0
    return score_map.astype(np.float32)


def _segment_seedling_root_support_mask(
    crop_bgr: np.ndarray,
    crop_shoot_mask_u8: np.ndarray,
    crop_container_mask_u8: np.ndarray,
    anchor_local: tuple[int, int],
    shoot_bbox_local: tuple[int, int, int, int],
) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros(crop_bgr.shape[:2], dtype=np.uint8)

    dark_background = _seedling_has_dark_root_background(
        crop_bgr,
        crop_shoot_mask_u8,
        crop_container_mask_u8,
        anchor_local,
    )
    ax, ay, xs, ys, half_width_px, top_half_width_px, bottom_half_width_px = _seedling_root_corridor_geometry(
        crop_bgr.shape[:2],
        anchor_local,
        shoot_bbox_local,
        dark_background=dark_background,
    )

    if dark_background:
        score_map = _compute_seedling_dark_root_score_map(
            crop_bgr,
            crop_shoot_mask_u8,
            crop_container_mask_u8,
            anchor_local,
            shoot_bbox_local,
        )
        positive_scores = score_map[score_map > 0]
        if positive_scores.size == 0:
            return np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
        primary_path = _trace_primary_root_path_from_score(score_map, anchor_local)
        if not primary_path:
            return np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
        lower_threshold = max(14.0, float(np.percentile(positive_scores, 68)))
        candidate = ((score_map >= lower_threshold).astype(np.uint8) * 255)
        primary_skeleton_mask_u8 = np.zeros_like(candidate)
        for x_coord, y_coord in primary_path:
            if 0 <= int(y_coord) < candidate.shape[0] and 0 <= int(x_coord) < candidate.shape[1]:
                primary_skeleton_mask_u8[int(y_coord), int(x_coord)] = 255
        primary_corridor_u8 = cv2.dilate(
            primary_skeleton_mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        upper_path_mask_u8 = np.zeros_like(candidate)
        lateral_span_y = int(anchor_local[1]) + max(120, int(round(candidate.shape[0] * 0.18)))
        for x_coord, y_coord in primary_path:
            if int(y_coord) <= lateral_span_y and 0 <= int(y_coord) < candidate.shape[0] and 0 <= int(x_coord) < candidate.shape[1]:
                upper_path_mask_u8[int(y_coord), int(x_coord)] = 255
        lateral_seed_mask_u8 = upper_path_mask_u8 if np.any(upper_path_mask_u8 > 0) else primary_skeleton_mask_u8
        lateral_corridor_u8 = cv2.dilate(
            lateral_seed_mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 47)),
            iterations=1,
        )
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_or(primary_corridor_u8, lateral_corridor_u8))
        candidate = cv2.bitwise_and(candidate, crop_container_mask_u8)
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(crop_shoot_mask_u8))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
        return candidate
    else:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        blurred = cv2.GaussianBlur(denoised, (0, 0), 7)
        contrast = cv2.subtract(denoised, blurred)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        candidate = ((contrast > 8) & (sat < 128) & (val > 40)).astype(np.uint8) * 255
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(crop_shoot_mask_u8))
        candidate = cv2.bitwise_and(candidate, crop_container_mask_u8)

    candidate = ((candidate > 0) & (np.abs(xs - float(ax)) <= half_width_px) & (ys >= float(max(0, ay - 8)))).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(candidate)

    keep = np.zeros_like(candidate)
    anchor_disk = np.zeros_like(candidate)
    cv2.circle(anchor_disk, (ax, ay), 16, 255, -1)
    max_component_area_px = max(1800 if dark_background else 2400, int(candidate.size * (0.009 if dark_background else 0.012)))

    component_meta: list[dict[str, object]] = []
    for label_idx in range(1, num_labels):
        area_px = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area_px < 6:
            continue
        component_mask = labels == label_idx
        bbox_left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        bbox_top = int(stats[label_idx, cv2.CC_STAT_TOP])
        bbox_width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        bbox_height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        ys_component, xs_component = np.where(component_mask)
        if xs_component.size == 0 or ys_component.size == 0:
            continue
        distance_to_anchor_px = float(
            np.min(((xs_component - ax) ** 2 + (ys_component - ay) ** 2).astype(np.float64)) ** 0.5
        )
        component_meta.append(
            {
                "label_idx": label_idx,
                "mask": component_mask,
                "area_px": area_px,
                "bbox_left": bbox_left,
                "bbox_top": bbox_top,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "distance_to_anchor_px": distance_to_anchor_px,
                "center_x": float(xs_component.mean()),
                "center_y": float(ys_component.mean()),
            }
        )
        if area_px <= max_component_area_px and (
            np.any(anchor_disk[component_mask] > 0) or distance_to_anchor_px <= 14.0
        ):
            keep[component_mask] = 255

    if not np.any(keep > 0):
        return keep

    for _ in range(4):
        dilation_kernel = np.ones((5, 5), dtype=np.uint8) if dark_background else np.ones((7, 7), dtype=np.uint8)
        dilated_keep = cv2.dilate(keep, dilation_kernel, iterations=1)
        changed = False
        for component in component_meta:
            component_mask = component["mask"]
            if np.any(keep[component_mask] > 0):
                continue
            area_px = int(component["area_px"])
            bbox_width = int(component["bbox_width"])
            bbox_height = int(component["bbox_height"])
            center_y = float(component["center_y"])
            center_row = int(np.clip(int(round(center_y)), 0, max(0, candidate.shape[0] - 1)))
            half_width_at_row_px = float(half_width_px[center_row, 0])
            allowed_center_drift_px = float(
                max(bottom_half_width_px * 2, half_width_at_row_px * (1.15 if dark_background else 1.35))
            )
            if area_px > max_component_area_px:
                continue
            if abs(float(component["center_x"]) - float(ax)) > allowed_center_drift_px:
                continue
            if bbox_height < 4:
                continue
            if bbox_width > max(72 if dark_background else 90, int(candidate.shape[1] * (0.14 if dark_background else 0.22))) and bbox_height < bbox_width:
                continue
            if np.any(dilated_keep[component_mask] > 0):
                keep[component_mask] = 255
                changed = True
        if not changed:
            break

    keep = cv2.bitwise_and(keep, crop_container_mask_u8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    if dark_background:
        keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    return keep


def _compute_seedling_primary_root_score_map(
    crop_bgr: np.ndarray,
    crop_shoot_mask_u8: np.ndarray,
    crop_container_mask_u8: np.ndarray,
    anchor_local: tuple[int, int],
    shoot_bbox_local: tuple[int, int, int, int],
) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros(crop_bgr.shape[:2], dtype=np.float32)

    dark_background = _seedling_has_dark_root_background(
        crop_bgr,
        crop_shoot_mask_u8,
        crop_container_mask_u8,
        anchor_local,
    )
    if dark_background:
        return _compute_seedling_dark_root_score_map(
            crop_bgr,
            crop_shoot_mask_u8,
            crop_container_mask_u8,
            anchor_local,
            shoot_bbox_local,
        )

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 7)
    contrast = cv2.subtract(denoised, blurred).astype(np.float32)
    bright_tophat = cv2.morphologyEx(
        denoised,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    ).astype(np.float32)
    warm_tophat = cv2.morphologyEx(
        lab[:, :, 2],
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    ).astype(np.float32)

    score_map = (
        np.maximum(contrast - 3.0, 0.0) * 1.3
        + np.maximum(bright_tophat - 4.0, 0.0)
        + np.maximum(warm_tophat - 3.0, 0.0) * 0.7
    )

    val = hsv[:, :, 2]
    score_map[(crop_shoot_mask_u8 > 0) | (crop_container_mask_u8 == 0) | (val <= 30)] = 0.0

    ax, ay, xs, ys, half_width_px, _, _ = _seedling_root_corridor_geometry(
        crop_bgr.shape[:2],
        anchor_local,
        shoot_bbox_local,
        dark_background=False,
    )
    tapered_corridor = (np.abs(xs - float(ax)) <= half_width_px) & (ys >= float(max(0, ay - 8)))
    score_map[~tapered_corridor] = 0.0
    return score_map


def _classify_seedling_root_parts(
    root_support_mask_u8: np.ndarray,
    anchor_local: tuple[int, int],
    primary_root_score_map: np.ndarray | None = None,
) -> dict[str, object]:
    empty_mask_u8 = np.zeros_like(root_support_mask_u8, dtype=np.uint8)
    if root_support_mask_u8.size == 0 or not np.any(root_support_mask_u8 > 0):
        return {
            "root_mask_u8": empty_mask_u8,
            "primary_root_mask_u8": empty_mask_u8,
            "lateral_root_mask_u8": empty_mask_u8,
            "skeleton_mask_u8": empty_mask_u8,
            "primary_skeleton_mask_u8": empty_mask_u8,
            "lateral_skeleton_mask_u8": empty_mask_u8,
            "total_root_length_px": 0.0,
            "primary_root_length_px": 0.0,
            "lateral_root_length_px": 0.0,
            "endpoint_count": 0,
            "lateral_root_count": 0,
            "avg_root_thickness_px": None,
        }

    support_mask_u8 = cv2.morphologyEx(root_support_mask_u8, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    support_mask_u8 = cv2.morphologyEx(support_mask_u8, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    skeleton_mask_u8 = _skeletonize_mask(support_mask_u8)
    start_xy = _nearest_nonzero_point(skeleton_mask_u8, anchor_local)
    connected_skeleton_mask_u8 = (
        _connected_skeleton_component_from_start(skeleton_mask_u8, start_xy)
        if start_xy is not None
        else np.zeros_like(skeleton_mask_u8)
    )
    skeleton_primary_path = _trace_primary_root_path(connected_skeleton_mask_u8, start_xy) if start_xy is not None else []
    score_primary_path = _trace_primary_root_path_from_score(primary_root_score_map, anchor_local)
    if score_primary_path:
        anchor_x = int(anchor_local[0])
        endpoint_deviation_px = abs(int(score_primary_path[-1][0]) - anchor_x)
        mean_lateral_deviation_px = float(
            np.mean([abs(int(x_coord) - anchor_x) for x_coord, _ in score_primary_path])
        )
        if endpoint_deviation_px > max(56, int(round(root_support_mask_u8.shape[1] * 0.16))) or mean_lateral_deviation_px > max(34.0, float(root_support_mask_u8.shape[1] * 0.18)):
            score_primary_path = []
    primary_path = (
        score_primary_path
        if _polyline_length_px(score_primary_path) > (_polyline_length_px(skeleton_primary_path) * 1.05)
        else skeleton_primary_path
    )
    primary_skeleton_mask_u8 = np.zeros_like(support_mask_u8)
    for x_coord, y_coord in primary_path:
        if 0 <= int(y_coord) < primary_skeleton_mask_u8.shape[0] and 0 <= int(x_coord) < primary_skeleton_mask_u8.shape[1]:
            primary_skeleton_mask_u8[int(y_coord), int(x_coord)] = 255

    remaining_skeleton_mask_u8 = cv2.bitwise_and(
        connected_skeleton_mask_u8,
        cv2.bitwise_not(primary_skeleton_mask_u8),
    )
    lateral_skeleton_mask_u8, lateral_root_count, lateral_root_length_px = _filter_lateral_root_skeleton(
        remaining_skeleton_mask_u8,
        min_length_px=max(8.0, _polyline_length_px(primary_path) * 0.01),
    )
    total_skeleton_mask_u8 = cv2.bitwise_or(primary_skeleton_mask_u8, lateral_skeleton_mask_u8)

    primary_support_mask_u8 = _binary_root_mask_from_score_map(primary_root_score_map)
    if not np.any(primary_support_mask_u8 > 0):
        primary_support_mask_u8 = support_mask_u8.copy()
    primary_root_mask_u8 = _refine_root_mask_from_skeleton(
        primary_support_mask_u8,
        primary_skeleton_mask_u8,
        radius_px=6,
    )
    lateral_root_mask_u8 = (
        _refine_root_mask_from_skeleton(support_mask_u8, lateral_skeleton_mask_u8, radius_px=3)
        if np.any(lateral_skeleton_mask_u8 > 0)
        else np.zeros_like(support_mask_u8)
    )
    root_mask_u8 = cv2.bitwise_or(primary_root_mask_u8, lateral_root_mask_u8)
    if not np.any(root_mask_u8 > 0):
        root_mask_u8 = support_mask_u8.copy()

    primary_root_length_px = _polyline_length_px(primary_path)
    total_root_length_px = primary_root_length_px + lateral_root_length_px
    endpoint_count = _count_skeleton_endpoints(total_skeleton_mask_u8)
    avg_root_thickness_px = _safe_ratio(np.count_nonzero(root_mask_u8), total_root_length_px, digits=4)

    return {
        "root_mask_u8": root_mask_u8,
        "primary_root_mask_u8": primary_root_mask_u8,
        "lateral_root_mask_u8": lateral_root_mask_u8,
        "skeleton_mask_u8": total_skeleton_mask_u8,
        "primary_skeleton_mask_u8": primary_skeleton_mask_u8,
        "lateral_skeleton_mask_u8": lateral_skeleton_mask_u8,
        "total_root_length_px": _round_or_none(total_root_length_px, 2) or 0.0,
        "primary_root_length_px": _round_or_none(primary_root_length_px, 2) or 0.0,
        "lateral_root_length_px": _round_or_none(lateral_root_length_px, 2) or 0.0,
        "endpoint_count": int(endpoint_count),
        "lateral_root_count": int(lateral_root_count),
        "avg_root_thickness_px": avg_root_thickness_px,
    }


def _seedling_root_trait_columns(
    root_parts: dict[str, object],
    pixels_per_cm: float | None,
    shoot_area_px: float,
    anchor_global: tuple[int, int],
) -> dict[str, object]:
    root_mask_u8 = root_parts["root_mask_u8"]
    primary_root_mask_u8 = root_parts["primary_root_mask_u8"]
    lateral_root_mask_u8 = root_parts["lateral_root_mask_u8"]

    root_stats = _shape_stats_from_mask(root_mask_u8 > 0)
    primary_root_stats = _shape_stats_from_mask(primary_root_mask_u8 > 0)
    lateral_root_stats = _shape_stats_from_mask(lateral_root_mask_u8 > 0)
    total_root_length_px = float(root_parts["total_root_length_px"])
    primary_root_length_px = float(root_parts["primary_root_length_px"])
    lateral_root_length_px = float(root_parts["lateral_root_length_px"])

    return {
        "Seedling Anchor X": int(anchor_global[0]),
        "Seedling Anchor Y": int(anchor_global[1]),
        "Root Area (px)": int(root_stats["area_px"]),
        "Root Area (cm^2)": _area_px_to_cm2(root_stats["area_px"], pixels_per_cm),
        "Primary Root Area (px)": int(primary_root_stats["area_px"]),
        "Primary Root Area (cm^2)": _area_px_to_cm2(primary_root_stats["area_px"], pixels_per_cm),
        "Lateral Root Area (px)": int(lateral_root_stats["area_px"]),
        "Lateral Root Area (cm^2)": _area_px_to_cm2(lateral_root_stats["area_px"], pixels_per_cm),
        "Total Root Length (px)": _round_or_none(total_root_length_px, 2),
        "Total Root Length (cm)": _length_px_to_cm(total_root_length_px, pixels_per_cm),
        "Primary Root Length (px)": _round_or_none(primary_root_length_px, 2),
        "Primary Root Length (cm)": _length_px_to_cm(primary_root_length_px, pixels_per_cm),
        "Lateral Root Length (px)": _round_or_none(lateral_root_length_px, 2),
        "Lateral Root Length (cm)": _length_px_to_cm(lateral_root_length_px, pixels_per_cm),
        "Root Endpoint Count": int(root_parts["endpoint_count"]),
        "Lateral Root Count": int(root_parts["lateral_root_count"]),
        "Avg Root Thickness (px)": root_parts["avg_root_thickness_px"],
        "Avg Root Thickness (cm)": _length_px_to_cm(root_parts["avg_root_thickness_px"], pixels_per_cm),
        "Root:Shoot Area Ratio": _safe_ratio(root_stats["area_px"], shoot_area_px, digits=4),
        "Primary:Total Root Length Ratio": _safe_ratio(primary_root_length_px, total_root_length_px, digits=4),
    }


def _skeletonize_mask(mask_u8: np.ndarray) -> np.ndarray:
    working_mask = (mask_u8 > 0).astype(np.uint8) * 255
    skeleton_mask = np.zeros_like(working_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(working_mask, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(working_mask, temp)
        skeleton_mask = cv2.bitwise_or(skeleton_mask, temp)
        working_mask = eroded
        if cv2.countNonZero(working_mask) == 0:
            break

    return skeleton_mask


def _nearest_nonzero_point(mask_u8: np.ndarray, anchor_xy: tuple[int, int]) -> tuple[int, int] | None:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    anchor_x, anchor_y = int(anchor_xy[0]), int(anchor_xy[1])
    distances_sq = ((xs - anchor_x) ** 2 + (ys - anchor_y) ** 2).astype(np.float64)
    nearest_idx = int(np.argmin(distances_sq))
    return int(xs[nearest_idx]), int(ys[nearest_idx])


def _connected_skeleton_component_from_start(
    skeleton_mask_u8: np.ndarray,
    start_xy: tuple[int, int],
) -> np.ndarray:
    binary = (skeleton_mask_u8 > 0).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(skeleton_mask_u8)

    start_x, start_y = int(start_xy[0]), int(start_xy[1])
    label_at_start = int(labels[start_y, start_x]) if 0 <= start_y < labels.shape[0] and 0 <= start_x < labels.shape[1] else 0
    if label_at_start <= 0:
        return np.zeros_like(skeleton_mask_u8)
    return ((labels == label_at_start).astype(np.uint8) * 255)


def _trace_primary_root_path(
    skeleton_mask_u8: np.ndarray,
    start_xy: tuple[int, int],
) -> list[tuple[int, int]]:
    ys, xs = np.where(skeleton_mask_u8 > 0)
    if xs.size == 0 or ys.size == 0:
        return []

    coordinates = list(zip(xs.tolist(), ys.tolist()))
    coordinate_to_index = {coordinate: idx for idx, coordinate in enumerate(coordinates)}
    neighbors: list[list[int]] = [[] for _ in coordinates]
    unique_neighbor_offsets = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),            (1, 0),
        (-1, 1),  (0, 1),   (1, 1),
    )
    for idx, (x_coord, y_coord) in enumerate(coordinates):
        for dx, dy in unique_neighbor_offsets:
            neighbor_idx = coordinate_to_index.get((x_coord + dx, y_coord + dy))
            if neighbor_idx is not None:
                neighbors[idx].append(neighbor_idx)

    start_index = coordinate_to_index.get((int(start_xy[0]), int(start_xy[1])))
    if start_index is None:
        start_index = int(
            np.argmin(
                [
                    (x_coord - int(start_xy[0])) ** 2 + (y_coord - int(start_xy[1])) ** 2
                    for x_coord, y_coord in coordinates
                ]
            )
        )

    parent_by_index = {start_index: -1}
    queue = deque([start_index])
    while queue:
        current_idx = queue.popleft()
        for neighbor_idx in neighbors[current_idx]:
            if neighbor_idx in parent_by_index:
                continue
            parent_by_index[neighbor_idx] = current_idx
            queue.append(neighbor_idx)

    start_x, start_y = coordinates[start_index]
    endpoint_candidates: list[tuple[float, int]] = []
    for idx in parent_by_index:
        neighbor_count = sum(1 for neighbor_idx in neighbors[idx] if neighbor_idx in parent_by_index)
        if idx != start_index and neighbor_count <= 1:
            x_coord, y_coord = coordinates[idx]
            score = (y_coord - start_y) - (0.35 * abs(x_coord - start_x))
            endpoint_candidates.append((float(score), idx))

    if endpoint_candidates:
        _, end_index = max(endpoint_candidates, key=lambda item: item[0])
    else:
        end_index = max(parent_by_index, key=lambda idx: coordinates[idx][1])

    path: list[tuple[int, int]] = []
    current_idx = end_index
    while current_idx != -1:
        path.append(coordinates[current_idx])
        current_idx = parent_by_index[current_idx]
    path.reverse()
    return path


def _trace_primary_root_path_from_score(
    score_map: np.ndarray | None,
    anchor_xy: tuple[int, int],
) -> list[tuple[int, int]]:
    if score_map is None or score_map.size == 0:
        return []

    map_h, map_w = score_map.shape[:2]
    anchor_x = int(np.clip(int(anchor_xy[0]), 0, max(0, map_w - 1)))
    anchor_y = int(np.clip(int(anchor_xy[1]), 0, max(0, map_h - 1)))
    current_x = anchor_x
    gap_rows = 0
    search_radius_px = max(12, int(round(map_w * 0.025)))
    max_gap_rows = 35
    min_signal_score = 12.0
    max_step_px = 6
    path: list[tuple[int, int]] = []
    started = False

    for y_coord in range(max(0, anchor_y - 3), map_h):
        x0 = max(0, current_x - search_radius_px)
        x1 = min(map_w, current_x + search_radius_px + 1)
        if x1 <= x0:
            break
        window = score_map[max(0, y_coord - 1): min(map_h, y_coord + 2), x0:x1].sum(axis=0)
        if window.size == 0:
            continue
        offsets = np.arange(x0, x1, dtype=np.int32)
        penalized_scores = window - (np.abs(offsets - current_x).astype(np.float32) * 0.9)
        best_idx = int(np.argmax(penalized_scores))
        best_x = int(offsets[best_idx])
        raw_score = float(window[best_idx])
        if raw_score < min_signal_score:
            gap_rows += 1
            if started and gap_rows > max_gap_rows:
                break
            continue
        started = True
        gap_rows = 0
        current_x = int(np.clip(best_x, current_x - max_step_px, current_x + max_step_px))
        path.append((current_x, y_coord))

    if len(path) <= 2:
        return path

    x_values = np.array([x_coord for x_coord, _ in path], dtype=np.float32)
    smoothed_x_values = x_values.copy()
    window_radius = 61
    for idx in range(len(path)):
        start_idx = max(0, idx - window_radius)
        end_idx = min(len(path), idx + window_radius + 1)
        smoothed_x_values[idx] = float(np.median(x_values[start_idx:end_idx]))
    return [(int(round(smoothed_x_values[idx])), int(y_coord)) for idx, (_, y_coord) in enumerate(path)]


def _binary_root_mask_from_score_map(score_map: np.ndarray | None) -> np.ndarray:
    if score_map is None or score_map.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if not np.any(score_map > 0):
        return np.zeros(score_map.shape[:2], dtype=np.uint8)
    threshold = max(18.0, float(np.percentile(score_map[score_map > 0], 93)))
    binary_mask_u8 = ((score_map >= threshold).astype(np.uint8) * 255)
    binary_mask_u8 = cv2.morphologyEx(binary_mask_u8, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    binary_mask_u8 = cv2.morphologyEx(binary_mask_u8, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8), iterations=1)
    return binary_mask_u8


def _filter_lateral_root_skeleton(
    remaining_skeleton_mask_u8: np.ndarray,
    min_length_px: float,
) -> tuple[np.ndarray, int, float]:
    binary = (remaining_skeleton_mask_u8 > 0).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(remaining_skeleton_mask_u8), 0, 0.0

    lateral_mask_u8 = np.zeros_like(remaining_skeleton_mask_u8)
    lateral_root_count = 0
    lateral_root_length_px = 0.0
    for label_idx in range(1, num_labels):
        component_mask_u8 = ((labels == label_idx).astype(np.uint8) * 255)
        component_length_px = _skeleton_length_px(component_mask_u8)
        if component_length_px < float(min_length_px):
            continue
        lateral_mask_u8 = cv2.bitwise_or(lateral_mask_u8, component_mask_u8)
        lateral_root_count += 1
        lateral_root_length_px += component_length_px

    return lateral_mask_u8, lateral_root_count, lateral_root_length_px


def _refine_root_mask_from_skeleton(
    support_mask_u8: np.ndarray,
    skeleton_mask_u8: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    if not np.any(skeleton_mask_u8 > 0):
        return np.zeros_like(support_mask_u8)

    diameter_px = max(3, (2 * int(radius_px)) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter_px, diameter_px))
    corridor_mask_u8 = cv2.dilate(skeleton_mask_u8, kernel, iterations=1)
    refined_mask_u8 = cv2.bitwise_and(support_mask_u8, corridor_mask_u8)
    if np.count_nonzero(refined_mask_u8) < np.count_nonzero(skeleton_mask_u8):
        refined_mask_u8 = corridor_mask_u8
    return refined_mask_u8


def _polyline_length_px(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    length_px = 0.0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        length_px += hypot(float(x1 - x0), float(y1 - y0))
    return length_px


def _skeleton_length_px(skeleton_mask_u8: np.ndarray) -> float:
    ys, xs = np.where(skeleton_mask_u8 > 0)
    if xs.size == 0 or ys.size == 0:
        return 0.0

    coordinates = {(int(x_coord), int(y_coord)) for x_coord, y_coord in zip(xs, ys)}
    unique_offsets = ((1, 0), (0, 1), (1, 1), (1, -1))
    length_px = 0.0
    for x_coord, y_coord in coordinates:
        for dx, dy in unique_offsets:
            if (x_coord + dx, y_coord + dy) in coordinates:
                length_px += hypot(float(dx), float(dy))
    return float(length_px)


def _count_skeleton_endpoints(skeleton_mask_u8: np.ndarray) -> int:
    binary = skeleton_mask_u8 > 0
    if not np.any(binary):
        return 0

    endpoint_count = 0
    ys, xs = np.where(binary)
    for x_coord, y_coord in zip(xs, ys):
        neighborhood = binary[
            max(0, y_coord - 1): min(binary.shape[0], y_coord + 2),
            max(0, x_coord - 1): min(binary.shape[1], x_coord + 2),
        ]
        neighbor_count = int(np.count_nonzero(neighborhood)) - 1
        if neighbor_count == 1:
            endpoint_count += 1
    return endpoint_count


def _normalize_rgb_image(image_rgb: np.ndarray) -> np.ndarray:
    if not isinstance(image_rgb, np.ndarray):
        image_rgb = np.asarray(image_rgb)
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


def _segment_growth_chamber_canopy_mask(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros(crop_bgr.shape[:2], dtype=np.uint8)

    working_bgr, scale = _downscale_for_analysis(crop_bgr, MAX_PLANT_ANALYSIS_DIM)
    hsv = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2LAB)
    rgb = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    exg = (2 * rgb[:, :, 1]) - rgb[:, :, 0] - rgb[:, :, 2]
    red_minus_blue = rgb[:, :, 0] - rgb[:, :, 2]
    green_minus_blue = rgb[:, :, 1] - rgb[:, :, 2]

    hsv_mask = cv2.inRange(
        hsv,
        np.array([25, 35, 80], dtype=np.uint8),
        np.array([60, 255, 255], dtype=np.uint8),
    )
    exg_mask = ((exg >= 18).astype(np.uint8) * 255)
    red_blue_mask = ((red_minus_blue >= 0).astype(np.uint8) * 255)
    green_blue_mask = ((green_minus_blue >= 12).astype(np.uint8) * 255)
    lab_b_mask = ((lab[:, :, 2] >= 138).astype(np.uint8) * 255)

    mask = cv2.bitwise_and(hsv_mask, exg_mask)
    mask = cv2.bitwise_and(mask, red_blue_mask)
    mask = cv2.bitwise_and(mask, green_blue_mask)
    mask = cv2.bitwise_and(mask, lab_b_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
    if scale != 1.0:
        mask = cv2.resize(mask, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _green_vegetation_mask(hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
    hsv_mask = cv2.inRange(
        hsv,
        np.array([30, 35, 20], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    lab_mask = cv2.inRange(lab[:, :, 1], 0, 124)
    return cv2.bitwise_and(hsv_mask, lab_mask)


def _anthocyanin_vegetation_mask(working_bgr: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
    bgr_i16 = working_bgr.astype(np.int16)
    rgb_i16 = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    blue = bgr_i16[:, :, 0]
    green = bgr_i16[:, :, 1]
    red = bgr_i16[:, :, 2]
    exg = (2 * rgb_i16[:, :, 1]) - rgb_i16[:, :, 0] - rgb_i16[:, :, 2]

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    hue_mask = (((hue <= 25) | (hue >= 150)).astype(np.uint8) * 255)
    sat_mask = ((sat >= 30).astype(np.uint8) * 255)
    val_mask = ((val >= 20).astype(np.uint8) * 255)
    a_mask = ((a_channel >= 126).astype(np.uint8) * 255)
    b_mask = ((b_channel >= 108).astype(np.uint8) * 255)
    not_blue_mask = (((blue <= green + 15) | (blue <= red + 15)).astype(np.uint8) * 255)
    vegetation_bias_mask = (((exg >= -22) | ((red - blue) >= -12)).astype(np.uint8) * 255)

    mask = cv2.bitwise_and(hue_mask, sat_mask)
    mask = cv2.bitwise_and(mask, val_mask)
    mask = cv2.bitwise_and(mask, a_mask)
    mask = cv2.bitwise_and(mask, b_mask)
    mask = cv2.bitwise_and(mask, not_blue_mask)
    mask = cv2.bitwise_and(mask, vegetation_bias_mask)
    return mask


def _cleanup_vegetation_mask(mask: np.ndarray) -> np.ndarray:
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
    return mask


def _segment_with_opencv(crop_bgr: np.ndarray) -> np.ndarray:
    working_bgr, scale = _downscale_for_analysis(crop_bgr, MAX_PLANT_ANALYSIS_DIM)
    hsv = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2LAB)
    green_mask = _green_vegetation_mask(hsv, lab)
    purple_mask = _anthocyanin_vegetation_mask(working_bgr, hsv, lab)
    mask = cv2.bitwise_or(green_mask, purple_mask)
    mask = _cleanup_vegetation_mask(mask)
    if scale != 1.0:
        mask = cv2.resize(mask, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _segment_with_plantcv(crop_bgr: np.ndarray) -> np.ndarray:
    working_bgr, scale = _downscale_for_analysis(crop_bgr, MAX_PLANT_ANALYSIS_DIM)
    crop_rgb = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2LAB)
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
    green_mask = cv2.bitwise_and(hsv_mask, lab_mask)
    purple_mask = _anthocyanin_vegetation_mask(working_bgr, hsv, lab)
    mask = cv2.bitwise_or(green_mask, purple_mask)

    try:
        mask = pcv.fill(bin_img=mask, size=max(64, mask.size // 400))
    except TypeError:
        mask = pcv.fill(mask, max(64, mask.size // 400))

    try:
        mask = pcv.fill_holes(mask)
    except Exception:
        pass

    mask = _cleanup_vegetation_mask(mask)
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


def _extract_growth_chamber_components(
    mask_u8: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray, int, int]:
    binary = (mask_u8 > 0).astype(np.uint8)
    if not np.any(binary):
        return [], np.zeros(mask_u8.shape, dtype=np.int32), 0, 0

    min_component_area_px = max(24, int(mask_u8.size * 0.0000025))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[dict[str, object]] = []
    for label_idx in range(1, num_labels):
        area_px = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area_px < min_component_area_px:
            continue
        x0 = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        pad_x = max(6, int(round(width * 0.35)))
        pad_y = max(6, int(round(height * 0.35)))
        crop_bbox_local = (
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(mask_u8.shape[1], x0 + width + pad_x),
            min(mask_u8.shape[0], y0 + height + pad_y),
        )
        components.append(
            {
                "label_id": int(label_idx),
                "bbox_local": (x0, y0, x0 + width, y0 + height),
                "crop_bbox_local": crop_bbox_local,
                "centroid_x": float(centroids[label_idx, 0]),
                "centroid_y": float(centroids[label_idx, 1]),
                "height_px": int(height),
                "area_px": int(area_px),
            }
        )

    if not components:
        return [], labels.astype(np.int32), 0, 0

    grid_rows, grid_cols = _assign_growth_chamber_grid_indices(components)
    components.sort(key=lambda component: (int(component["row_index"]), int(component["col_index"]), float(component["centroid_x"])))
    return components, labels.astype(np.int32), int(grid_rows), int(grid_cols)


def _assign_growth_chamber_grid_indices(components: list[dict[str, object]]) -> tuple[int, int]:
    if not components:
        return 0, 0

    median_height = float(np.median([float(component["height_px"]) for component in components]))
    row_tolerance_px = max(26.0, 0.85 * median_height)
    row_groups: list[dict[str, object]] = []

    for component in sorted(components, key=lambda item: float(item["centroid_y"])):
        centroid_y = float(component["centroid_y"])
        best_row_idx = -1
        best_distance = float("inf")
        for row_idx, row in enumerate(row_groups):
            distance = abs(centroid_y - float(row["center_y"]))
            if distance <= row_tolerance_px and distance < best_distance:
                best_row_idx = row_idx
                best_distance = distance
        if best_row_idx < 0:
            row_groups.append({"center_y": centroid_y, "members": [component]})
        else:
            row = row_groups[best_row_idx]
            row["members"].append(component)
            row["center_y"] = float(np.mean([float(member["centroid_y"]) for member in row["members"]]))

    row_groups.sort(key=lambda row: float(row["center_y"]))
    max_cols = 0
    for row_idx, row in enumerate(row_groups):
        members = sorted(row["members"], key=lambda item: float(item["centroid_x"]))
        max_cols = max(max_cols, len(members))
        for col_idx, component in enumerate(members):
            component["row_index"] = int(row_idx)
            component["col_index"] = int(col_idx)

    return len(row_groups), max_cols


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
        mask_ys, mask_xs = np.where(mask > 0)
        caption_x = max(12, x0 + int(mask_xs.min()) + 12)
        caption_y = max(28, y0 + int(mask_ys.min()) + 28)
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


def _draw_chamber_region_overlay(image_bgr: np.ndarray, plant_result: PlantLeafResult, color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = plant_result.bbox
    crop = image_bgr[y0:y1, x0:x1]
    mask = plant_result.mask

    if mask.size > 0 and np.any(mask > 0):
        overlay_layer = np.zeros_like(crop)
        overlay_layer[:] = color
        idx = mask > 0
        crop[idx] = cv2.addWeighted(crop[idx], 0.68, overlay_layer[idx], 0.32, 0)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(crop, contours, -1, color, 2)

    site_x0, site_y0, site_x1, site_y1 = plant_result.site_bbox
    center_x = int(round((site_x0 + site_x1) / 2.0))
    center_y = int(round((site_y0 + site_y1) / 2.0))
    caption = plant_result.name.replace("Plant ", "P")
    (caption_w, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    caption_x = max(8, center_x - (caption_w // 2))
    caption_y = max(18, y0 + 18)
    cv2.rectangle(
        image_bgr,
        (caption_x - 5, caption_y - 14),
        (caption_x + caption_w + 5, caption_y + 4),
        (18, 18, 18),
        -1,
    )
    cv2.putText(
        image_bgr,
        caption,
        (caption_x, caption_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_seedling_overlay(image_bgr: np.ndarray, plant_result: PlantLeafResult, color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = plant_result.bbox
    crop = image_bgr[y0:y1, x0:x1]
    shoot_mask = plant_result.shoot_mask if plant_result.shoot_mask is not None else plant_result.mask
    root_mask = plant_result.root_mask
    primary_root_mask = plant_result.primary_root_mask
    lateral_root_mask = plant_result.lateral_root_mask
    root_skeleton_mask = plant_result.root_skeleton_mask

    if shoot_mask is not None and np.any(shoot_mask > 0):
        shoot_layer = np.zeros_like(crop)
        shoot_layer[:] = color
        crop[shoot_mask > 0] = cv2.addWeighted(crop[shoot_mask > 0], 0.68, shoot_layer[shoot_mask > 0], 0.32, 0)

    if root_mask is not None and np.any(root_mask > 0):
        root_layer = np.zeros_like(crop)
        root_layer[:] = (80, 180, 255)
        crop[root_mask > 0] = cv2.addWeighted(crop[root_mask > 0], 0.75, root_layer[root_mask > 0], 0.25, 0)

    if primary_root_mask is not None and np.any(primary_root_mask > 0):
        primary_layer = np.zeros_like(crop)
        primary_layer[:] = (255, 220, 90)
        crop[primary_root_mask > 0] = cv2.addWeighted(crop[primary_root_mask > 0], 0.6, primary_layer[primary_root_mask > 0], 0.4, 0)

    if lateral_root_mask is not None and np.any(lateral_root_mask > 0):
        lateral_layer = np.zeros_like(crop)
        lateral_layer[:] = (214, 114, 255)
        crop[lateral_root_mask > 0] = cv2.addWeighted(crop[lateral_root_mask > 0], 0.72, lateral_layer[lateral_root_mask > 0], 0.28, 0)

    if root_skeleton_mask is not None and np.any(root_skeleton_mask > 0):
        crop[root_skeleton_mask > 0] = (255, 255, 255)

    label_map = plant_result.leaf_label_map
    for leaf_id in [int(value) for value in np.unique(label_map) if int(value) > 0]:
        leaf_mask = ((label_map == leaf_id).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(crop, contours, -1, color, 2)

    site_x0, site_y0, site_x1, site_y1 = plant_result.site_bbox
    anchor_x = int(round((site_x0 + site_x1) / 2.0))
    anchor_y = int(site_y1)
    cv2.circle(image_bgr, (anchor_x, anchor_y), 6, color, -1)

    caption_x = max(12, x0 + 12)
    caption_y = max(28, y0 + 28)
    root_length_cm = plant_result.plant_traits.get("Total Root Length (cm)")
    lateral_root_count = plant_result.plant_traits.get("Lateral Root Count")
    root_suffix = ""
    if root_length_cm is not None:
        root_suffix = f" | {root_length_cm:.2f} cm"
    elif lateral_root_count is not None:
        root_suffix = f" | LR {int(lateral_root_count)}"
    caption = f"{plant_result.name}: {plant_result.leaf_count}{root_suffix}"
    (caption_w, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
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
        0.56,
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
