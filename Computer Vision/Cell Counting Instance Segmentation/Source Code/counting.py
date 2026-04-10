"""Cell Counting Instance Segmentation — counting post-processing.

Handles touching-cell separation (watershed), small-object filtering, and
overlapping-mask merging.  All logic is modular and configurable.
"""

from __future__ import annotations

import cv2
import numpy as np

from segmentation import CellInstance, SegmentationResult, _mask_centroid


def postprocess(
    seg: SegmentationResult,
    *,
    min_area_px: int = 64,
    merge_overlap: float = 0.60,
    watershed_split: bool = True,
) -> SegmentationResult:
    """Apply counting-oriented post-processing to a raw segmentation.

    Steps
    -----
    1. Filter out instances smaller than *min_area_px*.
    2. Merge overlapping instances (IoU ≥ *merge_overlap*) to avoid
       double-counting.
    3. Optionally attempt watershed splitting on large connected
       components in the combined mask so that touching cells are
       counted individually.

    Returns a *new* ``SegmentationResult`` — the original is not mutated.
    """
    instances = seg.instances

    # 1. Filter small objects
    instances = [c for c in instances if c.area_px >= min_area_px]

    # 2. Merge heavily overlapping masks
    instances = _merge_overlapping(instances, merge_overlap)

    # 3. Watershed split for touching cells
    if watershed_split:
        instances = _watershed_split(instances, seg.image_hw)

    # Rebuild combined mask
    h, w = seg.image_hw
    combined = np.zeros((h, w), dtype=np.uint8)
    for c in instances:
        combined = cv2.bitwise_or(combined, c.mask)

    return SegmentationResult(
        instances=instances,
        combined_mask=combined,
        image_hw=seg.image_hw,
    )


# ── helpers ────────────────────────────────────────────────


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    inter = int(np.sum((a > 127) & (b > 127)))
    union = int(np.sum((a > 127) | (b > 127)))
    return inter / union if union > 0 else 0.0


def _merge_overlapping(
    instances: list[CellInstance],
    threshold: float,
) -> list[CellInstance]:
    """Merge pairs whose mask IoU exceeds *threshold*."""
    if len(instances) <= 1:
        return instances

    merged = list(instances)
    changed = True
    while changed:
        changed = False
        new: list[CellInstance] = []
        used = set()
        for i in range(len(merged)):
            if i in used:
                continue
            best_j = -1
            best_iou = 0.0
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                iou = _mask_iou(merged[i].mask, merged[j].mask)
                if iou >= threshold and iou > best_iou:
                    best_j = j
                    best_iou = iou
            if best_j >= 0:
                # Merge j into i
                m = cv2.bitwise_or(merged[i].mask, merged[best_j].mask)
                area = int((m > 127).sum())
                cx, cy = _mask_centroid(m)
                x1 = min(merged[i].bbox[0], merged[best_j].bbox[0])
                y1 = min(merged[i].bbox[1], merged[best_j].bbox[1])
                x2 = max(merged[i].bbox[2], merged[best_j].bbox[2])
                y2 = max(merged[i].bbox[3], merged[best_j].bbox[3])
                conf = max(merged[i].confidence, merged[best_j].confidence)
                new.append(CellInstance(
                    mask=m, confidence=conf, bbox=(x1, y1, x2, y2),
                    area_px=area, centroid=(cx, cy),
                ))
                used.update({i, best_j})
                changed = True
            else:
                new.append(merged[i])
        merged = new
    return merged


def _watershed_split(
    instances: list[CellInstance],
    image_hw: tuple[int, int],
) -> list[CellInstance]:
    """Attempt watershed separation on individual instance masks.

    Only processes masks where the connected-component count is > 1
    after erosion, suggesting touching cells.
    """
    result: list[CellInstance] = []
    for cell in instances:
        sub_cells = _split_single(cell)
        result.extend(sub_cells)
    return result


def _split_single(cell: CellInstance) -> list[CellInstance]:
    """Try to split one mask into multiple cells via watershed."""
    mask = cell.mask
    if mask.size == 0:
        return [cell]

    # Erode to find sure foreground seeds
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask, kernel, iterations=2)

    n_labels, labels = cv2.connectedComponents(eroded)
    if n_labels <= 2:
        # 1 background + ≤ 1 foreground → single cell, no split
        return [cell]

    # Watershed needs a 3-channel image — use mask as greyscale
    img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Markers: background=1, each seed gets label ≥ 2
    markers = np.zeros(mask.shape[:2], dtype=np.int32)
    markers[mask == 0] = 1  # sure background
    for lbl in range(1, n_labels):
        markers[labels == lbl] = lbl + 1

    cv2.watershed(img_color, markers)

    # Extract split cells
    sub_cells: list[CellInstance] = []
    for lbl in range(2, n_labels + 1):
        region = (markers == lbl).astype(np.uint8) * 255
        # Restrict to original mask area
        region = cv2.bitwise_and(region, mask)
        area = int((region > 127).sum())
        if area < 16:
            continue
        cx, cy = _mask_centroid(region)
        ys, xs = np.where(region > 127)
        if len(xs) == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        sub_cells.append(CellInstance(
            mask=region,
            confidence=cell.confidence,
            bbox=bbox,
            area_px=area,
            centroid=(cx, cy),
            class_id=cell.class_id,
        ))

    return sub_cells if sub_cells else [cell]
