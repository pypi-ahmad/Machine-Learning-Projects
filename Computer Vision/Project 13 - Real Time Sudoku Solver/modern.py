"""
Modern Sudoku Solver — OpenCV + Digit OCR + Backtracking
=========================================================
Replaces legacy Keras digit classifier with a full pipeline:
  1. Document scanner (OpenCV warp perspective)
  2. Grid extraction (contour → 9×9 cell splitting)
  3. Digit recognition (YOLO26-cls or simple template matching)
  4. Backtracking solver
  5. Overlay solution on original frame

Original: Sudoku.py (Keras load_model digit_model.h5 + OpenCV contour grid)
Modern:   OpenCV grid detection + YOLO26-cls or OCR digits + backtracking

Usage:
    python -m core.runner --import-all sudoku_solver_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


# ---------------------------------------------------------------------------
# Backtracking Sudoku solver
# ---------------------------------------------------------------------------

def _solve(board: np.ndarray) -> bool:
    """Solve a 9×9 board in-place using backtracking. Returns True if solved."""
    empty = _find_empty(board)
    if empty is None:
        return True
    r, c = empty
    for num in range(1, 10):
        if _is_valid(board, r, c, num):
            board[r, c] = num
            if _solve(board):
                return True
            board[r, c] = 0
    return False


def _find_empty(board: np.ndarray):
    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                return (r, c)
    return None


def _is_valid(board: np.ndarray, row: int, col: int, num: int) -> bool:
    if num in board[row, :]:
        return False
    if num in board[:, col]:
        return False
    r0, c0 = 3 * (row // 3), 3 * (col // 3)
    if num in board[r0:r0 + 3, c0:c0 + 3]:
        return False
    return True


# ---------------------------------------------------------------------------
# Grid extraction helpers
# ---------------------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _four_point_warp(image: np.ndarray, pts: np.ndarray, size: int = 450) -> np.ndarray:
    rect = _order_points(pts)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (size, size))


def _find_grid_contour(gray: np.ndarray):
    """Find the largest 4-sided contour (the Sudoku grid)."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            return approx.reshape(4, 2).astype("float32")
    return None


def _extract_cells(warped: np.ndarray) -> list:
    """Split a 450×450 warped grid image into 81 cell images."""
    cells = []
    step = warped.shape[0] // 9
    for r in range(9):
        row = []
        for c in range(9):
            cell = warped[r * step:(r + 1) * step, c * step:(c + 1) * step]
            row.append(cell)
        cells.append(row)
    return cells


def _classify_digit(cell_gray: np.ndarray) -> int:
    """Classify a single cell digit using contour area heuristics.

    Returns 0 for empty cells, or the predicted digit 1-9.
    For production accuracy, replace with a trained YOLO26-cls or CNN model.
    """
    h, w = cell_gray.shape
    margin = int(0.15 * h)
    roi = cell_gray[margin:h - margin, margin:w - margin]
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    nonzero = cv2.countNonZero(thresh)
    total = thresh.shape[0] * thresh.shape[1]
    if nonzero / total < 0.03:
        return 0

    # Attempt OCR via contour bounding and simple template if available
    # For now, return -1 to signal "digit present but needs OCR model"
    return -1


@register("sudoku_solver_v2")
class SudokuSolverV2(CVProject):
    display_name = "Sudoku Solver (OpenCV + Backtracking)"
    category = "detection"

    _digit_model = None
    _ocr_reader = None
    _ocr_type = None  # "easyocr" | "paddle" | None

    def load(self):
        # Priority 1: YOLO26-cls trained on digit data (only if custom weights registered)
        try:
            from models.registry import resolve
            weights, version, is_default = resolve("sudoku_solver", "cls")
            if not is_default:
                from utils.yolo import load_yolo
                self._digit_model = load_yolo(weights)
                print(f"  [sudoku_solver] digit classifier: version={version} weights={weights}")
                return
        except Exception:
            pass

        # Priority 2: PaddleOCR (best accuracy for single-digit recognition)
        try:
            from paddleocr import PaddleOCR
            self._ocr_reader = PaddleOCR(
                use_angle_cls=False, lang="en", use_gpu=True, show_log=False
            )
            self._ocr_type = "paddle"
            print("  [sudoku_solver] Using PaddleOCR for digit recognition")
            return
        except ImportError:
            pass

        # Priority 3: EasyOCR fallback
        try:
            import easyocr
            self._ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
            self._ocr_type = "easyocr"
            print("  [sudoku_solver] PaddleOCR not installed — using EasyOCR fallback")
            return
        except ImportError:
            pass

        print("  [sudoku_solver] No digit recognition engine available")
        print("  [sudoku_solver] Install paddleocr or easyocr for digit reading")

    def _read_digit(self, cell_gray: np.ndarray) -> int:
        """Classify a single cell using the best available engine."""
        # YOLO classifier
        if self._digit_model is not None:
            h, w = cell_gray.shape
            margin = int(0.15 * h)
            roi = cell_gray[margin:h - margin, margin:w - margin]
            if roi.size == 0:
                return 0
            rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            results = self._digit_model(rgb, verbose=False)
            if results and results[0].probs is not None:
                cls_id = int(results[0].probs.top1)
                conf = float(results[0].probs.top1conf)
                if conf > 0.5:
                    return cls_id % 10
            return 0

        # Check empty cell first
        h, w = cell_gray.shape
        margin = int(0.15 * h)
        roi = cell_gray[margin:h - margin, margin:w - margin]
        if roi.size == 0:
            return 0
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        nonzero = cv2.countNonZero(thresh)
        total = thresh.shape[0] * thresh.shape[1]
        if total == 0 or nonzero / total < 0.03:
            return 0

        # OCR on individual cell
        if self._ocr_reader is not None:
            rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            if self._ocr_type == "easyocr":
                results = self._ocr_reader.readtext(rgb, allowlist="123456789")
                for _, text, conf in results:
                    text = text.strip()
                    if text.isdigit() and 1 <= int(text) <= 9 and conf > 0.3:
                        return int(text)
            elif self._ocr_type == "paddle":
                results = self._ocr_reader.ocr(rgb, cls=False)
                if results and results[0]:
                    for line in results[0]:
                        text, conf = line[1][0].strip(), line[1][1]
                        if text.isdigit() and 1 <= int(text) <= 9 and conf > 0.3:
                            return int(text)
        return 0

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        grid_pts = _find_grid_contour(gray)
        if grid_pts is None:
            return {"found": False, "message": "No grid detected"}

        warped = _four_point_warp(gray, grid_pts, size=450)
        cells = _extract_cells(warped)

        board = np.zeros((9, 9), dtype=np.int32)
        for r in range(9):
            for c in range(9):
                board[r, c] = self._read_digit(cells[r][c])

        original_board = board.copy()
        solved = _solve(board)

        return {
            "found": True,
            "grid_pts": grid_pts,
            "original": original_board,
            "solution": board if solved else None,
            "solved": solved,
        }

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        vis = frame.copy()

        if not output.get("found", False):
            cv2.putText(vis, "No Sudoku grid detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis

        grid_pts = output["grid_pts"]
        pts_int = grid_pts.astype(np.int32)
        cv2.polylines(vis, [pts_int], True, (0, 255, 0), 2)

        if output.get("solved") and output.get("solution") is not None:
            original = output["original"]
            solution = output["solution"]

            # Overlay solution digits onto original frame via inverse warp
            overlay = np.zeros((450, 450, 3), dtype=np.uint8)
            step = 50  # 450 / 9
            for r in range(9):
                for c in range(9):
                    if original[r, c] == 0 and solution[r, c] > 0:
                        cx = c * step + step // 2 - 8
                        cy = r * step + step // 2 + 8
                        cv2.putText(overlay, str(solution[r, c]), (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Inverse warp overlay back to original perspective
            rect = _order_points(grid_pts)
            src_pts = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, rect)
            warped_overlay = cv2.warpPerspective(overlay, M, (vis.shape[1], vis.shape[0]))
            mask = warped_overlay.sum(axis=2) > 0
            vis[mask] = warped_overlay[mask]

            status = "SOLVED"
            color = (0, 255, 0)
        else:
            status = "UNSOLVED"
            color = (0, 0, 255)

        unknown = 0
        info = f"Sudoku: {status}"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis
