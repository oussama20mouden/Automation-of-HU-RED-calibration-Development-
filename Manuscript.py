#!/usr/bin/env python3
"""
HU–RED calibration automation for radiotherapy CT QA.

This script implements the workflow described in the manuscript
"Automation of HU-RED calibration: Development, validation and monthly consistency testing in radiotherapy"
and supports three heterogeneous phantoms:

- CIRS Model 062M (electron density phantom)
- Easy Cube (electron density module)
- Catphan 504 (CTP404 sensitometry module)

Key features
------------
- Loads a DICOM CT series (folder) and converts to Hounsfield Units (HU)
- Automatically selects the best slice (largest phantom cross-section)
- Detects phantom contour and (optionally) auto-identifies phantom type
- Detects insert locations via blob detection on edge-enhanced images
- Computes mean HU and HU std in a central ROI for each insert
- Pairs measured HU with known relative electron density (RED) values (configurable)
- Exports results (CSV + JSON summary) and optional plot (PNG)
- Optional comparison to a baseline CSV and tolerance evaluation

Notes
-----
1) Manufacturer RED values may differ by phantom revision and insert set.
   For clinical use, you SHOULD verify and/or override RED values using --config.
2) Catphan RED defaults are approximations (because Catphan specs may vary).
   Provide your official manufacturer RED list via --config for accurate calibration tables.

Example
-------
python hu_red_calibration.py --dicom /path/to/series --outdir ./out --phantom auto

Config override (recommended):
python hu_red_calibration.py --dicom /path/to/series --outdir ./out --config my_phantom_config.json

Author: Ou H (requested by user)
License: MIT (suggested)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pydicom

from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from skimage import filters, measure, morphology, segmentation
from skimage.feature import blob_log

# Matplotlib is optional; only imported if plotting is requested.
# (Avoids problems in headless environments.)
try:
    import matplotlib.pyplot as plt  # noqa
except Exception:  # pragma: no cover
    plt = None  # type: ignore


# ----------------------------
# Configuration / defaults
# ----------------------------

@dataclass(frozen=True)
class InsertDef:
    name: str
    red: float
    hu_hint: Optional[float] = None  # used for assignment only


@dataclass(frozen=True)
class PhantomDef:
    phantom: str
    inserts: List[InsertDef]
    # ROI: fraction of detected insert radius (avoid edges as per manuscript)
    roi_radius_fraction: float = 0.65


DEFAULT_PHANTOMS: Dict[str, PhantomDef] = {
    # RED values below are typical/representative values used in literature.
    # Always verify with manufacturer certificate for your phantom set.
    "cirs062": PhantomDef(
        phantom="cirs062",
        inserts=[
            InsertDef("air", 0.00, -1000),
            InsertDef("lung_inhale", 0.20, -800),
            InsertDef("lung_exhale", 0.50, -500),
            InsertDef("adipose", 0.95, -100),
            InsertDef("breast", 0.98, -50),
            InsertDef("water", 1.00, 0),
            InsertDef("muscle", 1.04, 40),
            InsertDef("liver", 1.05, 60),
            InsertDef("trabecular_bone", 1.17, 200),
            InsertDef("dense_bone", 1.46, 700),
        ],
        roi_radius_fraction=0.65,
    ),
    "easycube": PhantomDef(
        phantom="easycube",
        inserts=[
            InsertDef("lung", 0.20, -750),
            InsertDef("adipose", 0.97, -100),
            InsertDef("water", 1.00, 0),
            InsertDef("muscle", 1.07, 50),
            InsertDef("bone", 1.58, 800),
        ],
        roi_radius_fraction=0.65,
    ),
    # Catphan CTP404 module inserts; RED values must be verified from the certificate.
    # hu_hint values are typical CT numbers at 120 kV.
    "catphan_ctp404": PhantomDef(
        phantom="catphan_ctp404",
        inserts=[
            InsertDef("air", 0.00, -1000),
            InsertDef("foam", 0.30, -800),
            InsertDef("ldpe", 0.95, -100),
            InsertDef("polystyrene", 1.03, -35),
            InsertDef("acrylic", 1.16, 120),
            InsertDef("delrin", 1.36, 340),
            InsertDef("teflon", 1.87, 950),
        ],
        roi_radius_fraction=0.65,
    ),
}


DEFAULT_TOLERANCES_HU: Dict[str, float] = {
    # Derived from manuscript: ±30 HU water, ±50 HU lung & dense bone, ±40 HU adipose/soft tissue.
    "water": 30.0,
    "lung": 50.0,
    "lung_inhale": 50.0,
    "lung_exhale": 50.0,
    "air": 50.0,  # not explicitly stated; use lung-like tolerance
    "dense_bone": 50.0,
    "trabecular_bone": 40.0,
    "bone": 50.0,
    "adipose": 40.0,
    "breast": 40.0,
    "muscle": 40.0,
    "liver": 40.0,
    # Catphan plastics: treat as soft/bone depending on density
    "foam": 50.0,
    "ldpe": 40.0,
    "polystyrene": 40.0,
    "acrylic": 40.0,
    "delrin": 50.0,
    "teflon": 50.0,
}


# ----------------------------
# Utilities
# ----------------------------

def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def read_dicom_series(path: Path) -> Tuple[np.ndarray, pydicom.dataset.FileDataset, List[pydicom.dataset.FileDataset]]:
    """
    Read a DICOM CT series from a folder (or a single DICOM file).
    Returns:
        volume_hu: (z, y, x) float32 HU volume
        ref_ds: dataset of first slice (for metadata)
        slices: list of datasets sorted by position
    """
    if path.is_file():
        dss = [pydicom.dcmread(str(path), force=True)]
    else:
        files = sorted([p for p in path.rglob("*") if p.is_file()])
        dss = []
        for f in files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=False, force=True)
                if getattr(ds, "Modality", "").upper() == "CT" and hasattr(ds, "PixelData"):
                    dss.append(ds)
            except Exception:
                continue

        if not dss:
            raise FileNotFoundError(f"No CT DICOM slices found under: {path}")

    # Sort slices
    def sort_key(ds: pydicom.dataset.FileDataset) -> float:
        if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
            try:
                return float(ds.ImagePositionPatient[2])
            except Exception:
                pass
        if hasattr(ds, "InstanceNumber"):
            try:
                return float(ds.InstanceNumber)
            except Exception:
                pass
        return 0.0

    dss.sort(key=sort_key)
    ref = dss[len(dss) // 2] if len(dss) > 1 else dss[0]

    # Stack volume
    vol = np.stack([ds.pixel_array.astype(np.float32) for ds in dss], axis=0)

    # Convert to HU (assume constant slope/intercept; if not, do per-slice)
    slopes = [float(getattr(ds, "RescaleSlope", 1.0)) for ds in dss]
    intercepts = [float(getattr(ds, "RescaleIntercept", 0.0)) for ds in dss]
    if np.std(slopes) < 1e-6 and np.std(intercepts) < 1e-6:
        vol_hu = vol * slopes[0] + intercepts[0]
    else:
        vol_hu = np.empty_like(vol, dtype=np.float32)
        for i, ds in enumerate(dss):
            vol_hu[i] = vol[i] * slopes[i] + intercepts[i]

    return vol_hu.astype(np.float32), ref, dss


def robust_phantom_mask(hu_slice: np.ndarray) -> np.ndarray:
    """
    Create a binary mask of the phantom body (largest connected component).
    """
    # Background air is around -1000 HU; phantom body around -200..+200 HU for plastics/water.
    # Use a conservative threshold to include soft-tissue-equivalent materials.
    bw = hu_slice > -400
    bw = morphology.binary_closing(bw, morphology.disk(5))
    bw = ndimage.binary_fill_holes(bw)
    bw = segmentation.clear_border(bw)
    lab = measure.label(bw)
    props = measure.regionprops(lab)
    if not props:
        return np.zeros_like(bw, dtype=bool)
    largest = max(props, key=lambda r: r.area)
    mask = lab == largest.label
    mask = morphology.binary_opening(mask, morphology.disk(3))
    return mask


def select_best_slice(volume_hu: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, measure._regionprops.RegionProperties]:
    """
    Pick the slice with the largest phantom area.
    Returns (z_index, hu_slice, mask, regionprops).
    """
    best_i = 0
    best_area = -1.0
    best_mask = None
    best_prop = None
    for i in range(volume_hu.shape[0]):
        hu = volume_hu[i]
        mask = robust_phantom_mask(hu)
        lab = measure.label(mask)
        props = measure.regionprops(lab)
        if not props:
            continue
        largest = max(props, key=lambda r: r.area)
        if largest.area > best_area:
            best_area = largest.area
            best_i = i
            best_mask = mask
            best_prop = largest
    if best_mask is None or best_prop is None:
        raise RuntimeError("Could not find phantom in any slice.")
    return best_i, volume_hu[best_i], best_mask, best_prop


def circularity(mask: np.ndarray) -> float:
    """
    4πA / P^2. 1.0 = perfect circle.
    """
    lab = measure.label(mask)
    props = measure.regionprops(lab)
    if not props:
        return 0.0
    r = max(props, key=lambda p: p.area)
    area = float(r.area)
    perim = float(measure.perimeter(mask))
    if perim <= 1e-6:
        return 0.0
    return 4.0 * math.pi * area / (perim * perim)


def dedupe_blobs(blobs: np.ndarray, dist_px: float) -> np.ndarray:
    """
    Merge blob detections that are within dist_px of each other (greedy).
    blobs: array of (y, x, sigma)
    """
    if blobs.size == 0:
        return blobs
    kept: List[Tuple[float, float, float]] = []
    for y, x, s in sorted(blobs.tolist(), key=lambda t: t[2], reverse=True):
        ok = True
        for yy, xx, ss in kept:
            if (y - yy) ** 2 + (x - xx) ** 2 <= dist_px ** 2:
                ok = False
                break
        if ok:
            kept.append((y, x, s))
    return np.array(kept, dtype=float)


def detect_insert_blobs(hu_slice: np.ndarray, phantom_mask: np.ndarray) -> np.ndarray:
    """
    Detect circular insert candidates inside the phantom.
    Returns blobs as array (y, x, radius_px).
    """
    # Edge enhancement (insert boundaries)
    sm = filters.gaussian(hu_slice.astype(np.float32), sigma=1.0, preserve_range=True)
    ed = filters.sobel(sm)
    ed = (ed - ed.min()) / (ed.max() - ed.min() + 1e-9)

    blobs = blob_log(
        ed,
        min_sigma=2,
        max_sigma=20,
        num_sigma=18,
        threshold=0.05,
    )
    # blob_log gives sigma; radius ~ sigma * sqrt(2)
    if blobs.size == 0:
        return blobs

    # Keep blobs that are clearly inside the phantom (not on the border)
    filtered = []
    for y, x, sigma in blobs:
        y_i, x_i = int(round(y)), int(round(x))
        if 0 <= y_i < phantom_mask.shape[0] and 0 <= x_i < phantom_mask.shape[1] and phantom_mask[y_i, x_i]:
            radius = float(sigma * math.sqrt(2.0))
            # reject extreme sizes
            if 4.0 <= radius <= 40.0:
                filtered.append((float(y), float(x), radius))
    blobs2 = np.array(filtered, dtype=float)

    # Dedupe close detections
    return dedupe_blobs(blobs2, dist_px=10.0)


def roi_stats(hu_slice: np.ndarray, center_yx: Tuple[float, float], roi_radius_px: float) -> Tuple[float, float]:
    """
    Mean and std HU in a circular ROI.
    """
    cy, cx = center_yx
    yy, xx = np.ogrid[:hu_slice.shape[0], :hu_slice.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= roi_radius_px ** 2
    vals = hu_slice[mask]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def phantom_autodetect(phantom_mask: np.ndarray, blobs: np.ndarray) -> str:
    """
    Heuristic autodetection: easycube vs circular; then blob count for cirs/catphan.
    """
    circ = circularity(phantom_mask)
    logging.debug("Circularity=%.3f, blob_count=%d", circ, blobs.shape[0] if blobs is not None else -1)

    if circ < 0.80:
        return "easycube"

    # circular phantoms: use blob count range
    n = blobs.shape[0] if blobs is not None else 0
    if n >= 9:
        return "cirs062"
    if 6 <= n <= 8:
        return "catphan_ctp404"
    # fallback: assume cirs if large
    return "cirs062"


def assign_inserts(
    phantom_def: PhantomDef,
    blobs: np.ndarray,
    hu_slice: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Assign detected blobs to expected inserts using Hungarian assignment on HU hints.
    Returns list of measurement dicts (one per expected insert).
    """
    if blobs.size == 0:
        raise RuntimeError("No insert candidates detected (blob detection returned 0).")

    # Compute HU for each blob using a conservative ROI fraction
    blob_meas = []
    for y, x, r in blobs:
        roi_r = max(2.0, phantom_def.roi_radius_fraction * r)
        mean, std = roi_stats(hu_slice, (y, x), roi_r)
        blob_meas.append({"y": y, "x": x, "r": r, "roi_r": roi_r, "hu_mean": mean, "hu_std": std})

    # Build cost matrix: expected inserts x detected blobs
    exp = phantom_def.inserts
    m = len(exp)
    n = len(blob_meas)

    # If we have more blobs than inserts, keep the most plausible ones by proximity to any hu_hint.
    # (Some phantoms have extra circular features; this keeps analysis stable.)
    if n > m:
        scores = []
        for j in range(n):
            hu = blob_meas[j]["hu_mean"]
            best = 1e9
            for ins in exp:
                if ins.hu_hint is None or math.isnan(hu):
                    continue
                best = min(best, abs(hu - ins.hu_hint))
            scores.append((best, j))
        scores.sort(key=lambda t: t[0])
        keep_idx = [j for _, j in scores[: max(m, min(n, m + 3))]]  # keep a little slack
        blob_meas = [blob_meas[j] for j in keep_idx]
        n = len(blob_meas)

    cost = np.zeros((m, n), dtype=float)
    for i, ins in enumerate(exp):
        for j in range(n):
            hu = blob_meas[j]["hu_mean"]
            if ins.hu_hint is None or math.isnan(hu):
                cost[i, j] = 1e6
            else:
                cost[i, j] = abs(hu - ins.hu_hint)

    # If too few blobs vs inserts, still run assignment on available, mark missing.
    row_ind, col_ind = linear_sum_assignment(cost) if n > 0 else (np.array([], dtype=int), np.array([], dtype=int))
    assigned = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    results: List[Dict[str, Any]] = []
    for i, ins in enumerate(exp):
        if i in assigned and cost[i, assigned[i]] < 1e6:
            b = blob_meas[assigned[i]]
            results.append(
                {
                    "insert": ins.name,
                    "red": ins.red,
                    "hu_mean": b["hu_mean"],
                    "hu_std": b["hu_std"],
                    "center_x": float(b["x"]),
                    "center_y": float(b["y"]),
                    "roi_radius_px": float(b["roi_r"]),
                    "detected_radius_px": float(b["r"]),
                    "assignment_cost": float(cost[i, assigned[i]]),
                }
            )
        else:
            results.append(
                {
                    "insert": ins.name,
                    "red": ins.red,
                    "hu_mean": float("nan"),
                    "hu_std": float("nan"),
                    "center_x": float("nan"),
                    "center_y": float("nan"),
                    "roi_radius_px": float("nan"),
                    "detected_radius_px": float("nan"),
                    "assignment_cost": float("inf"),
                }
            )
    return results


def load_config(path: Optional[Path]) -> Tuple[Dict[str, PhantomDef], Dict[str, float]]:
    """
    Load optional JSON config to override phantom definitions and tolerances.
    """
    phantoms = dict(DEFAULT_PHANTOMS)
    tolerances = dict(DEFAULT_TOLERANCES_HU)

    if path is None:
        return phantoms, tolerances
    obj = json.loads(path.read_text(encoding="utf-8"))

    if "tolerances_hu" in obj and isinstance(obj["tolerances_hu"], dict):
        for k, v in obj["tolerances_hu"].items():
            tolerances[str(k)] = float(v)

    if "phantoms" in obj and isinstance(obj["phantoms"], dict):
        for pname, pdef in obj["phantoms"].items():
            inserts = []
            for ins in pdef.get("inserts", []):
                inserts.append(
                    InsertDef(
                        name=str(ins["name"]),
                        red=float(ins["red"]),
                        hu_hint=float(ins["hu_hint"]) if "hu_hint" in ins and ins["hu_hint"] is not None else None,
                    )
                )
            roi_frac = float(pdef.get("roi_radius_fraction", 0.65))
            phantoms[str(pname)] = PhantomDef(phantom=str(pname), inserts=inserts, roi_radius_fraction=roi_frac)

    return phantoms, tolerances


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return float("nan")
    x = x[m] - np.mean(x[m])
    y = y[m] - np.mean(y[m])
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if den <= 0:
        return float("nan")
    return float(np.sum(x * y) / den)


def compare_to_baseline(current: List[Dict[str, Any]], baseline_csv: Path) -> Dict[str, Any]:
    """
    Compare current results to baseline CSV exported by this script.
    """
    import csv

    base = {}
    with baseline_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ins = row.get("insert", "")
            try:
                base[ins] = float(row.get("hu_mean", "nan"))
            except Exception:
                base[ins] = float("nan")

    deltas = {}
    for row in current:
        ins = row["insert"]
        cur = float(row["hu_mean"])
        ref = float(base.get(ins, float("nan")))
        deltas[ins] = cur - ref if np.isfinite(cur) and np.isfinite(ref) else float("nan")

    r = pearson_r(
        [row["hu_mean"] for row in current],
        [base.get(row["insert"], float("nan")) for row in current],
    )
    return {"delta_hu": deltas, "pearson_r": r}


def evaluate_tolerances(
    current: List[Dict[str, Any]],
    baseline_csv: Optional[Path],
    tolerances: Dict[str, float],
) -> Dict[str, Any]:
    """
    If baseline is provided, evaluate per-insert ΔHU against tolerance thresholds.
    """
    if baseline_csv is None:
        return {"has_baseline": False}

    comp = compare_to_baseline(current, baseline_csv)
    delta = comp["delta_hu"]

    per_insert = []
    ok = True
    for row in current:
        ins = row["insert"]
        d = float(delta.get(ins, float("nan")))
        tol = float(tolerances.get(ins, 40.0))
        within = (abs(d) <= tol) if np.isfinite(d) else False
        if not within:
            ok = False
        per_insert.append({"insert": ins, "delta_hu": d, "tolerance_hu": tol, "within_tolerance": within})

    return {
        "has_baseline": True,
        "pearson_r": comp.get("pearson_r", float("nan")),
        "pass": ok,
        "per_insert": per_insert,
    }


def save_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def save_plot(rows: List[Dict[str, Any]], out_png: Path, title: str) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available in this environment.")
    # Sort by HU for a monotonic curve visualization (as in your original script)
    pts = [(r["hu_mean"], r["red"], r["insert"]) for r in rows if np.isfinite(r["hu_mean"]) and np.isfinite(r["red"])]
    if not pts:
        logging.warning("No finite points to plot; skipping plot.")
        return
    pts.sort(key=lambda t: t[0])
    hu = [p[0] for p in pts]
    red = [p[1] for p in pts]
    labels = [p[2] for p in pts]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(red, hu)
    plt.plot(red, hu, linestyle="--")
    for r, h, lab in zip(red, hu, labels):
        plt.text(r, h, lab, fontsize=8)
    plt.xlabel("Relative Electron Density (RED)")
    plt.ylabel("Mean HU")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_debug_overlay(hu_slice: np.ndarray, rows: List[Dict[str, Any]], out_png: Path, title: str) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available in this environment.")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(hu_slice, cmap="gray", vmin=-1000, vmax=1000)
    for r in rows:
        if np.isfinite(r["center_x"]) and np.isfinite(r["center_y"]) and np.isfinite(r["roi_radius_px"]):
            cx, cy, rr = r["center_x"], r["center_y"], r["roi_radius_px"]
            circ = plt.Circle((cx, cy), rr, fill=False, linewidth=1.5)
            plt.gca().add_patch(circ)
            plt.text(cx, cy, r["insert"], fontsize=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def series_metadata(ref_ds: pydicom.dataset.FileDataset) -> Dict[str, Any]:
    def safe_get(tag: str) -> Optional[str]:
        try:
            v = getattr(ref_ds, tag)
            if isinstance(v, (list, tuple)):
                return ",".join(str(x) for x in v)
            return str(v)
        except Exception:
            return None

    return {
        "SeriesDescription": safe_get("SeriesDescription"),
        "ProtocolName": safe_get("ProtocolName"),
        "KVP": safe_get("KVP"),
        "Manufacturer": safe_get("Manufacturer"),
        "ManufacturerModelName": safe_get("ManufacturerModelName"),
        "StationName": safe_get("StationName"),
        "StudyDate": safe_get("StudyDate"),
        "StudyTime": safe_get("StudyTime"),
        "SliceThickness": safe_get("SliceThickness"),
        "ReconstructionKernel": safe_get("ConvolutionKernel"),
        "PixelSpacing": safe_get("PixelSpacing"),
    }


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Automated HU–RED calibration from CT phantom DICOM series.")
    ap.add_argument("--dicom", required=True, help="Path to DICOM series folder (or a single DICOM file).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument(
        "--phantom",
        default="auto",
        choices=["auto", "cirs062", "easycube", "catphan_ctp404"],
        help="Phantom type. Use 'auto' to autodetect.",
    )
    ap.add_argument("--config", default=None, help="Optional JSON config overriding RED values / tolerances.")
    ap.add_argument("--baseline", default=None, help="Optional baseline CSV to compute ΔHU and pass/fail.")
    ap.add_argument("--no-plot", action="store_true", help="Disable plot output.")
    ap.add_argument("--debug-overlay", action="store_true", help="Save an overlay image with ROIs for QA review.")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    args = ap.parse_args()

    setup_logger(args.verbose)

    dicom_path = Path(args.dicom).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    phantoms, tolerances = load_config(config_path)

    volume_hu, ref_ds, _ = read_dicom_series(dicom_path)
    z_idx, hu_slice, phantom_mask, _ = select_best_slice(volume_hu)

    blobs = detect_insert_blobs(hu_slice, phantom_mask)

    phantom_name = args.phantom
    if phantom_name == "auto":
        phantom_name = phantom_autodetect(phantom_mask, blobs)

    if phantom_name not in phantoms:
        raise ValueError(f"Unknown phantom '{phantom_name}'. Available: {sorted(phantoms.keys())}")

    phantom_def = phantoms[phantom_name]
    measurements = assign_inserts(phantom_def, blobs, hu_slice)

    meta = series_metadata(ref_ds)
    meta["selected_slice_index"] = int(z_idx)
    meta["phantom"] = phantom_name
    meta["dicom_path"] = str(dicom_path)

    # Add metadata columns to each row for easier trending in QATrack+/CSV tools
    rows = []
    for m in measurements:
        row = dict(meta)
        row.update(m)
        rows.append(row)

    out_csv = outdir / f"hu_red_{phantom_name}.csv"
    save_csv(rows, out_csv)

    baseline_csv = Path(args.baseline).expanduser().resolve() if args.baseline else None
    tol_eval = evaluate_tolerances(measurements, baseline_csv, tolerances)

    summary = {
        "metadata": meta,
        "outputs": {
            "csv": str(out_csv),
        },
        "tolerance_evaluation": tol_eval,
        "note": "For clinical use, verify RED values against manufacturer certificate and override via --config.",
    }
    out_json = outdir / f"hu_red_{phantom_name}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Plots
    if not args.no_plot:
        out_png = outdir / f"hu_red_{phantom_name}_curve.png"
        save_plot(measurements, out_png, title=f"HU–RED curve ({phantom_name})")
        summary["outputs"]["curve_png"] = str(out_png)
        if args.debug_overlay:
            out_overlay = outdir / f"hu_red_{phantom_name}_overlay.png"
            save_debug_overlay(hu_slice, measurements, out_overlay, title=f"ROI overlay ({phantom_name})")
            summary["outputs"]["overlay_png"] = str(out_overlay)
            out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summary
    print(f"\n✅ Done. Phantom: {phantom_name}")
    print(f"   Slice used: z={z_idx}")
    print(f"   CSV:  {out_csv}")
    print(f"   JSON: {out_json}")
    if not args.no_plot:
        print(f"   PNG:  {outdir / f'hu_red_{phantom_name}_curve.png'}")
        if args.debug_overlay:
            print(f"   ROI overlay: {outdir / f'hu_red_{phantom_name}_overlay.png'}")

    if tol_eval.get("has_baseline"):
        status = "PASS ✅" if tol_eval.get("pass") else "FAIL ❌"
        r = tol_eval.get("pearson_r")
        print(f"   Baseline comparison: {status} | Pearson r = {r:.5f}" if isinstance(r, float) else f"   Baseline comparison: {status}")

        # Print out-of-tolerance inserts (if any)
        if not tol_eval.get("pass"):
            bad = [x for x in tol_eval.get("per_insert", []) if not x.get("within_tolerance")]
            print("   Out-of-tolerance inserts:")
            for b in bad:
                print(f"    - {b['insert']}: ΔHU={b['delta_hu']:.1f} (tol ±{b['tolerance_hu']:.0f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())