#!/usr/bin/env python3
"""
Train a regressor for layout_structure_score from contour-derived features.

Uses k-fold cross-validation to report validation MAE and RMSE; the saved model is fit on all rows after CV.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

_MODELS_DIR = Path(__file__).resolve().parent
_SRC = _MODELS_DIR.parent
_PROJECT_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from detection.detect_shapes import run_shape_detection

CSV_PATH = Path("data/score_annotation.csv")
MODEL_DIR = _MODELS_DIR
MODEL_PATH = MODEL_DIR / "layout_structure_model.joblib"
METRICS_PATH = MODEL_DIR / "layout_structure_metrics.json"
RANDOM_STATE = 42
# K-fold cross-validation for performance estimates; final model is fit on all rows.
CV_FOLDS = 3

TARGET_COL = "layout_structure_score"

# Fixed order for all contour-derived features (CSV scores are never inputs).
FEATURE_KEYS: list[str] = [
    "box_count",
    "mean_box_width",
    "std_box_width",
    "mean_box_height",
    "std_box_height",
    "mean_box_area",
    "std_box_area",
    "mean_center_x",
    "std_center_x",
    "mean_center_y",
    "std_center_y",
    "min_center_x",
    "max_center_x",
    "min_center_y",
    "max_center_y",
    "mean_nn_y_gap",
    "mean_nn_x_gap",
    "cv_center_x",
    "cv_center_y",
    "row_cluster_mean_abs_dist",
    "row_cluster_tightness",
    "col_cluster_mean_abs_dist",
    "col_cluster_tightness",
    "mean_aspect_ratio",
    "std_aspect_ratio",
]


def _safe_std(values: list[float]) -> float:
    """Sample std (ddof=1); 0.0 if fewer than two values."""
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _safe_cv(values: list[float]) -> float:
    """Coefficient of variation std/mean when |mean| is large enough."""
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    m = float(np.mean(arr))
    if abs(m) < 1e-9:
        return 0.0
    return float(np.std(arr, ddof=1) / m)


def _mean_nn_gap(sorted_centers: list[float]) -> float:
    """Mean nearest-neighbour gap along one axis (sorted coordinates)."""
    if len(sorted_centers) < 2:
        return 0.0
    arr = np.sort(np.asarray(sorted_centers, dtype=np.float64))
    return float(np.mean(np.diff(arr)))


def _three_cluster_quantile_centers(coords: np.ndarray) -> list[float]:
    """
    Split coordinates into up to three groups by 1/3 and 2/3 quantiles;
    return mean coordinate per non-empty group.
    """
    n = int(coords.size)
    if n == 0:
        return []
    if n == 1:
        return [float(coords[0])]
    flat = coords.astype(np.float64).ravel()
    q1, q2 = np.quantile(flat, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.zeros(n, dtype=np.int32)
    for i, v in enumerate(flat):
        if v <= q1:
            labels[i] = 0
        elif v <= q2:
            labels[i] = 1
        else:
            labels[i] = 2
    centers: list[float] = []
    for k in range(3):
        mask = labels == k
        if np.any(mask):
            centers.append(float(np.mean(flat[mask])))
    return centers


def _mean_abs_dist_to_nearest_center(
    coords: np.ndarray, centers: list[float]
) -> float:
    if coords.size == 0 or not centers:
        return 0.0
    flat = coords.astype(np.float64).ravel()
    c = np.asarray(centers, dtype=np.float64)
    dists = np.min(np.abs(flat[:, np.newaxis] - c[np.newaxis, :]), axis=1)
    return float(np.mean(dists))


def extract_contour_features_from_shapes(
    shapes: list[dict], image_shape: tuple[int, ...]
) -> dict[str, float]:
    """
    Build numeric layout features from contour shapes (normalized by image size).
    """
    zeros = {k: 0.0 for k in FEATURE_KEYS}
    if not image_shape or len(image_shape) < 2:
        return zeros

    img_h = float(image_shape[0])
    img_w = float(image_shape[1])
    if img_h <= 0 or img_w <= 0:
        return zeros

    boxes: list[tuple[float, float, float, float, float]] = []
    # (cx_n, cy_n, w_n, h_n, area_n) all normalized
    for shape in shapes:
        if shape.get("type") != "contour":
            continue
        if "bbox_xywh" in shape:
            x, y, w, h = (float(v) for v in shape["bbox_xywh"])
        else:
            cnt = shape.get("contour")
            if cnt is None:
                continue
            x, y, w, h = (float(v) for v in cv2.boundingRect(np.asarray(cnt)))

        if w < 40 or h < 20:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        cx_n = cx / img_w
        cy_n = cy / img_h
        w_n = w / img_w
        h_n = h / img_h
        area_n = (w * h) / (img_w * img_h)
        boxes.append((cx_n, cy_n, w_n, h_n, area_n))

    n = len(boxes)
    if n == 0:
        return zeros

    cx_list = [b[0] for b in boxes]
    cy_list = [b[1] for b in boxes]
    w_list = [b[2] for b in boxes]
    h_list = [b[3] for b in boxes]
    area_list = [b[4] for b in boxes]
    aspect_list = [w_list[i] / h_list[i] if h_list[i] > 1e-12 else 0.0 for i in range(n)]

    row_centers = _three_cluster_quantile_centers(np.asarray(cy_list))
    col_centers = _three_cluster_quantile_centers(np.asarray(cx_list))
    row_mad = _mean_abs_dist_to_nearest_center(np.asarray(cy_list), row_centers)
    col_mad = _mean_abs_dist_to_nearest_center(np.asarray(cx_list), col_centers)
    row_tight = 1.0 / (1.0 + row_mad)
    col_tight = 1.0 / (1.0 + col_mad)

    out: dict[str, float] = {
        "box_count": float(n),
        "mean_box_width": float(np.mean(w_list)),
        "std_box_width": _safe_std(w_list),
        "mean_box_height": float(np.mean(h_list)),
        "std_box_height": _safe_std(h_list),
        "mean_box_area": float(np.mean(area_list)),
        "std_box_area": _safe_std(area_list),
        "mean_center_x": float(np.mean(cx_list)),
        "std_center_x": _safe_std(cx_list),
        "mean_center_y": float(np.mean(cy_list)),
        "std_center_y": _safe_std(cy_list),
        "min_center_x": float(min(cx_list)),
        "max_center_x": float(max(cx_list)),
        "min_center_y": float(min(cy_list)),
        "max_center_y": float(max(cy_list)),
        "mean_nn_y_gap": _mean_nn_gap(cy_list),
        "mean_nn_x_gap": _mean_nn_gap(cx_list),
        "cv_center_x": _safe_cv(cx_list),
        "cv_center_y": _safe_cv(cy_list),
        "row_cluster_mean_abs_dist": row_mad,
        "row_cluster_tightness": row_tight,
        "col_cluster_mean_abs_dist": col_mad,
        "col_cluster_tightness": col_tight,
        "mean_aspect_ratio": float(np.mean(aspect_list)),
        "std_aspect_ratio": _safe_std(aspect_list),
    }
    return out


def extract_features_for_image(image_path: Path) -> dict[str, float]:
    """Run shape detection on an image and return contour layout features."""
    shapes, image_shape, _ = run_shape_detection(
        str(image_path.resolve()), output_path=None
    )
    return extract_contour_features_from_shapes(shapes, image_shape)


def _resolve_annotation_csv(csv_path: Path) -> Path:
    """Resolve CSV path."""
    if csv_path.is_absolute():
        return csv_path
    for base in (_SRC / csv_path, _PROJECT_ROOT / "src" / csv_path, _PROJECT_ROOT / csv_path):
        if base.is_file():
            return base
    return _SRC / csv_path


def _resolve_image_path(rel: Path) -> Path | None:
    """Resolve CSV image_path."""
    candidates: list[Path] = []
    if rel.is_absolute():
        candidates.append(rel)
    else:
        candidates.extend(
            [
                _PROJECT_ROOT / rel,
                _SRC / rel,
            ]
        )
        parts = rel.parts
        if parts and parts[0] == "data":
            candidates.append(_SRC / Path(*parts))
    for c in candidates:
        r = c.resolve()
        if r.is_file():
            return r
    return None


def build_training_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    Load annotations, extract contour features per image, attach targets.
    Skips missing files; warns and skips rows where feature extraction fails.
    """
    required = ["image_path", TARGET_COL]
    full_csv = _resolve_annotation_csv(csv_path)
    if not full_csv.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {full_csv}")
    df_raw = pd.read_csv(full_csv)
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    rows: list[dict[str, float | str]] = []
    skipped_missing = 0
    skipped_error = 0

    for _, row in df_raw.iterrows():
        rel = Path(str(row["image_path"]))
        img_path = _resolve_image_path(rel)
        if img_path is None:
            skipped_missing += 1
            continue
        try:
            feats = extract_features_for_image(img_path)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Shape extraction failed for {img_path}: {e}", UserWarning)
            skipped_error += 1
            continue
        rec: dict[str, float | str] = {**feats}
        rec["image_path"] = str(img_path)
        rec[TARGET_COL] = float(row[TARGET_COL])
        rows.append(rec)

    if skipped_missing:
        print(f"Skipped {skipped_missing} rows (image file not found).")
    if skipped_error:
        print(f"Skipped {skipped_error} rows (extraction error).")

    print(f"Successfully used {len(rows)} rows for training data.")
    return pd.DataFrame(rows, columns=["image_path"] + FEATURE_KEYS + [TARGET_COL])


def _rf_estimator() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
    )


def train_model(df: pd.DataFrame, n_splits: int | None = None) -> tuple[object, dict]:
    """
    Train RandomForest regressor with k-fold CV for reporting; return model and metrics.

    Each fold fits a fresh estimator on the training partition and scores the held-out fold. The returned model is fit on the full dataset for deployment.
    """
    k = CV_FOLDS if n_splits is None else n_splits
    if k < 2:
        raise ValueError("n_splits must be at least 2 for cross-validation.")

    feature_columns = [c for c in df.columns if c not in ("image_path", TARGET_COL)]
    X = df[feature_columns].values
    y = df[TARGET_COL].values
    n_samples = int(len(df))
    if n_samples < k:
        raise ValueError(
            f"Need at least {k} samples for {k}-fold CV, got {n_samples}."
        )

    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics: list[dict] = []
    val_maes: list[float] = []
    val_rmses: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        est = _rf_estimator()
        est.fit(X_train, y_train)
        y_pred_train = est.predict(X_train)
        y_pred_val = est.predict(X_val)
        tr_mae = float(mean_absolute_error(y_train, y_pred_train))
        va_mae = float(mean_absolute_error(y_val, y_pred_val))
        tr_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        va_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        val_maes.append(va_mae)
        val_rmses.append(va_rmse)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "train_mae": tr_mae,
                "val_mae": va_mae,
                "train_rmse": tr_rmse,
                "val_rmse": va_rmse,
            }
        )

    val_mae_arr = np.asarray(val_maes, dtype=np.float64)
    val_rmse_arr = np.asarray(val_rmses, dtype=np.float64)
    cv_mae_mean = float(np.mean(val_mae_arr))
    cv_mae_std = float(np.std(val_mae_arr, ddof=1)) if k > 1 else 0.0
    cv_rmse_mean = float(np.mean(val_rmse_arr))
    cv_rmse_std = float(np.std(val_rmse_arr, ddof=1)) if k > 1 else 0.0

    model = _rf_estimator()
    model.fit(X, y)

    metrics: dict = {
        "target": TARGET_COL,
        "cv_n_splits": k,
        "cv_shuffle": True,
        "cv_random_state": RANDOM_STATE,
        "feature_columns": feature_columns,
        "cv_folds": fold_metrics,
        "cv_val_mae_mean": cv_mae_mean,
        "cv_val_mae_std": cv_mae_std,
        "cv_val_rmse_mean": cv_rmse_mean,
        "cv_val_rmse_std": cv_rmse_std,
        "val_mae": {TARGET_COL: cv_mae_mean},
        "val_rmse": {TARGET_COL: cv_rmse_mean},
        "n_samples": n_samples,
        "n_feature_columns": len(feature_columns),
    }
    return model, metrics


def save_artifacts(model: object, metrics: dict) -> None:
    """Persist model and metrics JSON."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")


def load_model() -> object:
    """Load the trained regressor from disk."""
    return joblib.load(MODEL_PATH)


def _load_feature_columns() -> list[str]:
    with open(METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return list(data["feature_columns"])


def predict_scores_from_shapes(
    model: object,
    shapes: list[dict],
    image_shape: tuple[int, ...],
) -> dict[str, float]:
    """
    Predict layout_structure_score from an existing shapes list and image shape.
    """
    feats = extract_contour_features_from_shapes(shapes, image_shape)
    feature_columns = _load_feature_columns()
    row = {k: float(feats.get(k, 0.0)) for k in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)
    raw = float(model.predict(X.values).ravel()[0])
    clamped = float(min(1.0, max(0.0, raw))) if math.isfinite(raw) else 0.0
    return {TARGET_COL: clamped}


def predict_scores_from_image(model: object, image_path: Path) -> dict[str, float]:
    """
    Predict layout_structure_score for a single image file.
    """
    shapes, image_shape, _ = run_shape_detection(
        str(image_path.resolve()), output_path=None
    )
    return predict_scores_from_shapes(model, shapes, image_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run layout score regression.")
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        metavar="IMAGE_PATH",
        help="If set, load model and print predicted scores as JSON only.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Number of folds for cross-validation (default: "
            f"{CV_FOLDS}). Must be >= 2 and <= number of training rows."
        ),
    )
    args = parser.parse_args()

    if args.predict:
        img = _resolve_image_path(Path(args.predict))
        if img is None:
            raise SystemExit(f"Image not found: {args.predict}")
        model = load_model()
        scores = predict_scores_from_image(model, img)
        print(json.dumps(scores))
        return

    df = build_training_dataframe(CSV_PATH)
    if df.empty:
        raise SystemExit("No training rows; cannot train.")

    model, metrics = train_model(df, n_splits=args.cv_folds)
    save_artifacts(model, metrics)

    model_saved = MODEL_PATH.resolve()
    k = metrics["cv_n_splits"]

    print()
    print("=== Training summary ===")
    print(f"Number of samples: {metrics['n_samples']}")
    print(f"Number of feature columns: {metrics['n_feature_columns']}")
    print(f"Cross-validation: {k}-fold (shuffle=True, random_state={RANDOM_STATE})")
    for row in metrics["cv_folds"]:
        print(
            f"  Fold {row['fold']}: train={row['n_train']}, val={row['n_val']}, "
            f"val MAE={row['val_mae']:.6f}, val RMSE={row['val_rmse']:.6f}"
        )
    print(
        f"CV val MAE ({TARGET_COL}): {metrics['cv_val_mae_mean']:.6f} "
        f"± {metrics['cv_val_mae_std']:.6f}"
    )
    print(
        f"CV val RMSE ({TARGET_COL}): {metrics['cv_val_rmse_mean']:.6f} "
        f"± {metrics['cv_val_rmse_std']:.6f}"
    )
    print(f"Final model: fit on all {metrics['n_samples']} samples.")
    print(f"Saved model path: {model_saved}")


if __name__ == "__main__":
    main()
