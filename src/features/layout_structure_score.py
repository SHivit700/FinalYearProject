"""
Predict layout_structure_score using the trained regressor in 'models/layout_structure_model.joblib'.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from models.train_layout_model import load_model, predict_scores_from_shapes

_MODEL = None

def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def compute_layout_structure_score(
    shapes: List[Dict[str, Any]],
    image_shape: Tuple[int, ...],
) -> Dict[str, float]:
    """
    Return a dict with ``layout_structure_score`` in [0, 1] from the saved joblib model.
    """
    return predict_scores_from_shapes(_model(), shapes, image_shape)
