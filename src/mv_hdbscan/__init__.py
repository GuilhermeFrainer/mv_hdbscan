"""Multi-View HDBSCAN (MV-HDBSCAN) package.

Provides scikit-learn compatible clustering estimators for multi-view data.
"""

from .mv_hdbscan import MultiViewHDBSCAN
from .stacking import FeatureStackingHDBSCAN

__all__ = [
    "MultiViewHDBSCAN",
    "FeatureStackingHDBSCAN",
]
