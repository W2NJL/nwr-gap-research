"""
JUDY — FM Field Strength Predictor (Local XGBoost Model)

Standalone version of the RadioLand server's ML predictor.
Trained on ~24K FM station measurements; achieves 2.5–3.9 dBu MAE.

Usage:
    from judy_predictor import JudyPredictor

    judy = JudyPredictor()          # loads model from same directory
    dbu = judy.predict(
        frequency=88.5,
        erp=10000,
        haat=300,
        lat=39.95,
        lon=75.17,
        receiver_lat=40.26,
        receiver_lon=74.27,
        distance_miles=50.0,
        bearing=45.0,
    )
    print(f"{dbu:.1f} dBu")

NOTE ON NWR:
    JUDY was trained on FM stations (88.1–108.1 MHz).
    NWR operates at 162.400–162.550 MHz — outside the training range.
    The model clamps frequency to the training maximum (108.1 MHz) for
    NWR inputs. Treat NWR predictions as rough order-of-magnitude estimates
    and validate against RadioLand API results.
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb          # noqa: F401 — just to check availability
    from joblib import load
    _DEPS_AVAILABLE = True
except ImportError as _e:
    logger.warning("Missing dependency: %s — install xgboost and joblib", _e)
    _DEPS_AVAILABLE = False


# Map FCC class string → numeric power proxy used as a feature
_CLASS_MAP = {
    'A':  10.0,
    'B':   7.5,
    'B1':  7.5,
    'C':   5.0,
    'C0':  5.0,
    'C1':  5.0,
    'C2':  5.0,
    'C3':  5.0,
    'D':   2.5,
}

# Min/max ranges used for min-max normalization during training
_FEATURE_RANGES = {
    "frequency":          (88.1,               108.1),
    "antenna_type":       (0,                  1),
    "aant_rotation_deg":  (0,                  360),
    "erp":                (np.log10(0.1),      np.log10(100_000)),
    "lat":                (15,                 70),
    "lon":                (-170,               -50),
    "hagl":               (0,                  600),
    "haat":               (0,                  600),
    "class_flag":         (2.5,                10),
    "uneven_polarization":(0,                  1),
    "amsl":               (0,                  3500),
    "distance":           (0,                  np.log(400 + 1)),
    "receiver_lat":       (15,                 70),
    "receiver_lon":       (-170,               -50),
    "receiver_hagl":      (0,                  100),
    "bearing":            (0,                  360),
}

# Must match the column order used at training time
_FEATURE_ORDER = [
    "frequency", "antenna_type", "aant_rotation_deg",
    "erp", "lat", "lon", "hagl", "haat",
    "class_flag", "uneven_polarization", "amsl",
    "distance", "receiver_lat", "receiver_lon", "receiver_hagl", "bearing",
]


class JudyPredictor:
    """
    Local XGBoost FM field-strength predictor.

    Load once; call predict() or predict_batch() as many times as you like.
    Thread-safe for reads after __init__.
    """

    DEFAULT_MODEL_FILENAME = "xgboost_field_strength_stratified.joblib"

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self._loaded = False

        if not _DEPS_AVAILABLE:
            logger.error("Cannot load JUDY: xgboost and/or joblib not installed.")
            return

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.DEFAULT_MODEL_FILENAME,
            )

        if not os.path.exists(model_path):
            logger.error("Model file not found: %s", model_path)
            return

        try:
            self.model = load(model_path)
            self._loaded = True
            logger.info("JUDY model loaded from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load JUDY model: %s", exc)

    @property
    def ready(self) -> bool:
        return self._loaded and self.model is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        frequency: float,
        erp: float,
        haat: float,
        lat: float,
        lon: float,
        receiver_lat: float,
        receiver_lon: float,
        distance_miles: float,
        bearing: float,
        antenna_type: str = "NON",
        aant_rotation_deg: float = 0.0,
        hagl: float = 0.0,
        class_flag: str = "C",
        uneven_polarization: float = 0.0,
        amsl: float = 0.0,
        receiver_hagl: float = 10.0,
    ) -> Optional[float]:
        """
        Predict field strength for one transmitter–receiver pair.

        Parameters
        ----------
        frequency       : TX frequency in MHz (training range 88.1–108.1; NWR will clip)
        erp             : Effective Radiated Power in watts
        haat            : Height Above Average Terrain in meters
        lat, lon        : Transmitter coordinates (lon may be positive — forced negative)
        receiver_lat/lon: Receiver coordinates
        distance_miles  : TX–RX distance in miles (NOT km, despite legacy naming)
        bearing         : Bearing from receiver to transmitter, 0–360°
        antenna_type    : "DRL" for directional; anything else = non-directional
        aant_rotation_deg: Antenna pattern rotation in degrees
        hagl            : TX height above ground level in meters
        class_flag      : FCC class ("A", "B", "B1", "C", "C0"–"C3", "D")
        uneven_polarization: 0–1 (most stations = 0)
        amsl            : TX altitude above mean sea level in meters
        receiver_hagl   : Receiver height above ground in meters (default 10)

        Returns
        -------
        float | None  — predicted field strength in dBu, or None on error
        """
        if not self.ready:
            return None
        try:
            x = self._build_feature_array(
                frequency, erp, haat, lat, lon,
                receiver_lat, receiver_lon, distance_miles, bearing,
                antenna_type, aant_rotation_deg, hagl, class_flag,
                uneven_polarization, amsl, receiver_hagl,
            )
            return float(self.model.predict(x)[0])
        except Exception as exc:
            logger.warning("predict() failed: %s", exc)
            return None

    _PREDICT_DEFAULTS = {
        "antenna_type": "NON",
        "aant_rotation_deg": 0.0,
        "hagl": 0.0,
        "class_flag": "C",
        "uneven_polarization": 0.0,
        "amsl": 0.0,
        "receiver_hagl": 10.0,
    }

    def predict_batch(self, rows: list) -> list:
        """
        Predict field strength for many TX–RX pairs in one model call.

        Parameters
        ----------
        rows : list of dicts, each containing the same keyword arguments
               accepted by predict(). Optional params use the same defaults
               as predict() when omitted.

        Returns
        -------
        list of float | None  — same length as input, None for any failed row
        """
        if not self.ready:
            return [None] * len(rows)

        arrays, valid_idx = [], []
        for i, kw in enumerate(rows):
            try:
                merged = {**self._PREDICT_DEFAULTS, **kw}
                arrays.append(self._build_feature_array(**merged)[0])
                valid_idx.append(i)
            except Exception as exc:
                logger.debug("Row %d skipped: %s", i, exc)

        if not arrays:
            return [None] * len(rows)

        preds = self.model.predict(np.vstack(arrays))
        results = [None] * len(rows)
        for rank, idx in enumerate(valid_idx):
            results[idx] = float(preds[rank])
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_feature_array(
        self, frequency, erp, haat, lat, lon,
        receiver_lat, receiver_lon, distance_miles, bearing,
        antenna_type, aant_rotation_deg, hagl, class_flag,
        uneven_polarization, amsl, receiver_hagl,
    ) -> np.ndarray:
        raw = {
            "frequency":           frequency,
            "antenna_type":        1 if str(antenna_type).upper() == "DRL" else 0,
            "aant_rotation_deg":   aant_rotation_deg,
            "erp":                 np.log10(max(erp, 1e-3)),
            "lat":                 lat,
            "lon":                 -abs(lon),
            "hagl":                hagl,
            "haat":                haat,
            "class_flag":          _CLASS_MAP.get(str(class_flag).upper().strip(), 5.0),
            "uneven_polarization": uneven_polarization,
            "amsl":                amsl,
            "distance":            np.log(distance_miles + 1),
            "receiver_lat":        receiver_lat,
            "receiver_lon":        -abs(receiver_lon),
            "receiver_hagl":       receiver_hagl,
            "bearing":             bearing,
        }

        normed = []
        for name in _FEATURE_ORDER:
            v = raw[name]
            lo, hi = _FEATURE_RANGES[name]
            normed.append(float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)))

        return np.array([normed])
