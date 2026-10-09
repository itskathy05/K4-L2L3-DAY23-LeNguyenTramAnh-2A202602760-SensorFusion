"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()
    sensor_to_vehicle = np.asarray(meas.sensor.sens_to_veh, dtype=float)
    rotation = sensor_to_vehicle[:3, :3]
    translation = sensor_to_vehicle[:3, 3]
    position_sensor = np.asarray(meas.z, dtype=float).reshape(3)
    position_vehicle = rotation @ position_sensor + translation
    state = np.asmatrix(np.concatenate([position_vehicle, np.zeros(3)]).reshape(6, 1))

    measurement_covariance = np.asarray(meas.R, dtype=float)
    position_covariance = rotation @ measurement_covariance @ rotation.T
    covariance = np.zeros((6, 6), dtype=float)
    covariance[:3, :3] = position_covariance
    covariance[3:, 3:] = np.diag([
        params.sigma_p44**2,
        params.sigma_p55**2,
        params.sigma_p66**2,
    ])
    return {
        "x": state,
        "P": np.asmatrix(covariance),
        "state": "initialized",
        "score": 1.0 / params.window,
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    score = float(track["score"])
    if associated:
        score = min(1.0, score + 1.0 / params.window)
    else:
        score = max(0.0, score - 1.0 / params.window)
    track["score"] = score
    if track["state"] != "confirmed":
        track["state"] = (
            "confirmed" if score > params.confirmed_threshold else "tentative"
        )
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()
    covariance = np.asarray(track["P"], dtype=float)
    if covariance[0, 0] > params.max_P or covariance[1, 1] > params.max_P:
        return True
    if track["state"] == "confirmed":
        return float(track["score"]) < params.delete_threshold
    return float(track["score"]) <= 0.0
