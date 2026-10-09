"""Cache detections once, then replay tracking for bonus analyses.

Run from the repository root with the lab virtual environment active:

    python student/bonus/analyze_bonus.py cache
    python student/bonus/analyze_bonus.py analyze

The cache is written under student/artifacts (git-ignored). Final evidence is
written under student/bonus and is intentionally small enough to review.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
STUDENT = ROOT / "student"
ARTIFACTS = STUDENT / "artifacts"
BONUS = STUDENT / "bonus"
DEFAULT_SEGMENT = "training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord"


def _cache_path(segment_name: str) -> Path:
    tag = "default" if segment_name == DEFAULT_SEGMENT else "alternate"
    return ARTIFACTS / f"bonus_cache_{tag}.json.gz"


def _prepare_imports() -> None:
    os.environ.setdefault("DAY23_STUDENT_ROOT", str(STUDENT))
    os.environ.setdefault("FUSION_LAB_PLATFORM", str(ROOT / "platform"))
    from fusion_lab.paths import ensure_platform_on_path

    ensure_platform_on_path()


def cache_segment(segment: str | None = None, end_frame: int | None = None) -> None:
    _prepare_imports()
    from fusion_lab.evaluation import detection_counts, valid_ground_truth
    from fusion_lab.scripts import run_lab
    from fusion_lab.workspace_loader import load_workspace

    run_lab._setup_import_paths()
    cfg = run_lab._load_paths_config(STUDENT / "config" / "paths.yaml")
    if segment:
        cfg["segment"] = segment
    segment_name = cfg.get(
        "segment",
        "training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord",
    )
    tfrecord = Path(cfg["waymo_dir"]) / segment_name
    if not tfrecord.is_file():
        raise FileNotFoundError(f"Waymo segment not found: {tfrecord}")

    ws = load_workspace()
    det_pipe = ws["detection_pipeline"]
    bev_module = ws["bev_mapping"]
    det_metrics = ws["detection_metrics"]
    weights = run_lab._resolve_weights(cfg)
    det_cfg = det_pipe.load_fpn_resnet_config(str(weights) if weights else None)
    model = det_pipe.create_fpn_model(det_cfg, str(weights) if weights else None)

    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
    from simple_waymo_open_dataset_reader import utils as waymo_utils
    import torch

    from fusion_lab.lidar_pcl import pcl_from_range_image

    first = int(cfg.get("frame_start", 0))
    last = int(cfg.get("frame_end", 198))
    if end_frame is not None:
        last = min(last, end_frame)
    rng = np.random.default_rng(0)
    reader = WaymoDataFileReader(str(tfrecord))
    frames: list[dict[str, Any]] = []
    camera_calibration = None
    for frame_index, frame in enumerate(reader):
        if frame_index < first:
            continue
        if frame_index > last:
            break
        if camera_calibration is None:
            calibration = waymo_utils.get(
                frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT
            )
            camera_calibration = {
                "extrinsic": list(calibration.extrinsic.transform),
                "intrinsic": list(calibration.intrinsic),
                "width": int(calibration.width),
                "height": int(calibration.height),
            }
        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        bev = bev_module.bev_maps_from_pcl(points, det_cfg)
        tensor = torch.from_numpy(bev).unsqueeze(0).float()
        detections = det_pipe.detect_objects_from_bev(tensor, model, det_cfg)
        detections = np.asarray(detections, dtype=float).reshape(-1, 8).tolist()
        labels = valid_ground_truth(
            frame.laser_labels, det_cfg, label_pb2.Label.Type.TYPE_VEHICLE
        )
        counts = detection_counts(labels, detections, det_metrics)
        group = next(
            (g for g in frame.camera_labels if g.name == dataset_pb2.CameraName.FRONT),
            None,
        )
        camera_pixels = []
        if group is not None:
            for label in group.labels:
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                    pixel = np.array([label.box.center_x, label.box.center_y])
                    camera_pixels.append((pixel + rng.normal(0, 0.5, 2)).tolist())
        frames.append({
            "frame": frame_index,
            "detections": detections,
            "gt": [
                {
                    "id": label.id,
                    "center": [label.box.center_x, label.box.center_y, label.box.center_z],
                    "length": label.box.length,
                    "width": label.box.width,
                    "height": label.box.height,
                    "heading": label.box.heading,
                }
                for label in labels
            ],
            "detection_counts": counts,
            "camera_pixels": camera_pixels if group is not None else None,
        })
        print(f"cached frame {frame_index}: {len(detections)} detections", flush=True)

    if not frames or camera_calibration is None:
        raise RuntimeError("No frames or FRONT camera calibration were read")
    cache = {
        "segment": segment_name,
        "frames": [first, frames[-1]["frame"]],
        "seed": 0,
        "camera_calibration": camera_calibration,
        "records": frames,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(segment_name)
    with gzip.open(cache_path, "wt", encoding="utf-8") as stream:
        json.dump(cache, stream, separators=(",", ":"))
    print(f"Wrote {cache_path} ({len(frames)} frames)")


def _calibration(cache: dict[str, Any], yaw_degrees: float) -> Any:
    data = cache["camera_calibration"]
    transform = np.asarray(data["extrinsic"], dtype=float).reshape(4, 4).copy()
    yaw = np.deg2rad(yaw_degrees)
    yaw_rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    transform[:3, :3] = yaw_rotation @ transform[:3, :3]
    return SimpleNamespace(
        intrinsic=data["intrinsic"], width=data["width"], height=data["height"],
        extrinsic=SimpleNamespace(transform=transform.reshape(-1).tolist()),
    )


def _labels(frame: dict[str, Any]) -> list[Any]:
    return [
        SimpleNamespace(
            id=row["id"],
            box=SimpleNamespace(
                center_x=row["center"][0], center_y=row["center"][1],
                center_z=row["center"][2], length=row["length"],
                width=row["width"], height=row["height"], heading=row["heading"],
            ),
        )
        for row in frame["gt"]
    ]


def _snapshot(manager: Any) -> list[dict[str, Any]]:
    result = []
    for track in manager.track_list:
        xyz = np.asarray(track.x, dtype=float).reshape(-1)[:3]
        result.append({
            "id": int(track.id), "state": track.state, "score": float(track.score),
            "position": xyz.tolist(), "height": float(track.height),
            "width": float(track.width), "length": float(track.length),
            "yaw": float(track.yaw),
        })
    return result


def _match_track_ids(tracks: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[int, str]:
    confirmed = [track for track in tracks if track["state"] == "confirmed"]
    if not confirmed or not gt:
        return {}
    track_xy = np.asarray([track["position"][:2] for track in confirmed])
    gt_xy = np.asarray([label["center"][:2] for label in gt])
    distances = np.linalg.norm(track_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    eligible = np.isfinite(distances) & (distances <= 2.0)
    n, m = eligible.shape
    penalty = (n + m + 1) * (float(distances[eligible].max()) + 1) if eligible.any() else 1.0
    padded = np.full((n + m, n + m), penalty)
    padded[:n, :m] = np.where(eligible, distances, penalty)
    rows, cols = linear_sum_assignment(padded)
    return {
        int(confirmed[i]["id"]): str(gt[j]["id"])
        for i, j in zip(rows, cols)
        if i < n and j < m and eligible[i, j]
    }


def _match_unassigned_ids(row: dict[str, Any]) -> list[int]:
    matched = _match_track_ids(row["fused_tracks"], row["gt"])
    return [
        int(track["id"]) for track in row["fused_tracks"]
        if track["state"] == "confirmed" and int(track["id"]) not in matched
    ]


def replay(cache: dict[str, Any], yaw_degrees: float = 0.0) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _prepare_imports()
    from fusion_lab import tracking_params
    from fusion_lab.evaluation import aggregate_records, tracking_counts
    from fusion_lab.tracking.filter import Filter
    from fusion_lab.tracking.manager import TrackManager
    from fusion_lab.tracking.sensors import Sensor
    from fusion_lab.workspace_loader import load_workspace

    ws = load_workspace()
    kalman = ws["kalman"]
    association = ws["association"]
    camera_module = ws["camera_fusion"]
    lidar = Sensor("lidar", None, camera_module)
    camera = Sensor("camera", _calibration(cache, yaw_degrees), camera_module)
    filter_obj = Filter(kalman)
    manager = TrackManager(ws["track_management"])
    cfg = ws["detection_pipeline"].load_fpn_resnet_config()
    accepted_residuals: list[float] = []
    candidate_residuals: list[float] = []
    unmatched_camera = 0
    accepted_observations: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    class InstrumentedFilter:
        def predict(self, track: Any) -> None:
            filter_obj.predict(track)

        def update(self, track: Any, meas: Any) -> None:
            if meas.sensor.name == "camera":
                residual = np.asarray(kalman.innovation(track.x, meas), dtype=float).reshape(-1)
                norm = float(np.linalg.norm(residual))
                accepted_residuals.append(norm)
                accepted_observations.append({
                    "track_id": int(track.id), "pixel": np.asarray(meas.z).reshape(-1).tolist(),
                    "residual": residual.tolist(), "residual_norm": norm,
                })
            filter_obj.update(track, meas)

    instrumented = InstrumentedFilter()
    records = []
    id_history: dict[int, str] = {}
    switches: list[dict[str, Any]] = []
    ghost_frames: list[dict[str, Any]] = []
    for frame in cache["records"]:
        frame_index = int(frame["frame"])
        for track in manager.track_list:
            filter_obj.predict(track)
            track.set_t(frame_index * tracking_params.dt)
        lidar_observations = []
        for detection in frame["detections"]:
            if cfg.lim_x[0] <= detection[1] <= cfg.lim_x[1] and cfg.lim_y[0] <= detection[2] <= cfg.lim_y[1]:
                lidar.generate_measurement(frame_index, detection[1:], lidar_observations)
        association.associate_and_update(manager, lidar_observations, instrumented, lidar)
        before_camera = _snapshot(manager)

        if frame["camera_pixels"] is not None:
            camera_observations = []
            for pixel in frame["camera_pixels"]:
                camera.generate_measurement(frame_index, pixel, camera_observations)
            for track in manager.track_list:
                for measurement in camera_observations:
                    if camera.in_fov(track.x):
                        residual = np.asarray(kalman.innovation(track.x, measurement)).reshape(-1)
                        candidate_residuals.append(float(np.linalg.norm(residual)))
            accepted_start = len(accepted_observations)
            association.associate_and_update(manager, camera_observations, instrumented, camera)
            accepted_this_frame = len(accepted_observations) - accepted_start
            unmatched_camera += len(camera_observations) - accepted_this_frame
        else:
            accepted_this_frame = 0

        after_camera = _snapshot(manager)
        labels = _labels(frame)
        counts = tracking_counts(manager.track_list, labels)
        rec = {
            "mode": "fused", "frame": frame_index,
            **frame["detection_counts"], "valid_gt": len(labels), **counts,
        }
        records.append(rec)
        matched = _match_track_ids(after_camera, frame["gt"])
        for track_id, gt_id in matched.items():
            if track_id in id_history and id_history[track_id] != gt_id:
                switches.append({"frame": frame_index, "track_id": track_id,
                                 "from_gt": id_history[track_id], "to_gt": gt_id})
            id_history[track_id] = gt_id
        matched_ids = set(matched)
        ghosts = [t for t in after_camera if t["state"] == "confirmed" and t["id"] not in matched_ids]
        if ghosts:
            ghost_frames.append({"frame": frame_index, "tracks": ghosts})
        snapshots.append({
            "frame": frame_index, "detections": frame["detections"], "gt": frame["gt"],
            "lidar_tracks": before_camera, "fused_tracks": after_camera,
            "camera_updates": accepted_observations[-accepted_this_frame:] if accepted_this_frame else [],
        })

    metrics = aggregate_records(
        records, "fused", cache["frames"], int(cache["seed"]), cache["segment"]
    )
    diagnostics = {
        "yaw_degrees": yaw_degrees,
        "candidate_innovation_norms": candidate_residuals,
        "accepted_innovation_norms": accepted_residuals,
        "camera_updates": len(accepted_residuals),
        "camera_unmatched": unmatched_camera,
        "id_switches": switches,
        "ghost_frames": ghost_frames,
    }
    return metrics, snapshots, diagnostics


def _summary(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.median(values)), float(np.percentile(values, 95))


def _save_visuals(cache: dict[str, Any], snapshots: list[dict[str, Any]], tag: str) -> list[Path]:
    out_dir = BONUS / f"visualization_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for row in snapshots:
        before = {track["id"]: track for track in row["lidar_tracks"]}
        after = {track["id"]: track for track in row["fused_tracks"]}
        corrections = [
            (track_id, float(np.linalg.norm(np.asarray(after[track_id]["position"][:2]) -
                                            np.asarray(track["position"][:2]))))
            for track_id, track in before.items() if track_id in after
        ]
        visible = [value for _, value in corrections if 0.5 <= value <= 10.0]
        strength = float(np.median(visible)) if visible else 0.0
        if row["camera_updates"] and strength >= 0.05:
            candidates.append((strength, row))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    chosen = []
    seen = set()
    for _, row in candidates:
        frame = row["frame"]
        if all(abs(frame - existing) >= 5 for existing in seen):
            chosen.append(row)
            seen.add(frame)
        if len(chosen) == 3:
            break
    if not chosen:
        raise RuntimeError("No camera updates were available for visualization")

    paths = []
    for row in chosen:
        fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
        before = {track["id"]: track for track in row["lidar_tracks"]}
        after = {track["id"]: track for track in row["fused_tracks"]}
        visible_ids = {
            track_id for track_id, prior in before.items() if track_id in after
            and 0.5 <= float(np.linalg.norm(
                np.asarray(after[track_id]["position"][:2]) -
                np.asarray(prior["position"][:2])
            )) <= 10.0
        }
        visible_gt = [gt for gt in row["gt"] if gt["center"][0] <= 45.0]
        positions = [gt["center"] for gt in visible_gt]
        positions.extend(track["position"] for track_id, track in before.items()
                         if track_id in visible_ids)
        positions.extend(after[track_id]["position"] for track_id in visible_ids)
        if positions:
            y_values = [point[1] for point in positions]
            x_values = [point[0] for point in positions]
            ax.set_xlim(max(-28, min(y_values) - 8), min(28, max(y_values) + 8))
            ax.set_ylim(min(58, max(x_values) + 8), max(-5, min(x_values) - 8))
        else:
            ax.set_xlim(-25, 25)
            ax.set_ylim(58, -5)
        ax.set_aspect("equal")
        ax.set_xlabel("vehicle left/right y (m)")
        ax.set_ylabel("forward x (m)")
        ax.set_title(f"Frame {row['frame']}: LiDAR prior and camera-updated tracks")
        for gt in row["gt"]:
            x, y, _ = gt["center"]
            ax.add_patch(Rectangle((y - gt["width"] / 2, x - gt["length"] / 2),
                                   gt["width"], gt["length"], fill=False,
                                   edgecolor="black", linewidth=1.1))
            ax.scatter(y, x, marker="x", s=35, color="black")
        for gt in visible_gt:
            x, y, _ = gt["center"]
            ax.add_patch(Rectangle((y - gt["width"] / 2, x - gt["length"] / 2),
                                   gt["width"], gt["length"], fill=False,
                                   edgecolor="black", linewidth=1.1))
            ax.scatter(y, x, marker="x", s=35, color="black")
        for detection in row["detections"]:
            if detection[1] <= 45.0:
                ax.scatter(detection[2], detection[1], s=18, color="#9ca3af", alpha=0.45)
        for track_id, prior in before.items():
            if track_id not in visible_ids:
                continue
            posterior = after[track_id]
            px, py = prior["position"][:2]
            qx, qy = posterior["position"][:2]
            ax.add_patch(Rectangle((py - prior["width"] / 2,
                                    px - prior["length"] / 2),
                                   prior["width"], prior["length"], fill=False,
                                   edgecolor="#2563eb", linewidth=1.4,
                                   label="LiDAR prior" if track_id == min(visible_ids) else None))
            ax.add_patch(Rectangle((qy - posterior["width"] / 2,
                                    qx - posterior["length"] / 2),
                                   posterior["width"], posterior["length"], fill=False,
                                   edgecolor="#dc2626", linewidth=1.4,
                                   label="After camera" if track_id == min(visible_ids) else None))
            ax.scatter(py, px, marker="o", s=34, color="#2563eb")
            ax.scatter(qy, qx, marker="s", s=31, color="#dc2626")
            ax.annotate("", xy=(qy, qx), xytext=(py, px),
                        arrowprops={"arrowstyle": "->", "color": "#7c3aed", "lw": 1.5})
            ax.text(qy + 0.2, qx, f"ID {track_id}  Δ={np.linalg.norm(np.asarray([qx-px, qy-py])):.2f} m",
                    fontsize=8, bbox={"facecolor": "white", "alpha": 0.7})
        update_text = (
            f"{len(row['camera_updates'])} camera updates; shown tracks: {len(visible_ids)}"
        )
        ax.text(0.02, 0.02, update_text, transform=ax.transAxes, fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8})
        ax.legend(loc="upper right")
        fig.tight_layout()
        path = out_dir / f"camera_update_frame_{row['frame']:03d}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def _export_cvat(snapshots: list[dict[str, Any]], selection: list[int], tag: str) -> tuple[Path, Path, Path]:
    from fusion_lab.export_cvat import export_tracks_json

    selected = [row for row in snapshots if row["frame"] in selection]
    image_dir = BONUS / f"cvat_frames_{tag}"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_path = BONUS / f"cvat_tracks_{tag}.json"
    xml_path = BONUS / f"cvat_annotations_{tag}.xml"
    track_payload = []
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = "Day23 BEV tracking evidence"
    ET.SubElement(task, "size").text = str(len(selected))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker")
    ET.SubElement(task, "flipped").text = "False"
    labels = ET.SubElement(task, "labels")
    for label_name in ("GT vehicle", "LiDAR track", "Fused track", "Ghost track"):
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = label_name
        ET.SubElement(label, "color").text = "#ff0000"
        ET.SubElement(label, "type").text = "bbox"
        attributes = ET.SubElement(label, "attributes")
        names = ("ground_truth_id",) if label_name == "GT vehicle" else ("track_id", "lifecycle_state")
        for attribute_name in names:
            attribute = ET.SubElement(attributes, "attribute")
            ET.SubElement(attribute, "name").text = attribute_name
            ET.SubElement(attribute, "mutable").text = "False"
            ET.SubElement(attribute, "input_type").text = "text"
            ET.SubElement(attribute, "default_value").text = ""
            ET.SubElement(attribute, "values").text = ""
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "id").text = "0"
    ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = str(max(0, len(selected) - 1))
    ET.SubElement(segment, "url")
    owner = ET.SubElement(task, "owner")
    ET.SubElement(owner, "username")
    ET.SubElement(owner, "email")
    ET.SubElement(task, "assignee")
    ET.SubElement(task, "created")
    ET.SubElement(task, "updated")
    ET.SubElement(task, "dumped")

    for image_index, row in enumerate(selected):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.set_xlim(-25, 25)
        ax.set_ylim(50, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("vehicle y (m)")
        ax.set_ylabel("vehicle x (m)")
        ax.set_title(f"Waymo frame {row['frame']} — CVAT track inspection")
        image = ET.SubElement(root, "image", {
            "id": str(image_index), "name": f"frame_{image_index:03d}.png",
            "width": "800", "height": "600",
        })
        output_tracks = []
        frame_matches = _match_track_ids(row["fused_tracks"], row["gt"])
        frame_ghosts = {
            int(track["id"]) for track in row["fused_tracks"]
            if track["state"] == "confirmed" and int(track["id"]) not in frame_matches
        }
        for gt in row["gt"]:
            x, y, _ = gt["center"]
            ax.scatter(y, x, marker="x", color="black")
            image_box = {"xtl": 400 + (y - gt["width"] / 2) * 12,
                         "xbr": 400 + (y + gt["width"] / 2) * 12,
                         "ytl": 550 - (x + gt["length"] / 2) * 10,
                         "ybr": 550 - (x - gt["length"] / 2) * 10}
            box = ET.SubElement(image, "box", {
                "label": "GT vehicle", "occluded": "0", "source": "manual",
                "z_order": "0", "xtl": f"{image_box['xtl']:.2f}",
                "ytl": f"{image_box['ytl']:.2f}", "xbr": f"{image_box['xbr']:.2f}",
                "ybr": f"{image_box['ybr']:.2f}",
            })
            attr = ET.SubElement(box, "attribute", {"name": "ground_truth_id"})
            attr.text = str(gt["id"])
        for label_name, key, color in (("LiDAR track", "lidar_tracks", "#2563eb"),
                                       ("Fused track", "fused_tracks", "#dc2626")):
            for track in row[key]:
                x, y, _ = track["position"]
                actual_label = (
                    "Ghost track" if key == "fused_tracks"
                    and track["state"] == "confirmed" and int(track["id"]) in frame_ghosts
                    else label_name
                )
                actual_color = "#ff00ff" if actual_label == "Ghost track" else color
                ax.scatter(y, x, color=actual_color, marker="o" if key == "lidar_tracks" else "s")
                if actual_label == "Ghost track":
                    ax.text(y + 0.2, x, f"GHOST ID {track['id']}", color=actual_color, fontsize=8)
                half_w, half_l = track["width"] / 2, track["length"] / 2
                box = ET.SubElement(image, "box", {
                    "label": actual_label, "occluded": "0", "source": "manual", "z_order": "1",
                    "xtl": f"{400 + (y - half_w) * 12:.2f}",
                    "xbr": f"{400 + (y + half_w) * 12:.2f}",
                    "ytl": f"{550 - (x + half_l) * 10:.2f}",
                    "ybr": f"{550 - (x - half_l) * 10:.2f}",
                })
                ET.SubElement(box, "attribute", {"name": "track_id"}).text = str(track["id"])
                if actual_label != "GT vehicle":
                    ET.SubElement(box, "attribute", {"name": "lifecycle_state"}).text = track["state"]
                output_tracks.append({**track, "is_ghost": actual_label == "Ghost track"})
        ax.legend(["GT centre", "LiDAR tracks", "Fused tracks"], loc="upper right")
        fig.tight_layout()
        image_path = image_dir / f"frame_{image_index:03d}.png"
        fig.savefig(image_path)
        plt.close(fig)
        track_payload.append({"frame": row["frame"], "image": image_path.name,
                              "tracks": output_tracks, "ground_truth": row["gt"]})

    export_tracks_json(track_payload, json_path)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return json_path, xml_path, image_dir


def analyze(cache_path: Path | None = None) -> None:
    _prepare_imports()
    cache_path = cache_path or _cache_path(DEFAULT_SEGMENT)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Run the cache command first: {cache_path}")
    # Cache names from the documented ``cache`` command are stable, while a
    # manually supplied path such as ``bonus_cache.json.gz`` is ambiguous.
    tag = "alternate" if "alternate" in cache_path.name else "default"
    with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
        cache = json.load(stream)

    rows = []
    baseline_snapshots = None
    baseline_diagnostics = None
    cvat_snapshots = None
    cvat_diagnostics = None
    cvat_yaw = 0.0
    for yaw in (0.0, 1.0, 3.0, 5.0):
        metrics, snapshots, diagnostics = replay(cache, yaw)
        candidate = diagnostics["candidate_innovation_norms"]
        accepted = diagnostics["accepted_innovation_norms"]
        mean, median, p95 = _summary(candidate)
        accepted_mean, accepted_median, accepted_p95 = _summary(accepted)
        fused = metrics["tracking"]["fused"]
        row = {
            "yaw_error_deg": yaw,
            "candidate_innovation_mean_px": mean,
            "candidate_innovation_median_px": median,
            "candidate_innovation_p95_px": p95,
            "accepted_innovation_mean_px": accepted_mean,
            "accepted_innovation_median_px": accepted_median,
            "accepted_innovation_p95_px": accepted_p95,
            "camera_updates": diagnostics["camera_updates"],
            "camera_unmatched": diagnostics["camera_unmatched"],
            "rmse_m": fused["rmse"], "matches": fused["matches"],
            "ghost_track_frames": fused["ghost_track_frames"],
            "missed_gt_frames": fused["missed_gt_frames"],
            "id_switches": len(diagnostics["id_switches"]),
        }
        rows.append(row)
        if yaw == 0.0:
            baseline_snapshots = snapshots
            baseline_diagnostics = diagnostics
            cvat_snapshots = snapshots
            cvat_diagnostics = diagnostics
        elif cvat_diagnostics is not None and not (
            cvat_diagnostics["ghost_frames"] or cvat_diagnostics["id_switches"]
        ) and (diagnostics["ghost_frames"] or diagnostics["id_switches"]):
            cvat_snapshots = snapshots
            cvat_diagnostics = diagnostics
            cvat_yaw = yaw
        print(json.dumps(row), flush=True)

    BONUS.mkdir(parents=True, exist_ok=True)
    csv_path = BONUS / f"calibration_results_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_calibration(rows, BONUS / f"calibration_plot_{tag}.png")

    if baseline_snapshots is None or baseline_diagnostics is None:
        raise RuntimeError("Baseline replay was not produced")
    visual_paths = _save_visuals(cache, baseline_snapshots, tag)
    if cvat_snapshots is None or cvat_diagnostics is None:
        raise RuntimeError("No replay was available for CVAT export")
    target_frames = _evidence_frames(cvat_snapshots, cvat_diagnostics)
    cvat_json, cvat_xml, image_dir = _export_cvat(cvat_snapshots, target_frames, tag)
    evidence = {
        "segment": cache["segment"], "frames": cache["frames"],
        "yaw_sweep": rows, "ghost_frames": baseline_diagnostics["ghost_frames"],
        "id_switches": baseline_diagnostics["id_switches"],
        "cvat_source_yaw_error_deg": cvat_yaw,
        "cvat_source_ghost_frames": cvat_diagnostics["ghost_frames"],
        "cvat_source_id_switches": cvat_diagnostics["id_switches"],
        "cvat_frames": target_frames,
        "visualization_files": [str(path.relative_to(ROOT)) for path in visual_paths],
        "cvat_ghost_track_ids_by_frame": {
            str(frame): [int(track_id) for track_id in _match_unassigned_ids(
                next(row for row in cvat_snapshots if row["frame"] == frame)
            )]
            for frame in target_frames
        },
        "cvat_json": str(cvat_json.relative_to(ROOT)),
        "cvat_xml": str(cvat_xml.relative_to(ROOT)),
        "cvat_images": str(image_dir.relative_to(ROOT)),
    }
    (BONUS / f"analysis_summary_{tag}.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"Wrote bonus evidence under {BONUS}")


def _evidence_frames(snapshots: list[dict[str, Any]], diagnostics: dict[str, Any]) -> list[int]:
    for row in diagnostics["ghost_frames"]:
        return [row["frame"]]
    for row in diagnostics["id_switches"]:
        return [row["frame"]]
    # A ghost or ID switch may not occur in the selected sequence. Use frames
    # with the largest measured camera corrections and document the limitation.
    scored = []
    for row in snapshots:
        before = {track["id"]: track for track in row["lidar_tracks"]}
        after = {track["id"]: track for track in row["fused_tracks"]}
        largest = max((np.linalg.norm(np.asarray(after[k]["position"][:2]) -
                                      np.asarray(t["position"][:2]))
                       for k, t in before.items() if k in after), default=0.0)
        if largest > 0:
            scored.append((float(largest), row["frame"]))
    return [frame for _, frame in sorted(scored, reverse=True)[:3]]


def _plot_calibration(rows: list[dict[str, Any]], path: Path) -> None:
    fig, left = plt.subplots(figsize=(8, 5), dpi=150)
    angles = [row["yaw_error_deg"] for row in rows]
    left.plot(angles, [row["candidate_innovation_median_px"] for row in rows],
              "o-", color="#2563eb", label="Median candidate innovation")
    left.plot(angles, [row["candidate_innovation_p95_px"] for row in rows],
              "s--", color="#60a5fa", label="P95 candidate innovation")
    left.set_xlabel("Camera extrinsic yaw error (degrees)")
    left.set_ylabel("Innovation norm (pixels)")
    left.grid(alpha=0.25)
    right = left.twinx()
    right.plot(angles, [row["rmse_m"] for row in rows],
               "d-", color="#dc2626", label="Fused RMSE")
    right.set_ylabel("3D tracking RMSE (m)")
    lines1, labels1 = left.get_legend_handles_labels()
    lines2, labels2 = right.get_legend_handles_labels()
    left.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("cache", "analyze"))
    parser.add_argument("--segment", default=None, help="Optional alternate Waymo segment")
    parser.add_argument("--end-frame", type=int, default=None, help="Optional cache frame cap")
    parser.add_argument("--cache", type=Path, default=None, help="Cache file to replay")
    args = parser.parse_args()
    if args.command == "cache":
        cache_segment(args.segment, args.end_frame)
    else:
        analyze(args.cache)


if __name__ == "__main__":
    main()
