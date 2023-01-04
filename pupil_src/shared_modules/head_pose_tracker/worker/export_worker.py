"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import csv
import logging
import os

logger = logging.getLogger(__name__)

ROTATION_HEADER = tuple("rotation_" + dim for dim in "xyz")
TRANSLATION_HEADER = tuple("translation_" + dim for dim in "xyz")
ORIENTATION_HEADER = ("pitch", "yaw", "roll")
VERTICES_HEADER = tuple(f"vert_{idx}_{dim}" for idx in range(4) for dim in "xyz")

MODEL_HEADER = ("marker_id",) + VERTICES_HEADER
POSES_HEADER = (
    ("timestamp",) + ROTATION_HEADER + TRANSLATION_HEADER + ORIENTATION_HEADER
)


def export_routine(rec_dir, model, poses):
    _export_model(rec_dir, model)
    _export_poses(rec_dir, poses)


def _export_model(rec_dir, model):
    logger.info(f"Exporting head pose model to {rec_dir}")
    model_path = os.path.join(rec_dir, "head_pose_tracker_model.csv")
    _write_csv(model_path, MODEL_HEADER, model)


def _export_poses(rec_dir, poses):
    logger.info(f"Exporting {len(poses)} head poses to {rec_dir}")
    poses_path = os.path.join(rec_dir, "head_pose_tracker_poses.csv")
    poses_flat = [
        (p["timestamp"], *p["camera_poses"], *p["euler_orientation"]) for p in poses
    ]
    _write_csv(poses_path, POSES_HEADER, poses_flat)


def _write_csv(path, header, rows):
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(header)
        writer.writerows(rows)
