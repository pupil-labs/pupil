import typing
import logging

import cv2
import numpy as np

import methods

from surface_tracker.surface_marker import Surface_Marker_UID
from surface_tracker.surface_marker_aggregate import Surface_Marker_Aggregate

from .surface_location import Surface_Location

logger = logging.getLogger(__name__)


Surface_Marker_UID_To_Aggregate_Mapping = typing.Mapping[
    Surface_Marker_UID, Surface_Marker_Aggregate
]


def perspective_transform_points(points, trans_matrix):
    orig_shape = points.shape
    points.shape = (-1, 1, 2)
    img_points = cv2.perspectiveTransform(points, trans_matrix)
    img_points.shape = orig_shape
    return img_points


def find_homographies(points_A, points_B):
    points_A = points_A.reshape((-1, 1, 2))
    points_B = points_B.reshape((-1, 1, 2))

    B_to_A, mask = cv2.findHomography(
        points_A, points_B, method=cv2.RANSAC, ransacReprojThreshold=100
    )

    A_to_B, mask = cv2.findHomography(points_B, points_A)
    return A_to_B, B_to_A


def bounding_quadrangle(vertices):

    # According to OpenCV implementation, cv2.convexHull only accepts arrays with
    # 32bit floats (CV_32F) or 32bit signed ints (CV_32S).
    # See: https://github.com/opencv/opencv/blob/3.4/modules/imgproc/src/convhull.cpp#L137
    # See: https://github.com/pupil-labs/pupil/issues/1544
    vertices = np.asarray(vertices, dtype=np.float32)

    hull_points = cv2.convexHull(vertices, clockwise=False)

    # The convex hull of a list of markers must have at least 4 corners, since a
    # single marker already has 4 corners. If the convex hull has more than 4
    # corners we reduce that number with approximations of the hull.
    if len(hull_points) > 4:
        new_hull = cv2.approxPolyDP(hull_points, epsilon=1, closed=True)
        if new_hull.shape[0] >= 4:
            hull_points = new_hull

    if len(hull_points) > 4:
        curvature = abs(methods.GetAnglesPolyline(hull_points, closed=True))
        most_acute_4_threshold = sorted(curvature)[3]
        hull_points = hull_points[curvature <= most_acute_4_threshold]

    # Vertices space is flipped in y.  We need to change the order of the
    # hull_points vertices
    hull_points = hull_points[[1, 0, 3, 2], :, :]

    # Roll the hull_points vertices until we have the right orientation:
    # vertices space has its origin at the image center. Adding 1 to the
    # coordinates puts the origin at the top left.
    distance_to_top_left = np.sqrt(
        (hull_points[:, :, 0] + 1) ** 2 + (hull_points[:, :, 1] + 1) ** 2
    )
    bot_left_idx = np.argmin(distance_to_top_left) + 1
    hull_points = np.roll(hull_points, -bot_left_idx, axis=0)
    return hull_points


def perspective_transform_to_norm_corners(verts):
    norm_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
    return cv2.getPerspectiveTransform(verts, norm_corners)


def locate_surface(
    visible_markers, camera_model, registered_markers_undist, registered_markers_dist
):
    """Computes a Surface_Location based on a list of visible markers."""

    visible_registered_marker_ids = set(visible_markers.keys()) & set(
        registered_markers_undist.keys()
    )

    # If the surface is defined by 2+ markers, we require 2+ markers to be detected.
    # If the surface is defined by 1 marker, we require 1 marker to be detected.
    if not visible_registered_marker_ids or len(visible_registered_marker_ids) < min(
        2, len(registered_markers_undist)
    ):
        return Surface_Location(detected=False)

    visible_verts_dist = np.array(
        [visible_markers[id].verts_px for id in visible_registered_marker_ids]
    )
    registered_verts_undist = np.array(
        [
            registered_markers_undist[uid].verts_uv
            for uid in visible_registered_marker_ids
        ]
    )
    registered_verts_dist = np.array(
        [registered_markers_dist[uid].verts_uv for uid in visible_registered_marker_ids]
    )

    visible_verts_dist.shape = (-1, 2)
    registered_verts_undist.shape = (-1, 2)
    registered_verts_dist.shape = (-1, 2)

    dist_img_to_surf_trans, surf_to_dist_img_trans = find_homographies(
        registered_verts_dist, visible_verts_dist
    )

    visible_verts_undist = camera_model.undistort_points_on_image_plane(
        visible_verts_dist
    )
    img_to_surf_trans, surf_to_img_trans = find_homographies(
        registered_verts_undist, visible_verts_undist
    )

    return Surface_Location(
        True,
        dist_img_to_surf_trans,
        surf_to_dist_img_trans,
        img_to_surf_trans,
        surf_to_img_trans,
        len(visible_registered_marker_ids),
    )


def uniform_heatmap(resolution):
    if len(resolution) != 2:
        raise ValueError(
            "resolution has to be two dimensional but found dimension {}!".format(
                len(resolution)
            )
        )

    hm = np.zeros((*resolution, 4), dtype=np.uint8)
    hm[:, :, :3] = cv2.applyColorMap(hm[:, :, :3], cv2.COLORMAP_JET)
    hm[:, :, 3] = 125
    return hm


def placeholder_heatmap(resolution=(1, 1)):
    hm = np.zeros((*resolution, 4), dtype=np.uint8)
    hm[:, :, 3] = 175
    return hm


def map_to_surf(
    points,
    camera_model,
    img_to_surf_trans,
    dist_img_to_surf_trans,
    compensate_distortion=True,
    trans_matrix=None,
):
    """Map points from image pixel space to normalized surface space.

    Args:
        points (ndarray): An array of 2D points with shape (2,) or (N, 2).
        camera_model: Camera Model object.
        compensate_distortion: If `True` camera distortion will be correctly
        compensated using the `camera_model`. Note that points that lie outside
        of the image can not be undistorted correctly and the attempt may
        introduce a large error.
        trans_matrix: The transformation matrix defining the location of
        the surface. If `None`, the current transformation matrix saved in the
        Surface object will be used.

    Returns:
        ndarray: Points mapped into normalized surface space. Has the same shape
        as the input.

    """
    if trans_matrix is None:
        if compensate_distortion:
            trans_matrix = img_to_surf_trans
        else:
            trans_matrix = dist_img_to_surf_trans

    if compensate_distortion:
        orig_shape = points.shape
        points = camera_model.undistort_points_on_image_plane(points)
        points.shape = orig_shape

    points_on_surf = perspective_transform_points(points, trans_matrix)

    return points_on_surf


def map_from_surf(
    points,
    camera_model,
    surf_to_img_trans,
    surf_to_dist_img_trans,
    compensate_distortion=True,
    trans_matrix=None,
):
    """Map points from normalized surface space to image pixel space.

    Args:
        points (ndarray): An array of 2D points with shape (2,) or (N, 2).
        camera_model: Camera Model object.
        compensate_distortion: If `True` camera distortion will be correctly
        compensated using the `camera_model`. Note that points that lie outside
        of the image can not be undistorted correctly and the attempt may
        introduce a large error.
        trans_matrix: The transformation matrix defining the location of
        the surface. If `None`, the current transformation matrix saved in the
        Surface object will be used.

    Returns:
        ndarray: Points mapped into image space. Has the same shape
        as the input.

    """

    if trans_matrix is None:
        if compensate_distortion:
            trans_matrix = surf_to_img_trans
        else:
            trans_matrix = surf_to_dist_img_trans

    img_points = perspective_transform_points(points, trans_matrix)

    if compensate_distortion:
        orig_shape = points.shape
        img_points = camera_model.distortPoints(img_points)
        img_points.shape = orig_shape

    return img_points


def map_gaze_and_fixation_event(
    event,
    camera_model,
    img_to_surf_trans,
    dist_img_to_surf_trans,
    compensate_distortion,
    trans_matrix,
):
    """
    Map a list of gaze or fixation events onto the surface and return the
    corresponding list of gaze/fixation on surface events.

    Args:
        events: List of gaze or fixation events.
        camera_model: Camera Model object.
        trans_matrix: The transformation matrix defining the location of
        the surface. If `None`, the current transformation matrix saved in the
        Surface object will be used.

    Returns:
        List of gaze or fixation on surface events.

    """

    gaze_norm_pos = event["norm_pos"]
    gaze_img_point = methods.denormalize(
        gaze_norm_pos, camera_model.resolution, flip_y=True
    )
    gaze_img_point = np.array(gaze_img_point)

    surf_norm_pos = map_to_surf(
        points=gaze_img_point,
        camera_model=camera_model,
        img_to_surf_trans=img_to_surf_trans,
        dist_img_to_surf_trans=dist_img_to_surf_trans,
        compensate_distortion=compensate_distortion,
        trans_matrix=trans_matrix,
    )
    on_surf = bool((0 <= surf_norm_pos[0] <= 1) and (0 <= surf_norm_pos[1] <= 1))

    mapped_datum = {
        "topic": f"{event['topic']}_on_surface",
        "norm_pos": surf_norm_pos.tolist(),
        "confidence": event["confidence"],
        "on_surf": on_surf,
        "base_data": (event["topic"], event["timestamp"]),
        "timestamp": event["timestamp"],
    }
    if event["topic"] == "fixations":
        mapped_datum["id"] = event["id"]
        mapped_datum["duration"] = event["duration"]
        mapped_datum["dispersion"] = event["dispersion"]
    return mapped_datum


def add_marker(
    marker_uid: Surface_Marker_UID,
    verts_px,
    camera_model,
    markers: Surface_Marker_UID_To_Aggregate_Mapping,
    img_to_surf_trans,
    dist_img_to_surf_trans,
    compensate_distortion: bool,
    should_copy_markers: bool = True,
) -> Surface_Marker_UID_To_Aggregate_Mapping:
    if should_copy_markers:
        markers = markers.copy()
    surface_marker_dist = Surface_Marker_Aggregate(uid=marker_uid)
    marker_verts_dist = np.array(verts_px).reshape((4, 2))
    uv_coords_dist = map_to_surf(
        points=marker_verts_dist,
        camera_model=camera_model,
        img_to_surf_trans=img_to_surf_trans,
        dist_img_to_surf_trans=dist_img_to_surf_trans,
        compensate_distortion=compensate_distortion,
    )
    surface_marker_dist.add_observation(uv_coords_dist)
    markers[marker_uid] = surface_marker_dist
    return markers


def move_corner(
    corner_idx: int,
    new_pos,
    camera_model,
    marker_aggregate_mapping: Surface_Marker_UID_To_Aggregate_Mapping,
    img_to_surf_trans,
    dist_img_to_surf_trans,
    compensate_distortion: bool,
    should_copy_marker_aggregate_mapping: bool = True,
) -> Surface_Marker_UID_To_Aggregate_Mapping:
    """Update surface definition by moving one of the corners to a new position.

    Args:
        corner_idx: Identifier of the corner to be moved. The order of corners is
        `[(0, 0), (1, 0), (1, 1), (0, 1)]`
        new_pos: The updated position of the corner in image pixel coordinates.
        camera_model: Camera Model object.

    """
    if should_copy_marker_aggregate_mapping:
        marker_aggregate_mapping = marker_aggregate_mapping.copy()

    # Markers undistorted
    new_corner_pos = map_to_surf(
        points=new_pos,
        camera_model=camera_model,
        img_to_surf_trans=img_to_surf_trans,
        dist_img_to_surf_trans=dist_img_to_surf_trans,
        compensate_distortion=compensate_distortion,
    )
    old_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)

    new_corners = old_corners.copy()
    new_corners[corner_idx] = new_corner_pos

    transform = cv2.getPerspectiveTransform(new_corners, old_corners)
    for marker_aggregate in marker_aggregate_mapping.values():
        marker_aggregate.verts_uv = cv2.perspectiveTransform(
            marker_aggregate.verts_uv.reshape((-1, 1, 2)), transform
        ).reshape((-1, 2))

    return marker_aggregate_mapping
