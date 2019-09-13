import typing
import logging

import cv2
import numpy as np

import methods

from .surface_location import Surface_Location


logger = logging.getLogger(__name__)


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
    visible_markers,
    camera_model,
    registered_markers_undist,
    registered_markers_dist,
):
    """Computes a Surface_Location based on a list of visible markers."""

    visible_registered_marker_ids = set(visible_markers.keys()) & set(
        registered_markers_undist.keys()
    )

    # If the surface is defined by 2+ markers, we require 2+ markers to be detected.
    # If the surface is defined by 1 marker, we require 1 marker to be detected.
    if not visible_registered_marker_ids or len(
        visible_registered_marker_ids
    ) < min(2, len(registered_markers_undist)):
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
        [
            registered_markers_dist[uid].verts_uv
            for uid in visible_registered_marker_ids
        ]
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
        raise ValueError("resolution has to be two dimensional but found dimension {}!".format(len(resolution)))

    hm = np.zeros((*resolution, 4), dtype=np.uint8)
    hm[:, :, :3] = cv2.applyColorMap(hm[:, :, :3], cv2.COLORMAP_JET)
    hm[:, :, 3] = 125
    return hm


def placeholder_heatmap(resolution=(1, 1)):
    hm = np.zeros((*resolution, 4), dtype=np.uint8)
    hm[:, :, 3] = 175
    return hm
