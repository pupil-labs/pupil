"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import cv2
import numpy as np
from methods import dist_pts_ellipse, normalize
from numpy import linalg as LA


class CircleTracker:
    def __init__(self, wait_interval=30, roi_wait_interval=120):
        self.wait_interval = wait_interval
        self.roi_wait_interval = roi_wait_interval
        self._previous_markers = []
        self._predict_motion = []
        self._wait_count = 0
        self._roi_wait_count = 0
        self._flag_check = False
        self._flag_check_roi = False
        self._world_size = None

    def update(self, img):
        """
        Decide whether to track the marker in the roi or in the whole frame
        Return all detected markers

        :param img: input gray image
        :type img: numpy.ndarray
        :return: all detected markers including the information about their ellipses, center positions and their type
        (Ref/Stop)
        :rtype: a list containing dictionary with keys: 'ellipses', 'img_pos', 'norm_pos', 'marker_type'
        """
        img_size = img.shape[::-1]
        if self._world_size is None:
            self._world_size = img_size
        elif self._world_size != img_size:
            self._previous_markers = []
            self._predict_motion = []
            self._wait_count = 0
            self._roi_wait_count = 0
            self._world_size = img_size

        if self._wait_count <= 0 or self._roi_wait_count <= 0:
            self._flag_check = True
            self._flag_check_roi = False
            self._wait_count = self.wait_interval
            self._roi_wait_count = self.roi_wait_interval

        markers = []
        if self._flag_check:
            markers = self._check_frame(img)
            predict_motion = []
            if len(markers) > 0:
                if len(self._previous_markers) in (0, len(markers)):
                    self._flag_check = True
                    self._flag_check_roi = True
                    self._roi_wait_count -= 1
                    for i in range(len(self._previous_markers)):
                        predict_motion.append(
                            np.array(markers[i]["img_pos"])
                            - np.array(self._previous_markers[i]["img_pos"])
                        )
            else:
                if self._flag_check_roi:
                    self._flag_check = True
                    self._flag_check_roi = False
                else:
                    self._flag_check = False
                    self._flag_check_roi = False

        self._wait_count -= 1
        self._previous_markers = markers
        return markers

    def _check_frame(self, img):
        """
        Track the markers in the ROIs / in the whole frame

        :param img: input gray image
        :type img: numpy.ndarray
        :return: all detected markers including the information about their ellipses, center positions and their type
        (Ref/Stop)
        :rtype: a list containing dictionary with keys: 'ellipses', 'img_pos', 'norm_pos', 'marker_type'
        """
        img_size = img.shape[::-1]
        scale = 0.5 if img_size[0] >= 1280 else 640 / img_size[0]

        marker_list = []
        # Check whole frame
        if not self._flag_check_roi:
            ellipses_list = find_pupil_circle_marker(img, scale)

            # Save the markers in dictionaries
            for ellipses_ in ellipses_list:
                ellipses = ellipses_["ellipses"]
                img_pos = ellipses[0][0]
                norm_pos = normalize(img_pos, img_size, flip_y=True)
                marker_list.append(
                    {
                        "ellipses": ellipses,
                        "img_pos": img_pos,
                        "norm_pos": norm_pos,
                        "marker_type": ellipses_["marker_type"],
                    }
                )

        # Check roi
        else:
            for i in range(len(self._previous_markers)):
                largest_ellipse = self._previous_markers[i]["ellipses"][-1]

                # Set up the boundary of the roi
                if self._predict_motion:
                    predict_center = (
                        largest_ellipse[0][0] + self._predict_motion[i][0],
                        largest_ellipse[0][1] + self._predict_motion[i][1],
                    )
                    b0 = (
                        predict_center[0]
                        - largest_ellipse[1][1]
                        - abs(self._predict_motion[i][0]) * 2
                    )
                    b1 = (
                        predict_center[0]
                        + largest_ellipse[1][1]
                        + abs(self._predict_motion[i][0]) * 2
                    )
                    b2 = (
                        predict_center[1]
                        - largest_ellipse[1][0]
                        - abs(self._predict_motion[i][1]) * 2
                    )
                    b3 = (
                        predict_center[1]
                        + largest_ellipse[1][0]
                        + abs(self._predict_motion[i][1]) * 2
                    )
                else:
                    predict_center = largest_ellipse[0]
                    b0 = predict_center[0] - largest_ellipse[1][1]
                    b1 = predict_center[0] + largest_ellipse[1][1]
                    b2 = predict_center[1] - largest_ellipse[1][0]
                    b3 = predict_center[1] + largest_ellipse[1][0]

                b0 = 0 if b0 < 0 else int(b0)
                b1 = img_size[0] - 1 if b1 > img_size[0] - 1 else int(b1)
                b2 = 0 if b2 < 0 else int(b2)
                b3 = img_size[1] - 1 if b3 > img_size[1] - 1 else int(b3)
                col_slice = b0, b1
                row_slice = b2, b3

                ellipses_list = find_pupil_circle_marker(
                    img[slice(*row_slice), slice(*col_slice)], scale
                )

                # Track the marker which was detected last frame;
                # To avoid more than one markers are detected in one ROI
                if len(ellipses_list):
                    if len(ellipses_list) == 1:
                        right_ellipses = ellipses_list[0]
                    else:
                        pre_pos = np.array(
                            (
                                self._previous_markers[i]["img_pos"][0] - b0,
                                self._previous_markers[i]["img_pos"][1] - b2,
                            )
                        )
                        temp_dist = [
                            LA.norm(e["ellipses"][0][0] - pre_pos)
                            for e in ellipses_list
                        ]
                        right_ellipses = ellipses_list[temp_dist.index(min(temp_dist))]
                    ellipses = [
                        ((e[0][0] + b0, e[0][1] + b2), e[1], e[2])
                        for e in right_ellipses["ellipses"]
                    ]
                    img_pos = ellipses[0][0]
                    norm_pos = normalize(img_pos, img_size, flip_y=True)
                    # Save the marker in dictionary
                    marker_list.append(
                        {
                            "ellipses": ellipses,
                            "img_pos": img_pos,
                            "norm_pos": norm_pos,
                            "marker_type": right_ellipses["marker_type"],
                        }
                    )

        return marker_list


def find_pupil_circle_marker(img, scale):
    img_size = img.shape[::-1]
    # Resize the image
    img_resize = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)

    # Use three kinds of adaptive threshold to extract the edges of the image
    # The first one is for complicated scene
    # The Second one is for normal scene
    # The last one is for marker in low contrast
    img_resize_blur = cv2.GaussianBlur(img_resize, (3, 3), 0.25)
    edges = [
        cv2.adaptiveThreshold(
            img_resize_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            29,
            36,
        ),
        cv2.adaptiveThreshold(
            img_resize_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            29,
            18,
        ),
        cv2.adaptiveThreshold(
            img_resize_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            29,
            3,
        ),
    ]

    ellipses_list = []
    found_pos = []
    found_size = []
    for i in range(len(edges)):
        edge = edges[i]
        circle_clusters = find_concentric_circles(
            edge,
            None,
            None,
            found_pos,
            found_size,
            first_check=True,
            min_ellipses_num=2,
        )

        for ellipses, boundary in circle_clusters:
            ellipse_pos = np.array(ellipses[0][0])
            ellipse_size = min(ellipses[-1][1])
            # Discard duplicates
            duplicates = [
                k
                for k in range(len(found_pos))
                if LA.norm(ellipse_pos - found_pos[k]) < found_size[k]
            ]
            if len(duplicates) > 0:
                continue

            # Set up the boundary of the ellipses
            b0, b1 = boundary[0][0] / scale - 2, boundary[0][1] / scale + 2
            b2, b3 = boundary[1][0] / scale - 2, boundary[1][1] / scale + 2

            b0 = 0 if b0 < 0 else int(b0)
            b1 = img_size[0] - 1 if b1 > img_size[0] - 1 else int(b1)
            b2 = 0 if b2 < 0 else int(b2)
            b3 = img_size[1] - 1 if b3 > img_size[1] - 1 else int(b3)

            col_slice = b0, b1
            row_slice = b2, b3
            img_ellipse = img[slice(*row_slice), slice(*col_slice)]
            img_ellipse_size = img_ellipse.shape[::-1]
            if not min(img_ellipse_size):
                continue
            # Calculate the brightness within and outside the ring
            img_median = np.median(img_ellipse)
            darker_peak = np.ma.median(
                np.ma.array(img_ellipse, mask=img_ellipse > img_median)
            )
            brighter_peak = np.ma.median(
                np.ma.array(img_ellipse, mask=img_ellipse < img_median)
            )
            img_contrast = brighter_peak - darker_peak

            # Calculate the kernel_size for edge extraction
            block_size = max(5, int(max(img_ellipse_size) / 4) * 2 - 1)
            c = img_contrast / 128

            # Extract the edges of the candidate marker again with more appropriate kernel_size
            img_ellipse_blur = cv2.GaussianBlur(img_ellipse, (3, 3), 1)
            mask_edge = cv2.adaptiveThreshold(
                img_ellipse_blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                c,
            )
            temp = find_concentric_circles(
                mask_edge,
                scale,
                img_contrast,
                None,
                None,
                first_check=False,
                min_ellipses_num=3,
            )

            if len(temp) == 0:
                continue
            single_marker = temp[0][0]
            if (
                len(single_marker) > 3
                and sum(single_marker[2][1]) / sum(single_marker[0][1]) < 9
            ):
                single_marker = [single_marker[0], single_marker[1], single_marker[2]]

            # Get the ellipses of the dot and the ring
            outer_ellipse = single_marker[-1]
            inner_ellipse = single_marker[-2]
            dot_ellipse = single_marker[-3]
            # Calculate the ring ratio and dot ratio
            ring_ratio = sum(outer_ellipse[1]) / sum(inner_ellipse[1])
            dot_ratio = sum(outer_ellipse[1]) / sum(dot_ellipse[1])

            # Check if it is a Ref / stop marker by the mean gray scale of the ring
            mask_ring = np.ones_like(img_ellipse) * 255
            cv2.ellipse(mask_ring, outer_ellipse, color=(0, 0, 0), thickness=-1)
            cv2.ellipse(mask_ring, inner_ellipse, color=(255, 255, 255), thickness=-1)
            mask_ring_mean = np.ma.array(mask_edge, mask=mask_ring).mean()

            mask_outer = np.zeros_like(img_ellipse)
            cv2.ellipse(mask_outer, outer_ellipse, color=(255, 255, 255), thickness=-1)
            outer_mean = np.ma.array(img_ellipse, mask=mask_outer).mean()
            ring_median = np.ma.median(np.ma.array(img_ellipse, mask=mask_ring))

            # Check if it is a Ref marker
            if mask_ring_mean >= 128:
                # Check the ring ratio and dot ratio
                if not 1.2 < ring_ratio < 2 or not 2.5 < dot_ratio < 6:
                    continue

                # The gray scale of the outer part of the ring should be brighter than the gray scale of the ring
                if outer_mean - ring_median < 0:
                    continue

                mask_middle = np.ones_like(img_ellipse) * 255
                cv2.ellipse(mask_middle, inner_ellipse, color=(0, 0, 0), thickness=-1)
                cv2.ellipse(
                    mask_middle, inner_ellipse, color=(255, 255, 255), thickness=2
                )
                cv2.ellipse(
                    mask_middle, dot_ellipse, color=(255, 255, 255), thickness=-1
                )
                mask_middle_value = np.ma.array(img_ellipse, mask=mask_middle)
                middle_median = np.ma.median(mask_middle_value)
                # The gray scale of the part between the ring and the dot should be brighter than the gray scale of the ring
                if middle_median - ring_median < img_contrast / 4:
                    continue

                # The std of the part between the ring and the dot should not be too large
                if len(np.where(mask_middle == 0)[0]) > 15:
                    middle_std = mask_middle_value.std()
                    if middle_std > img_contrast / 2:
                        continue

                single_marker = [
                    ((e[0][0] + b0, e[0][1] + b2), e[1], e[2]) for e in single_marker
                ]
                ellipses_list.append({"ellipses": single_marker, "marker_type": "Ref"})
                found_pos.append(ellipse_pos)
                found_size.append(ellipse_size)

            # Check if it is a stop marker
            else:
                # Check the ring ratio and dot ratio
                if not 1.3 < ring_ratio < 2.1 or not 2.5 < dot_ratio < 5:
                    continue

                # The gray scale of the outer part of the ring should be darker than the gray scale of the ring
                if ring_median - outer_mean < 0:
                    continue

                mask_middle = np.ones_like(img_ellipse) * 255
                cv2.ellipse(mask_middle, inner_ellipse, color=(0, 0, 0), thickness=-1)
                cv2.ellipse(
                    mask_middle, inner_ellipse, color=(255, 255, 255), thickness=1
                )
                cv2.ellipse(
                    mask_middle, dot_ellipse, color=(255, 255, 255), thickness=-1
                )
                mask_middle_value = np.ma.array(img_ellipse, mask=mask_middle)
                middle_median = np.ma.median(mask_middle_value)
                # The gray scale of the part between the ring and the dot should be darker than the gray scale of the ring
                if ring_median - middle_median < img_contrast / 4:
                    continue

                # The std of the part between the ring and the dot should not be too large
                if len(np.where(mask_middle == 0)[0]) > 15:
                    middle_std = mask_middle_value.std()
                    if middle_std > img_contrast / 2:
                        continue

                single_marker = [
                    ((e[0][0] + b0, e[0][1] + b2), e[1], e[2]) for e in single_marker
                ]
                ellipses_list.append({"ellipses": single_marker, "marker_type": "Stop"})
                found_pos.append(ellipse_pos)
                found_size.append(ellipse_size)

    return ellipses_list


def find_concentric_circles(
    edge,
    scale,
    img_contrast,
    found_pos,
    found_size,
    first_check=True,
    min_ellipses_num=2,
):
    if first_check:
        concentric_circle_clusters = []

        # OpenCV version compatibility note:
        # - Opencv3 returns three results: image, contours, hierarchy
        # - Opencv4 returns two results: contours, hierarchy
        # We do not use `image` in any case. Therefore, we can ensure
        # compatibility by using `*_`
        *_, contours, hierarchy = cv2.findContours(
            edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_KCOS
        )
        # We use CHAIN_APPROX_TC89_KCOS because it is faster than
        # the default method CV_CHAIN_APPROX_NONE

        if contours is None or hierarchy is None:
            return []
        clusters = get_nested_clusters(contours, hierarchy[0], min_ellipses_num)
        # speed up code by caching computed ellipses
        ellipses = {}

        for cluster in clusters:
            candidate_ellipses = []
            first_ellipse = True
            for i in cluster:
                c = contours[i]
                if len(c) > 100:
                    continue
                if i in ellipses:
                    e, fit = ellipses[i]
                else:
                    if len(c) >= 5:
                        e = cv2.fitEllipse(c)
                        # Discard duplicates
                        if first_ellipse:
                            duplicates = [
                                k
                                for k in range(len(found_pos))
                                if LA.norm(e[0] - found_pos[k])
                                < found_size[k] + min(e[1])
                            ]
                            if len(duplicates) > 0:
                                ellipses[i] = e, 100.0
                                break
                            fit = 0
                        else:
                            fit = max(dist_pts_ellipse(e, c)) if min(e[1]) else 0.0
                        e = e if min(e[1]) else (e[0], (0.1, 0.1), e[2])
                    else:
                        e_center = (
                            float(c[len(c) // 2][0][0]),
                            float(c[len(c) // 2][0][1]),
                        )
                        e = (e_center, (0.1, 0.1), 0.0)
                        # Discard duplicates
                        if first_ellipse:
                            duplicates = [
                                k
                                for k in range(len(found_pos))
                                if LA.norm(e_center - found_pos[k]) < found_size[k] + 1
                            ]
                            if len(duplicates) > 0:
                                ellipses[i] = e, 100.0
                                break
                        fit = 0

                    ellipses[i] = e, fit

                # Discard the contour which does not fit the ellipse so well
                if first_ellipse or fit < max(1, max(e[1]) / 50):
                    e = (e[0], e[1], e[2], i)
                    candidate_ellipses.append(e)
                first_ellipse = False

            # Discard false positives
            if len(candidate_ellipses) < min_ellipses_num:
                continue

            # Discard the ellipses whose center is far away from the center of the second innermost ellipse
            cluster_center = np.array(candidate_ellipses[1][0])
            if max(candidate_ellipses[-1][1]) > 200:
                candidate_ellipses = [
                    e
                    for e in candidate_ellipses
                    if LA.norm(e[0] - cluster_center) < max(e[1]) / 5
                ]
            elif max(candidate_ellipses[-1][1]) > 100:
                candidate_ellipses = [
                    e
                    for e in candidate_ellipses
                    if LA.norm(e[0] - cluster_center) < max(e[1]) / 10
                ]
            else:
                candidate_ellipses = [
                    e
                    for e in candidate_ellipses
                    if LA.norm(e[0] - cluster_center) < max(max(e[1]) / 20, 3)
                ]

            # Discard false positives
            if len(candidate_ellipses) < min_ellipses_num:
                continue

            c = contours[candidate_ellipses[-1][3]]
            boundary = (
                (np.amin(c, axis=0)[0][0], np.amax(c, axis=0)[0][0]),
                (np.amin(c, axis=0)[0][1], np.amax(c, axis=0)[0][1]),
            )

            candidate_ellipses = [(e[0], e[1], e[2]) for e in candidate_ellipses]
            concentric_circle_clusters.append((candidate_ellipses, boundary))

        # Return clusters sorted by the number of ellipses and the size of largest ellipse
        return sorted(
            concentric_circle_clusters, key=lambda x: (-len(x[0]), -max(x[0][-1][1]))
        )

    else:
        *_, contours, hierarchy = cv2.findContours(
            edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
        )

        if contours is None or hierarchy is None:
            return []
        clusters = get_nested_clusters(contours, hierarchy[0], min_ellipses_num)
        # speed up code by caching computed ellipses
        ellipses = {}

        for cluster in clusters:
            candidate_ellipses = []
            first_ellipse = True
            for i in cluster:
                c = contours[i]
                if i in ellipses:
                    e, fit = ellipses[i]
                else:
                    if len(c) >= 5:
                        e = cv2.fitEllipse(c)
                        fit = max(dist_pts_ellipse(e, c)) if min(e[1]) else 0.0
                        if min(e[1]) == 0:
                            e = (e[0], (e[1][0] + 1.0, e[1][1] + 1.0), e[2])
                    else:
                        fit = 0
                        e_center = (
                            float(c[len(c) // 2][0][0]),
                            float(c[len(c) // 2][0][1]),
                        )
                        e_size = (
                            float(abs(c[-1][0][0] - c[0][0][0]) + 1),
                            float(abs(c[-1][0][1] - c[0][0][1]) + 1),
                        )
                        e = (e_center, e_size, 0)
                    ellipses[i] = e, fit
                # Discard the contour which does not fit the ellipse so well
                if first_ellipse:
                    fit_thres = 0.5 + (256 - img_contrast) / 256
                else:
                    if img_contrast <= 96:
                        fit_thres = max(e[1]) * scale / 10 + (256 - img_contrast) / 256
                    else:
                        fit_thres = max(0.5, max(e[1]) * scale / 10)

                if fit < fit_thres:
                    candidate_ellipses.append(e)
                    if len(candidate_ellipses) == 4:
                        break
                first_ellipse = False

            # Discard false positives
            if len(candidate_ellipses) < min_ellipses_num:
                continue

            # Discard the ellipses whose center is far away from the center of the innermost ellipse
            cluster_center = np.array(candidate_ellipses[0][0])
            if max(candidate_ellipses[-1][1]) * scale > 200:
                candidate_ellipses = [
                    e
                    for e in candidate_ellipses
                    if LA.norm(e[0] - cluster_center) < max(e[1]) / 5
                ]
            elif max(candidate_ellipses[-1][1]) * scale > 100:
                candidate_ellipses = [
                    e
                    for e in candidate_ellipses
                    if LA.norm(e[0] - cluster_center) < max(e[1]) / 10
                ]
            else:
                candidate_ellipses = [
                    e
                    for e in candidate_ellipses
                    if LA.norm(e[0] - cluster_center) < max(max(e[1]) / 20, 2)
                ]

            # Discard false positives
            if len(candidate_ellipses) < min_ellipses_num:
                continue

            return [(candidate_ellipses, [[0, 0], [0, 0]])]

    return []


def add_parents(child, graph, family):
    family.append(child)
    parent = graph[child, -1]
    if parent != -1:
        family = add_parents(parent, graph, family)
    return family


def get_nested_clusters(contours, hierarchy, min_nested_count):
    clusters = {}
    # nesting of countours where many children are grouping in a single parent happens a lot (your screen with stuff in it. A page with text...)
    # we create a cluster for each of these children.
    # to reduce CPU load we only keep the biggest cluster for clusters where the innermost parent is the same.
    for i in np.where(hierarchy[:, 2] == -1)[0]:  # contours with no children
        cluster = add_parents(i, hierarchy, [])
        # is this cluster bigger that the current contender in the innermost parent group if if already exists?
        if min_nested_count <= len(cluster) > len(clusters.get(cluster[1], [])):
            clusters[cluster[1]] = cluster
    return clusters.values()


def getEllipsePts(e, num_pts=10):
    c1 = e[0][0]
    c2 = e[0][1]
    a = e[1][0]
    b = e[1][1]
    angle = e[2]

    steps = np.linspace(0, 2 * np.pi, num=num_pts, endpoint=False)
    rot = cv2.getRotationMatrix2D((0, 0), -angle, 1)

    pts1 = a / 2.0 * np.cos(steps)
    pts2 = b / 2.0 * np.sin(steps)
    pts = np.column_stack((pts1, pts2, np.ones(pts1.shape[0])))

    pts_rot = np.matmul(rot, pts.T)
    pts_rot = pts_rot.T

    pts_rot[:, 0] += c1
    pts_rot[:, 1] += c2

    return pts_rot


def marker_3d_pose(marker, cam_model, marker_diameter=7.6):
    target_circle = [[0, 0], [marker_diameter, marker_diameter], 0]
    target_pts = getEllipsePts(target_circle)
    target_pts3D = np.zeros(
        (target_pts.shape[0], target_pts.shape[1] + 1), dtype=np.float32
    )
    target_pts3D[:, :-1] = target_pts
    target_pts3D.shape = -1, 1, 3

    e = marker["ellipses"][-1]

    pts = getEllipsePts(e)
    pts.shape = -1, 1, 2
    pts = pts.astype("float32")

    _, rot, trans = cam_model.solvePnP(target_pts3D, pts.astype("float32"))

    return trans, rot


if __name__ == "__main__":

    def bench():
        import cv2

        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        for x in range(100):
            sts, img = cap.read()
            # img = cv2.imread('/Users/mkassner/Desktop/manual_calibration_marker-01.png')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(len(find_concentric_circles(gray, visual_debug=img)))
            cv2.imshow("img", img)
            cv2.waitKey(1)
            # return

    import cProfile
    import os
    import subprocess

    cProfile.runctx("bench()", {}, locals(), "world.pstats")
    loc = os.path.abspath(__file__).rsplit("pupil_src", 1)
    gprof2dot_loc = os.path.join(loc[0], "pupil_src", "shared_modules", "gprof2dot.py")
    subprocess.call(
        "python "
        + gprof2dot_loc
        + " -f pstats world.pstats | dot -Tpng -o world_cpu_time.png",
        shell=True,
    )
    print(
        "created  time graph for  process. Please check out the png next to this file"
    )
