"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import numpy as np
from numpy import linalg as LA
import cv2
from methods import dist_pts_ellipse, normalize


class CircleTracker(object):
    def __init__(self, wait_interval=30):
        self.wait_interval = wait_interval
        self.previous_markers = []
        self.predict_motion = []
        self.wait_count = 0
        self.roi_wait_count = 0
        self.flag_check = False
        self.flag_check_roi = False

    def update(self, img):
        """
        Decide whether to track the marker in the roi or in the whole frame
        Return all detected markers
        """
        if self.wait_count <= 0 or self.roi_wait_count <= 0:
            self.flag_check = True
            self.flag_check_roi = False
            self.wait_count = self.wait_interval
            self.roi_wait_count = self.wait_interval * 4

        markers = []
        if self.flag_check:
            markers = self.check_frame(img)
            predict_motion = []
            if len(markers) > 0:
                if len(self.previous_markers) in (0, len(markers)):
                    self.flag_check = True
                    self.flag_check_roi = True
                    self.roi_wait_count -= 1
                    for i in range(len(self.previous_markers)):
                        predict_motion.append(np.array(markers[i]['img_pos']) - np.array(self.previous_markers[i]['img_pos']))
            else:
                if self.flag_check_roi:
                    self.flag_check = True
                    self.flag_check_roi = False
                else:
                    self.flag_check = False
                    self.flag_check_roi = False

        self.wait_count -= 1
        self.previous_markers = markers
        return markers

    def check_frame(self, img):
        """
        Resize the image and then track the markers in the ROIs / in the whole frame
        Return all detected markers including the information about their ellipses, center positions and
        whether they are stop markers
        """
        img_size = img.shape[::-1]
        scale = 0.5 if img_size[0] >= 1280 else 640 / img_size[0]

        marker_list = []
        # Check whole frame
        if not self.flag_check_roi:
            ellipses_list = find_pupil_circle_marker(img, scale)

            # Save the markers in dictionaries
            for ellipses_ in ellipses_list:
                ellipses = ellipses_['ellipses']
                img_pos = ellipses[0][0]
                norm_pos = normalize(img_pos, img_size, flip_y=True)
                marker_list.append({'ellipses': ellipses, 'img_pos': img_pos, 'norm_pos': norm_pos,
                                    'stop_marker': ellipses_['stop_marker']})

        # Check roi
        else:
            for i in range(len(self.previous_markers)):
                largest_ellipse = self.previous_markers[i]['ellipses'][-1]

                # Set up the boundary of the roi
                if self.predict_motion:
                    predict_center = (largest_ellipse[0][0] + self.predict_motion[i][0],
                                      largest_ellipse[0][1] + self.predict_motion[i][1])
                    b0 = predict_center[0] - largest_ellipse[1][1] - abs(self.predict_motion[i][0]) * 2
                    b1 = predict_center[0] + largest_ellipse[1][1] + abs(self.predict_motion[i][0]) * 2
                    b2 = predict_center[1] - largest_ellipse[1][0] - abs(self.predict_motion[i][1]) * 2
                    b3 = predict_center[1] + largest_ellipse[1][0] + abs(self.predict_motion[i][1]) * 2
                else:
                    predict_center = largest_ellipse[0]
                    b0 = predict_center[0] - largest_ellipse[1][1] * 3
                    b1 = predict_center[0] + largest_ellipse[1][1] * 3
                    b2 = predict_center[1] - largest_ellipse[1][0] * 3
                    b3 = predict_center[1] + largest_ellipse[1][0] * 3

                b0 = 0 if b0 < 0 else int(b0)
                b1 = img_size[0] - 1 if b1 > img_size[0] - 1 else int(b1)
                b2 = 0 if b2 < 0 else int(b2)
                b3 = img_size[1] - 1 if b3 > img_size[1] - 1 else int(b3)
                col_slice = b0, b1
                row_slice = b2, b3

                ellipses_list = find_pupil_circle_marker(img[slice(*row_slice), slice(*col_slice)], scale)

                # Save the markers in dictionaries
                for ellipses_ in ellipses_list:
                    ellipses = [((e[0][0]+b0, e[0][1]+b2), e[1], e[2]) for e in ellipses_['ellipses']]
                    img_pos = ellipses[0][0]
                    norm_pos = normalize(img_pos, img_size, flip_y=True)
                    marker_list.append({'ellipses': ellipses, 'img_pos': img_pos, 'norm_pos': norm_pos,
                                        'stop_marker': ellipses_['stop_marker']})

        return marker_list


def find_pupil_circle_marker(img, scale):
    """
    :param img: gray image
    :param scale: the ratio of 640 to the width of the img
    :return: all detected markers with the information of their ellipses and whether a stop marker or not
    """
    img_size = img.shape[::-1]
    # Resize the image
    img_resize = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)

    # Use two kinds of adaptive threshold to extract the edges of the image
    # The first one is for normal and complicated scene
    # The Second one is for marker in low contrast
    img_resize_blur = cv2.GaussianBlur(img_resize, (3, 3), 0)
    edges = [cv2.adaptiveThreshold(img_resize_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 29, 35),
             cv2.adaptiveThreshold(img_resize_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 29, 3)]

    ellipses_list = []
    marker_found_pos = []
    for edge in edges:
        circle_clusters = find_concentric_circles(edge, first_check=True, min_ellipses_num=2)

        for ellipses, boundary in circle_clusters:
            img_pos = ellipses[0][0]
            # Discard duplicates
            if len(marker_found_pos) and min((abs(np.array(img_pos) - np.array(marker_found_pos))).max(axis=1)) < max(ellipses[1][1]):
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
            img_ellipse_mean = np.mean(img_ellipse)

            # Calculate the kernel_size for edge extraction
            block_size = max(5, int(max(img_ellipse_size) / 4) * 2 - 1)
            c = img_ellipse_mean / 64

            # Extract the edges of the candidate marker again with more appropriate kernel_size
            img_ellipse_blur = cv2.GaussianBlur(img_ellipse, (3, 3), 1)
            mask_outer = cv2.adaptiveThreshold(img_ellipse_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)
            temp = find_concentric_circles(mask_outer, first_check=False, min_ellipses_num=3)

            if len(temp) == 0:
                continue

            single_marker = temp[0][0]

            # Get the ellipses of the dot and the ring
            divider_size = sum(single_marker[-1][1]) * 0.2
            larger_ellipses = [e for e in single_marker if sum(e[1]) >= divider_size]
            if len(larger_ellipses) != 3:
                continue
            dot_ellipse = larger_ellipses[0]
            inner_ellipse = larger_ellipses[1]
            outer_ellipse = larger_ellipses[2]

            # Calculate the ring ratio
            ring_ratio = sum(outer_ellipse[1]) / sum(inner_ellipse[1])

            # Check if it is a normal / stop marker by the mean grayscale of the ring and the dot
            mask_ring_dot = np.ones_like(img_ellipse) * 255
            cv2.ellipse(mask_ring_dot, outer_ellipse, color=(0, 0, 0), thickness=-1)
            cv2.ellipse(mask_ring_dot, inner_ellipse, color=(255, 255, 255), thickness=-1)
            cv2.ellipse(mask_ring_dot, dot_ellipse, color=(0, 0, 0), thickness=-1)
            mask_ring_dot_mean = np.ma.array(mask_outer, mask=mask_ring_dot).mean()

            # Check if it is a normal marker
            if mask_ring_dot_mean >= 128:
                # Check the ring ratio
                if not 1.2 < ring_ratio < 2.1:
                    continue

                outer_mean = np.ma.array(img_ellipse, mask=mask_outer).mean()

                ring_dot_median = np.ma.median(np.ma.array(img_ellipse, mask=mask_ring_dot))

                # The grayscale of the outer part of the ring should be brighter than the grayscale of the ring
                if outer_mean - ring_dot_median < img_ellipse_mean / 4:
                    continue

                mask_middle = np.ones_like(img_ellipse) * 255
                cv2.ellipse(mask_middle, inner_ellipse, color=(0, 0, 0), thickness=-1)
                cv2.ellipse(mask_middle, inner_ellipse, color=(255, 255, 255), thickness=2)
                cv2.ellipse(mask_middle, dot_ellipse, color=(255, 255, 255), thickness=-1)
                mask_middle_value = np.ma.array(img_ellipse, mask=mask_middle)
                middle_median = np.ma.median(mask_middle_value)

                # The grayscale of the part between the ring and the dot should be brighter than the grayscale of the ring
                if middle_median - ring_dot_median < img_ellipse_mean / 4:
                    continue

                middle_std = mask_middle_value.std()
                white_median = np.ma.median(np.ma.array(img_ellipse, mask=mask_outer))
                # The std of the part between the ring and the dot should not be too large
                if middle_std / white_median > 0.4:
                    continue

                single_marker = [((e[0][0]+b0, e[0][1]+b2), e[1], e[2]) for e in single_marker]
                ellipses_list.append({'ellipses': single_marker, 'stop_marker': False})
                marker_found_pos.append(img_pos)
                return ellipses_list

            # Check if it is a stop marker
            else:
                # Check the ring ratio
                if not 1.4 < ring_ratio < 2.7:
                    continue

                outer_mean = np.ma.array(img_ellipse, mask=cv2.bitwise_not(mask_outer)).mean()

                # cv2.ellipse(mask_ring_dot, outer_ellipse, color=(255, 255, 255), thickness=1)
                ring_dot_median = np.ma.median(np.ma.array(img_ellipse, mask=mask_ring_dot))

                # The grayscale of the outer part of the ring should be darker than the grayscale of the ring
                if ring_dot_median - outer_mean < img_ellipse_mean / 4:
                    continue

                mask_middle = np.ones_like(img_ellipse)*255
                cv2.ellipse(mask_middle, inner_ellipse, color=(0, 0, 0), thickness=-1)
                cv2.ellipse(mask_middle, inner_ellipse, color=(255, 255, 255), thickness=1)
                cv2.ellipse(mask_middle, dot_ellipse, color=(255, 255, 255), thickness=-1)
                mask_middle_value = np.ma.array(img_ellipse, mask=mask_middle)
                middle_median = np.ma.median(mask_middle_value)

                # The grayscale of the part between the ring and the dot should be darker than the grayscale of the ring
                if ring_dot_median - middle_median < img_ellipse_mean / 4:
                    continue

                middle_std = mask_middle_value.std()
                white_median = np.ma.median(np.ma.array(img_ellipse, mask=mask_outer))

                # The std of the part between the ring and the dot should not be too large
                if middle_std / white_median > 0.4:
                    continue

                single_marker = [((e[0][0]+b0, e[0][1]+b2), e[1], e[2]) for e in single_marker]
                ellipses_list.append({'ellipses': single_marker, 'stop_marker': True})
                marker_found_pos.append(img_pos)
                return ellipses_list

    return ellipses_list


def find_concentric_circles(edge, first_check=True, min_ellipses_num=2):
    """
    :param edge: the edge extraction of the image
    :param first_check: if it is the first time to find contours
    :param min_ellipses_num: minimum requirement of the number of the ellipses in the marker
    :return: all candidate markers
    """

    if first_check:
        concentric_circle_clusters = []
        _, contours, hierarchy = cv2.findContours(edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE, offset=(0, 0)) # TC89_KCOS

        if contours is None or hierarchy is None:
            return []
        clusters = get_nested_clusters(contours, hierarchy[0], min_ellipses_num)
        # speed up code by caching computed ellipses
        ellipses = {}

        for cluster in clusters:
            candidate_ellipses = []
            if len(cluster) < min_ellipses_num:
                continue
            first_ellipse = True
            for i in cluster:
                c = contours[i]
                if i in ellipses:
                    e, fit = ellipses[i]
                else:
                    if len(c) >= 5:
                        e = cv2.fitEllipse(c)
                        fit = max(dist_pts_ellipse(e, c)) if min(e[1]) else 0
                        e = e if min(e[1]) else (e[0], (1, 1), e[2])
                    else:
                        fit = 0
                        center = c[len(c)//2][0]
                        e = ((center[0], center[1]), (1, 1), 0)

                    ellipses[i] = e, fit

                # Discard the contour which does not fit the ellipse so well
                if first_ellipse or fit < max(1, max(e[1]) / 50):
                    e = (e[0], e[1], e[2], i)
                    candidate_ellipses.append(e)
                first_ellipse = False

            # Discard false positives
            if len(candidate_ellipses) < min_ellipses_num or sum(candidate_ellipses[-1][1]) < 5:
                continue

            # Discard the ellipses whose center is far away from the center of the innermost ellipse
            cluster_center = np.array(candidate_ellipses[0][0])
            if max(candidate_ellipses[-1][1]) > 200:
                candidate_ellipses = [e for e in candidate_ellipses if LA.norm(e[0] - cluster_center) < max(e[1]) / 5]
            elif max(candidate_ellipses[-1][1]) > 100:
                candidate_ellipses = [e for e in candidate_ellipses if LA.norm(e[0] - cluster_center) < max(e[1]) / 10]
            else:
                candidate_ellipses = [e for e in candidate_ellipses if LA.norm(e[0] - cluster_center) < max(max(e[1]) / 20, 3)]

            # Discard false positives
            if len(candidate_ellipses) < min_ellipses_num:
                continue

            c = contours[candidate_ellipses[-1][3]]
            boundary = (np.amin(c, axis=0)[0][0], np.amax(c, axis=0)[0][0]), (np.amin(c, axis=0)[0][1], np.amax(c, axis=0)[0][1])

            candidate_ellipses = [(e[0], e[1], e[2]) for e in candidate_ellipses]
            concentric_circle_clusters.append((candidate_ellipses, boundary))

        # Return clusters sorted by the number of ellipses and the size of largest ellipse
        return sorted(concentric_circle_clusters, key=lambda x: (-len(x[0]), -max(x[0][-1][1])))
    else:
        for j in (0, 1):
            # Need to inverse the edge so that the curves of a stop marker on the boundary could be taken as a contour
            if j == 1:
                edge = 255-edge
            _, contours, hierarchy = cv2.findContours(edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE, offset=(0, 0))  # TC89_KCOS

            if contours is None or hierarchy is None:
                return []
            clusters = get_nested_clusters(contours, hierarchy[0], min_ellipses_num)
            # speed up code by caching computed ellipses
            ellipses = {}

            for cluster in clusters:
                candidate_ellipses = []
                if len(cluster) < min_ellipses_num:
                    continue
                for i in cluster:
                    c = contours[i]
                    if i in ellipses:
                        e, fit = ellipses[i]
                    else:
                        if len(c) >= 5:
                            e = cv2.fitEllipse(c)
                            fit = max(dist_pts_ellipse(e, c)) if min(e[1]) else 0
                            e = e if min(e[1]) else (e[0], (e[1][0]+1, e[1][1]+1), e[2])
                        else:
                            fit = 0
                            center = c[len(c) // 2][0]
                            e = ((center[0], center[1]), (c[-1][0][0] - c[0][0][0] + 1, c[-1][0][1] - c[0][0][1] + 1), 0)

                        ellipses[i] = e, fit

                    # Discard the contour which does not fit the ellipse so well
                    if fit < max(1.5, max(e[1]) / 50):
                        candidate_ellipses.append(e)
                        if len(candidate_ellipses) == 3 and sum(candidate_ellipses[0][1])/sum(e[1]) > 0.2:
                            break
                        elif len(candidate_ellipses) == 4:
                            break

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
        if len(contours[i]) >= 1:  # we assume that valid markers innermost contour is more than 1 pixel
            cluster = add_parents(i, hierarchy, [])
            # is this cluster bigger that the current contender in the innermost parent group if if already exsists?
            if min_nested_count <= len(cluster) > len(clusters.get(cluster[1], [])):
                clusters[cluster[1]] = cluster
    return clusters.values()


if __name__ == '__main__':
    def bench():
        import cv2
        cap = cv2.VideoCapture(0)
        cap.set(3,1280)
        cap.set(4,720)
        for x in range(100):
            sts,img = cap.read()
            # img = cv2.imread('/Users/mkassner/Desktop/manual_calibration_marker-01.png')
            gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            print(len(find_concetric_circles(gray,visual_debug=img)))
            cv2.imshow('img',img)
            cv2.waitKey(1)
            # return


    import cProfile,subprocess,os
    cProfile.runctx('bench()',{},locals(),'world.pstats')
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call('python '+gprof2dot_loc+' -f pstats world.pstats | dot -Tpng -o world_cpu_time.png', shell=True)
    print('created  time graph for  process. Please check out the png next to this file')
