'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

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
        Resize the image and then track the markers in the rois / in the whole frame
        Return all detected markers including the information about their ellipses, center positions and
        whether they are stop markers
        """
        img_size = img.shape[::-1]
        scale = 0.5 if img_size[0] >= 1280 else 640 / img_size[0]

        marker_list = []
        # Check whole frame
        if not self.flag_check_roi:
            ellipses_list = self.find_pupil_circle_marker(img, scale)

            # Save the markers in their original size
            for ellipses_ in ellipses_list:
                ellipses = ellipses_['ellipses']
                img_pos = ellipses[0][0]
                norm_pos = normalize(img_pos, img_size, flip_y=True)
                marker_list.append({'ellipses': ellipses, 'img_pos': img_pos, 'norm_pos': norm_pos, 'stop_marker': ellipses_['stop_marker']})

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

                ellipses_list = self.find_pupil_circle_marker(img[slice(*row_slice), slice(*col_slice)], scale)

                # Save the markers in their original size
                for ellipses_ in ellipses_list:
                    ellipses = [((e[0][0]+b0, e[0][1]+b2), e[1], e[2]) for e in ellipses_['ellipses']]
                    img_pos = ellipses[0][0]
                    norm_pos = normalize(img_pos, img_size, flip_y=True)
                    marker_list.append({'ellipses': ellipses, 'img_pos': img_pos, 'norm_pos': norm_pos, 'stop_marker': ellipses_['stop_marker']})

        return marker_list


    def find_pupil_circle_marker(self, img, scale):
        """
        :param img: gray image
        :return: all detected markers in resized frame
        """

        img_size = img.shape[::-1]
        # Resize the image
        img_resize = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)

        # Extract the edges of the image by two kinds of threshold
        edges = []
        # For normal and complicated scene
        # Need to inverse the edges so that the curves on the boundary could be taken as a contour
        edges.append(cv2.adaptiveThreshold(cv2.blur(img_resize, (3, 3)), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 30))
        # # For marker in low contrast
        edges.append(cv2.adaptiveThreshold(cv2.blur(img_resize, (3, 3)), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 2))

        ellipses_list = []
        marker_found_pos = []
        for i in range(len(edges)):
            edge = edges[i]
            circle_clusters = find_concentric_circles(edge, first_check=True, min_ring_count=2)

            for ellipses, boundary in circle_clusters:
                img_pos = ellipses[0][0]
                # Discard duplicates
                if len(marker_found_pos) and min((abs(np.array(img_pos) - np.array(marker_found_pos))).max(axis=1)) < max(ellipses[1][1]):
                    continue

                # Set up the boundary of the ellipses
                b0, b1 = boundary[0][0] / scale, boundary[0][1] / scale + 2
                b2, b3 = boundary[1][0] / scale, boundary[1][1] / scale + 2

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

                # Find the edge of the marker again with new kernel_size
                block_size = max(5, int(max(img_ellipse_size) / 4) * 2 + 1)
                c = img_ellipse_mean / 64

                mask_outer = cv2.adaptiveThreshold(cv2.GaussianBlur(img_ellipse, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)
                temp = find_concentric_circles(mask_outer, first_check=False, min_ring_count=3)

                if len(temp) == 0:
                    continue
                single_marker = temp[0][0]

                # Get all the ellipses on the ring
                divider_size = sum(single_marker[-1][1]) * 0.4

                small_ellipses = [e for e in single_marker if sum(e[1]) < divider_size]
                if len(small_ellipses) == 0:
                    continue
                dot_ellipse = small_ellipses[-1]

                large_ellipses = [e for e in single_marker if sum(e[1]) >= divider_size]
                if len(large_ellipses) < 2:
                    continue
                outer_ellipse = large_ellipses[-1]
                inner_ellipse = large_ellipses[0]

                # Calculate the ring ratio
                ring_ratio = sum(outer_ellipse[1]) / sum(inner_ellipse[1])

                dot_center = int(round(dot_ellipse[0][1])), int(round(dot_ellipse[0][0]))

                if mask_outer[dot_center]:
                    # Check the ring ratio
                    if not 1.2 < ring_ratio < 2.1:
                        continue
                    # Discard false positive by the ring color compared to the color of the outer/inner part of the ring

                    outer_mean = np.ma.array(img_ellipse, mask=mask_outer).mean()

                    mask_ring = np.ones_like(img_ellipse)*255
                    cv2.ellipse(mask_ring, outer_ellipse, color=(0, 0, 0), thickness=-1)
                    cv2.ellipse(mask_ring, inner_ellipse, color=(255, 255, 255), thickness=-1)
                    ring_median = np.ma.median(np.ma.array(img_ellipse, mask=mask_ring))

                    if outer_mean - ring_median < img_ellipse_mean / 4:
                        continue

                    mask_middle = np.ones_like(img_ellipse)*255
                    cv2.ellipse(mask_middle, inner_ellipse, color=(0, 0, 0), thickness=-1)
                    cv2.ellipse(mask_middle, inner_ellipse, color=(255, 255, 255), thickness=2)
                    cv2.ellipse(mask_middle, dot_ellipse, color=(255, 255, 255), thickness=-1)
                    mask_middle_value = np.ma.array(img_ellipse, mask=mask_middle)
                    middle_median = np.ma.median(mask_middle_value)

                    if middle_median - ring_median < img_ellipse_mean / 4:
                        continue

                    middle_std = mask_middle_value.std()
                    white_median = np.ma.median(np.ma.array(img_ellipse, mask=mask_outer))
                    if middle_std / white_median > 0.35:
                        continue

                    single_marker = [((e[0][0]+b0, e[0][1]+b2), e[1], e[2]) for e in single_marker]
                    ellipses_list.append({'ellipses': single_marker, 'stop_marker': False})
                    marker_found_pos.append(img_pos)
                    return ellipses_list

                else:
                    # Check the ring ratio
                    if not 1.4 < ring_ratio < 2.7:
                        continue

                    # Discard false positive by the ring color compared to the color of the outer/inner part of the ring
                    outer_mean = np.ma.array(img_ellipse, mask=cv2.bitwise_not(mask_outer)).mean()

                    mask_ring = np.ones_like(img_ellipse)*255
                    cv2.ellipse(mask_ring, outer_ellipse, color=(0, 0, 0), thickness=-1)
                    cv2.ellipse(mask_ring, outer_ellipse, color=(255, 255, 255), thickness=1)
                    cv2.ellipse(mask_ring, inner_ellipse, color=(255, 255, 255), thickness=-1)
                    ring_median = np.ma.median(np.ma.array(img_ellipse, mask=mask_ring))

                    if ring_median - outer_mean < img_ellipse_mean / 4:
                        continue

                    mask_middle = np.ones_like(img_ellipse)*255
                    cv2.ellipse(mask_middle, inner_ellipse, color=(0, 0, 0), thickness=-1)
                    cv2.ellipse(mask_middle, inner_ellipse, color=(255, 255, 255), thickness=1)
                    cv2.ellipse(mask_middle, dot_ellipse, color=(255, 255, 255), thickness=-1)
                    mask_middle_value = np.ma.array(img_ellipse, mask=mask_middle)
                    middle_median = np.ma.median(mask_middle_value)

                    if ring_median - middle_median < img_ellipse_mean / 4:
                        continue

                    middle_std = mask_middle_value.std()
                    white_median = np.ma.median(np.ma.array(img_ellipse, mask=mask_outer))

                    if middle_std / white_median > 0.4:
                        continue

                    single_marker = [((e[0][0]+b0, e[0][1]+b2), e[1], e[2]) for e in single_marker]
                    ellipses_list.append({'ellipses': single_marker, 'stop_marker': True})
                    marker_found_pos.append(img_pos)
                    return ellipses_list
        return ellipses_list

def find_concentric_circles(edge, first_check=True, min_ring_count=2):
    """
    :param edge: the edge extraction of the image
    :param min_ring_count: minimum number of the ellipses of the marker
    :return: all possible markers
    """

    if first_check:
        concentric_circle_clusters = []
        _, contours, hierarchy = cv2.findContours(edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE, offset=(0, 0))  # TC89_KCOS
        if contours is None or hierarchy is None:
            return []
        clusters = get_nested_clusters(contours, hierarchy[0], min_ring_count)
        # speed up code by caching computed ellipses
        ellipses = {}

        # for each cluster fit ellipses and cull members that dont have good ellipse fit
        for cluster in clusters:
            candidate_ellipses = []
            if len(cluster) < min_ring_count:
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

                if first_ellipse or fit < max(1, max(e[1]) / 50):
                    e = (e[0], e[1], e[2], i)
                    candidate_ellipses.append(e)
                first_ellipse = False

            # Discard false positives
            if len(candidate_ellipses) < min_ring_count or np.mean(candidate_ellipses[-1][1]) < 3:
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
            if len(candidate_ellipses) < min_ring_count:
                continue

            c = contours[candidate_ellipses[-1][3]]
            boundary = (np.amin(c, axis=0)[0][0], np.amax(c, axis=0)[0][0]), (np.amin(c, axis=0)[0][1], np.amax(c, axis=0)[0][1])

            candidate_ellipses = [(e[0], e[1], e[2]) for e in candidate_ellipses]
            concentric_circle_clusters.append((candidate_ellipses, boundary))

        # Return clusters sorted by the number of ellipses and the size of largest ellipse
        return sorted(concentric_circle_clusters, key=lambda x: (-len(x[0]), -max(x[0][-1][1])))
    else:
        for j in (0, 1):
            if j == 0:  # For normal marker
                _, contours, hierarchy = cv2.findContours(edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE, offset=(0, 0))  # TC89_KCOS
            else:       # For stop marker
                _, contours, hierarchy = cv2.findContours(255 - edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE, offset=(0, 0))  # TC89_KCOS
            if contours is None or hierarchy is None:
                return []
            clusters = get_nested_clusters(contours, hierarchy[0], min_ring_count)
            # speed up code by caching computed ellipses
            ellipses = {}

            # for each cluster fit ellipses and cull members that dont have good ellipse fit
            for cluster in clusters:
                candidate_ellipses = []
                if len(cluster) < min_ring_count:
                    continue
                for i in cluster:
                    c = contours[i]
                    if i in ellipses:
                        e, fit = ellipses[i]
                    else:
                        if len(c) >= 5:
                            e = cv2.fitEllipse(c)
                            fit = np.mean(dist_pts_ellipse(e, c)) if min(e[1]) else 0
                            e = e if min(e[1]) else (e[0], (e[1][0]+1, e[1][1]+1) , e[2])
                        else:
                            fit = 0
                            center = c[len(c) // 2][0]
                            e = ((center[0], center[1]), (c[-1][0][0] - c[0][0][0] + 1, c[-1][0][1] - c[0][0][1] + 1), 0)
                        ellipses[i] = e, fit
                    if fit < max(0.65, max(e[1]) / 50):
                        candidate_ellipses.append(e)

                # Discard false positives
                if len(candidate_ellipses) < min_ring_count:
                    continue

                # Discard the ellipses whose center is far away from the center of the innermost ellipse
                cluster_center = np.array(candidate_ellipses[0][0])
                if max(candidate_ellipses[-1][1]) > 200:
                    candidate_ellipses = [e for e in candidate_ellipses if LA.norm(e[0] - cluster_center) < max(e[1]) / 5]
                elif max(candidate_ellipses[-1][1]) > 100:
                    candidate_ellipses = [e for e in candidate_ellipses if LA.norm(e[0] - cluster_center) < max(e[1]) / 10]
                else:
                    candidate_ellipses = [e for e in candidate_ellipses if LA.norm(e[0] - cluster_center) < max(max(e[1]) / 20, 1.75)]

                # Discard false positives
                if len(candidate_ellipses) < min_ring_count:
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
    # nesting of countours where many children are grouping in a sigle parent happens a lot (your screen with stuff in it. A page with text...)
    # we create a cluster for each of these children.
    # to reduce CPU load we onle keep the biggest cluster for clusters where the innermost parent is the same.
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
    cProfile.runctx("bench()",{},locals(),"world.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
    print("created  time graph for  process. Please check out the png next to this file")
