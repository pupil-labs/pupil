'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import cv2
import numpy as np
from methods import normalize


class FingertipTracker(object):
    def __init__(self, wait_interval=30, roi_wait_interval=120):
        self.wait_interval = wait_interval
        self.roi_wait_interval = roi_wait_interval
        self.train_done = 0
        self._ROIpts_1 = []
        self._ROIpts_2 = []
        self._defineROIpts()
        self.ROIpts = self._ROIpts_1
        self.method = HSV_Bound()

        self._previous_finger_dict = None
        self._predict_motion = None
        self._wait_count = 0
        self._roi_wait_count = 0
        self._flag_check = False
        self._flag_check_roi = False

        self._contourwidthThres = None
        self._contourheightThres = None
        self._epsilon = None
        self._margin = None
        self._kernel_OPEN = None
        self._kernel_CLOSE = None
        self._finger_area_ratios = []
        self._finger_rects = []

    def _initParam(self, img_size):
        self._contourwidthThres = img_size[0] // 128
        self._contourheightThres = img_size[1] // 24
        self._epsilon = img_size[0] / 256.
        self._margin = img_size[0] // 128
        kernel_size = img_size[0] // 256
        self._kernel_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel_size = img_size[0] // 36
        self._kernel_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        self._previous_finger_dict = None
        self._predict_motion = None
        self._wait_count = 0
        self._roi_wait_count = 0
        self.train_done = 0
        self.method.hsv_median = []
        self.ROIpts = self._ROIpts_1

    def update(self, img, press_key):
        finger_dict = None
        if press_key == -1 or press_key == 2:
            img_size = img.shape[1], img.shape[0]
            self._initParam(img_size)

        if self.train_done == 0:
            if press_key == 1:
                self._trainSkinColorDetector(img)
                self.train_done = 1
                self.ROIpts = self._ROIpts_2
        elif self.train_done == 1:
            if press_key == 1:
                self._trainSkinColorDetector(img)
                self.train_done = 2
                self.ROIpts = None
        elif self.train_done == 2:
            if press_key == 1:
                self.train_done = 3
        elif self.train_done == 3:
            self.method.train()
            self.train_done = 4
        elif self.train_done == 4:
            if self._wait_count <= 0 or self._roi_wait_count <= 0:
                self._flag_check = True
                self._flag_check_roi = False
                self._wait_count = self.wait_interval
                self._roi_wait_count = self.roi_wait_interval

            if self._flag_check:
                finger_dict = self._checkFrame(img)
                self._predict_motion = None
                if finger_dict is not None:
                    self._flag_check = True
                    self._flag_check_roi = True
                    self._roi_wait_count -= 1
                    if self._previous_finger_dict is not None:
                        self._predict_motion = np.array(finger_dict['screen_pos']) - np.array(
                            self._previous_finger_dict['screen_pos'])
                else:
                    if self._flag_check_roi:
                        self._flag_check = True
                        self._flag_check_roi = False
                    else:
                        self._flag_check = False
                        self._flag_check_roi = False

            self._wait_count -= 1
            self._previous_finger_dict = finger_dict

        return finger_dict

    def _checkFrame(self, img):
        img_size = img.shape[1], img.shape[0]

        # Check whole frame
        if not self._flag_check_roi:
            b0, b1, b2, b3 = 0, img_size[0], 0, img_size[1]

        # Check roi
        else:
            previous_fingertip_center = self._previous_finger_dict['screen_pos']
            # Set up the boundary of the roi
            temp = img_size[0] / 16
            if self._predict_motion is not None:
                predict_center = previous_fingertip_center[0] + self._predict_motion[0], previous_fingertip_center[1] + self._predict_motion[1]
                b0 = predict_center[0] - temp * 0.5 - abs(self._predict_motion[0]) * 2
                b1 = predict_center[0] + temp * 0.5 + abs(self._predict_motion[0]) * 2
                b2 = predict_center[1] - temp * 0.8 - abs(self._predict_motion[1]) * 2
                b3 = predict_center[1] + temp * 2.0 + abs(self._predict_motion[1]) * 2
            else:
                predict_center = previous_fingertip_center
                b0 = predict_center[0] - temp * 0.5
                b1 = predict_center[0] + temp * 0.5
                b2 = predict_center[1] - temp * 0.8
                b3 = predict_center[1] + temp * 2.0

            b0 = 0 if b0 < 0 else int(b0)
            b1 = img_size[0] - 1 if b1 > img_size[0] - 1 else int(b1)
            b2 = 0 if b2 < 0 else int(b2)
            b3 = img_size[1] - 1 if b3 > img_size[1] - 1 else int(b3)
            col_slice = b0, b1
            row_slice = b2, b3
            img = img[slice(*row_slice), slice(*col_slice)]

        handmask = self.method.generateMask(img)
        handmask_smooth = self._smoothmask(handmask)
        f_dict = self._findFingertip(handmask_smooth, img_size, b0, b2)

        if f_dict is not None:
            norm_pos = normalize(f_dict['fingertip_center'], img_size, flip_y=True)
            norm_rect_points = [normalize(p, img_size, flip_y=True) for p in f_dict['rect_points']]
            return {'screen_pos': f_dict['fingertip_center'], 'norm_pos': norm_pos, 'norm_rect_points': norm_rect_points}
        else:
            return None

    def _findFingertip(self, handmask_smooth, img_size, b0, b2):
        _, contours, _ = cv2.findContours(handmask_smooth, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_KCOS)
        if len(contours) == 0:
            return None

        self._finger_area_ratios = []
        self._finger_rects = []
        for contour in contours:
            x0, y0, w0, h0 = cv2.boundingRect(contour)
            if w0 < self._contourwidthThres or h0 < self._contourheightThres:
                continue

            width_sum = np.sum(handmask_smooth[y0:y0 + h0, x0:x0 + w0], axis=1)
            if np.std(width_sum) > 12 * 255:
                x1, y1, w1, h1 = x0, y0, w0, np.argmax(width_sum)
                if h0 < self._contourheightThres:
                    continue

                width_sum = width_sum[:h1]
                hist, bin_edges = np.histogram(width_sum, bins=5)
                local_max = np.where(np.r_[True, hist[1:] > hist[:-1]] & np.r_[hist[:-1] > hist[1:], True] == True)[0]
                if len(local_max) == 0:
                    continue

                x2, y2, w2, h2 = x1, y1, w1, np.max(np.where(width_sum < bin_edges[np.min(local_max) + 1]))
                if h2 < self._contourheightThres:
                    continue

                new_mask = handmask_smooth[y2:y2 + h2, x2:x2 + w2]
                if np.sum(new_mask) == 0:
                    continue

                length_sum = np.sum(new_mask, axis=0) // 255
                length_sum_max_index = np.argmax(length_sum)

                x3, y3, w3, h3 = x2, y2, w2, h2
                small_length_sum_index = np.where(length_sum < np.max(length_sum) / 3)[0]
                temp_min = np.where(small_length_sum_index > length_sum_max_index)[0]
                if len(temp_min) > 0:
                    local_min_index = small_length_sum_index[temp_min[0]]
                    w3 = local_min_index
                temp_min = np.where(small_length_sum_index < length_sum_max_index)[0]
                if len(temp_min) > 0:
                    local_min_index = small_length_sum_index[temp_min[-1]]
                    w3 = w3 - local_min_index
                    x3 = x3 + local_min_index
                if w3 < self._contourwidthThres:
                    continue

                new_mask = handmask_smooth[y3:y3 + h3, x3:x3 + w3]
                _, new_contours, _ = cv2.findContours(new_mask, mode=cv2.RETR_EXTERNAL,
                                                      method=cv2.CHAIN_APPROX_TC89_KCOS)
                if len(new_contours) == 0:
                    continue

                contour = new_contours[np.argmax(np.array([len(c) for c in new_contours]))]
                contour += np.array((x3, y3))

            contour_poly = cv2.approxPolyDP(contour, epsilon=self._epsilon, closed=True)
            if not 3 <= len(contour_poly) <= 20:
                continue

            rect = cv2.minAreaRect(contour_poly)
            _, rect_size, _ = rect
            rect_width, rect_height = min(rect_size), max(rect_size)
            if rect_width < self._contourwidthThres or rect_height < self._contourheightThres or rect_height < rect_width * 1.2:
                continue

            finger_area_ratio = cv2.contourArea(contour_poly) / (rect_size[0] * rect_size[1])
            if finger_area_ratio < 1 / 3:
                continue

            self._finger_area_ratios.append(finger_area_ratio)
            self._finger_rects.append(rect)

        if len(self._finger_area_ratios) == 0:
            return None

        best_rect = self._finger_rects[int(np.argmax(self._finger_area_ratios))]
        (rect_x, rect_y), (rect_size_1, rect_size_2), rect_angle = best_rect
        rect_angle = -rect_angle
        rect_angle_cos = np.cos(rect_angle * np.pi / 180.)
        rect_angle_sin = np.sin(rect_angle * np.pi / 180.)
        if rect_size_2 >= rect_size_1 and rect_angle <= 45:
            rect_x = rect_x - (rect_size_2 - rect_size_1) / 2 * rect_angle_sin
            rect_y = rect_y - (rect_size_2 - rect_size_1) / 2 * rect_angle_cos
            rect_size_2 = rect_size_1
            rect_size_1 += self._epsilon * 2
        elif rect_size_2 <= rect_size_1 and rect_angle >= 45:
            rect_x = rect_x + (rect_size_1 - rect_size_2) / 2 * rect_angle_cos
            rect_y = rect_y - (rect_size_1 - rect_size_2) / 2 * rect_angle_sin
            rect_size_1 = rect_size_2
            rect_size_2 += self._epsilon * 2
        elif rect_size_2 <= rect_size_1 and rect_angle <= 45:
            rect_size_2 += self._epsilon * 2
        elif rect_size_2 >= rect_size_1 and rect_angle >= 45:
            rect_size_1 += self._epsilon * 2

        if rect_x + b0 > img_size[0] - self._margin or rect_x + b0 < self._margin or rect_y + b2 < self._margin:
            return None

        new_rect = (rect_x, rect_y), (rect_size_1, rect_size_2), -rect_angle
        new_mask = np.zeros_like(handmask_smooth)
        cv2.drawContours(new_mask, [np.int0(cv2.boxPoints(new_rect))], 0, 255, thickness=-1)
        _, new_contours, _ = cv2.findContours(np.bitwise_and(new_mask, handmask_smooth), mode=cv2.RETR_EXTERNAL,
                                              method=cv2.CHAIN_APPROX_TC89_KCOS)
        if len(new_contours) == 0:
            return None

        new_contour = new_contours[np.argmax(np.array([len(c) for c in new_contours]))]
        (rect_x, rect_y), (rect_size_1, rect_size_2), rect_angle = cv2.minAreaRect(new_contour)
        rect_angle = -rect_angle
        rect_angle_cos = np.cos(rect_angle * np.pi / 180.)
        rect_angle_sin = np.sin(rect_angle * np.pi / 180.)
        if rect_size_2 >= rect_size_1 and rect_angle <= 45:
            rect_x = rect_x - (rect_size_2 - rect_size_1) / 2 * rect_angle_sin
            rect_y = rect_y - (rect_size_2 - rect_size_1) / 2 * rect_angle_cos
        elif rect_size_2 <= rect_size_1 and rect_angle >= 45:
            rect_x = rect_x + (rect_size_1 - rect_size_2) / 2 * rect_angle_cos
            rect_y = rect_y - (rect_size_1 - rect_size_2) / 2 * rect_angle_sin

        fingertip_center = float(rect_x + b0), float(rect_y + b2)
        if fingertip_center[0] > img_size[0] - self._margin or min(fingertip_center) < self._margin:
            return None

        rect_points = cv2.boxPoints(best_rect) + np.array((b0, b2))  # For visualization of the fingertip detector
        return {'fingertip_center': fingertip_center, 'rect_points': rect_points}


    def _defineROIpts(self):
        roi_len_h = 1 / 20
        roi_len_w = 1 / 100
        self._ROIpts_1 = []
        self._ROIpts_1.append((1 / 3 + 1 / 10 * 0, 1 / 6, roi_len_h, roi_len_w))
        self._ROIpts_1.append((1 / 3 + 1 / 10 * 1, 1 / 6, roi_len_h, roi_len_w))
        self._ROIpts_1.append((1 / 3 + 1 / 10 * 2, 1 / 6, roi_len_h, roi_len_w))
        self._ROIpts_1.append((1 / 3 + 1 / 10 * 3, 1 / 6, roi_len_h, roi_len_w))

        self._ROIpts_2 = []
        self._ROIpts_2.append((1 / 3 + 1 / 10 * 0, 5 / 6, roi_len_h, roi_len_w))
        self._ROIpts_2.append((1 / 3 + 1 / 10 * 1, 5 / 6, roi_len_h, roi_len_w))
        self._ROIpts_2.append((1 / 3 + 1 / 10 * 2, 5 / 6, roi_len_h, roi_len_w))
        self._ROIpts_2.append((1 / 3 + 1 / 10 * 3, 5 / 6, roi_len_h, roi_len_w))

    def _trainSkinColorDetector(self, img):
        h, w = img.shape[0], img.shape[1]
        ROIpts = np.array(self.ROIpts * np.array((h, w, h, w)), dtype=np.int)
        ROIimg = [img[p[0]:p[0] + p[2], p[1]:p[1] + p[3]] for p in ROIpts]
        self.method.findSkinColorMedian(ROIimg)

    def _smoothmask(self, handmask):
        handmask_smooth = cv2.morphologyEx(handmask, cv2.MORPH_OPEN, self._kernel_OPEN)
        handmask_smooth = cv2.morphologyEx(handmask_smooth, cv2.MORPH_CLOSE, self._kernel_CLOSE)
        return handmask_smooth


class HSV_Bound(object):
    def __init__(self):
        self.c_lower = np.array((2, 5, 25), dtype=np.float)
        self.c_upper = np.array((2, 5, 25), dtype=np.float)
        self.lowerBound = []
        self.upperBound = []
        self.hsv_median = []

    def findSkinColorMedian(self, ROIimg):
        for r in ROIimg:
            ROIimgHSV = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
            ROIimgH = (ROIimgHSV[:, :, 0] + 75) % 180
            ROIimgS = ROIimgHSV[:, :, 1]
            ROIimgV = ROIimgHSV[:, :, 2]
            ROIimgH_median = np.median(ROIimgH)
            index = np.where(ROIimgH <= ROIimgH_median)
            self.hsv_median.append((np.median(ROIimgH[index]), np.median(ROIimgS[index]), np.median(ROIimgV[index])))
            index = np.where(ROIimgH >= ROIimgH_median)
            self.hsv_median.append((np.median(ROIimgH[index]), np.median(ROIimgS[index]), np.median(ROIimgV[index])))

    def train(self):
        if len(self.hsv_median):
            hsv_median = np.array(self.hsv_median, dtype=np.float)
        else:
            hsv_median = np.array((80, 80, 128), dtype=np.float)
        self.lowerBound = np.clip(hsv_median - self.c_lower, 0, 255)
        self.upperBound = np.clip(hsv_median + self.c_upper, 0, 255)

    def generateMask(self, img_test):
        frameHSV = cv2.cvtColor(img_test, cv2.COLOR_BGR2HSV)
        frameHSV[:, :, 0] = (frameHSV[:, :, 0] + 75) % 180
        frameHSV = cv2.blur(frameHSV, (3, 3))

        bwList = [cv2.inRange(frameHSV, l, p) for l, p in zip(self.lowerBound, self.upperBound)]
        HandMask = np.zeros_like(bwList[0])
        for li in bwList:
            HandMask = np.bitwise_or(HandMask, li)

        return HandMask
