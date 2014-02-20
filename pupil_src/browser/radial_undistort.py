'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
import numpy as np

def radial_undistort(img, K, dist_coefs):
	img_undistort = cv2.undistort(img_distort, camera_matrix, dist_coefs)
	return img_undistort
