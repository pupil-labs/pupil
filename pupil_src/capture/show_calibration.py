'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
from gl_utils import draw_gl_polyline_norm
from plugin import Plugin
from calibration_routines.calibrate import get_map_from_cloud

#logging
import logging
logger = logging.getLogger(__name__)

class Show_Calibration(Plugin):
    """Calibration results visualization plugin"""
    def __init__(self,g_pool):
        super(Show_Calibration, self).__init__(g_pool)

        height,width = frame.height, frame.width

        if g_pool.app == 'capture':
            cal_pt_path =  os.path.join(g_pool.user_dir,"cal_pt_cloud.npy")
        else:
            # cal_pt_path =  os.path.join(g_pool.rec_dir,"cal_pt_cloud.npy")
            logger.error('Plugin only works in capture.')
            self.close()
            return

        try:
            cal_pt_cloud = np.load(cal_pt_path)
        except:
            logger.warning("Please calibrate first")
            self.close()
            return

        map_fn,inlier_map = get_map_from_cloud(cal_pt_cloud,(width, height),return_inlier_map=True)
        cal_pt_cloud[:,0:2] =  np.array(map_fn(cal_pt_cloud[:,0:2].transpose())).transpose()
        ref_pts = cal_pt_cloud[inlier_map][:,np.newaxis,2:4]
        ref_pts = np.array(ref_pts,dtype=np.float32)
        logger.debug("calibration ref_pts %s"%ref_pts)

        if len(ref_pts)== 0:
            logger.warning("Calibration is bad. Please re-calibrate")
            self.close()
            return

        self.calib_bounds =  cv2.convexHull(ref_pts)
        # create a list [[px1,py1],[wx1,wy1],[px2,py2],[wx2,wy2]...] of outliers and inliers for gl_lines
        self.outliers = np.concatenate((cal_pt_cloud[~inlier_map][:,0:2],cal_pt_cloud[~inlier_map][:,2:4])).reshape(-1,2)
        self.inliers = np.concatenate((cal_pt_cloud[inlier_map][:,0:2],cal_pt_cloud[inlier_map][:,2:4]),axis=1).reshape(-1,2)


        self.inlier_ratio = cal_pt_cloud[inlier_map].shape[0]/float(cal_pt_cloud.shape[0])
        self.inlier_count = cal_pt_cloud[inlier_map].shape[0]
        # hull = cv2.approxPolyDP(self.calib_bounds, 0.001,closed=True)
        full_screen_area = 1.
        logger.debug("calibration bounds %s"%self.calib_bounds)
        self.calib_area_ratio = cv2.contourArea(self.calib_bounds)/full_screen_area



    def init_gui(self):
        help_str = "yellow: indicates calibration error; red:discarded outliers; outline shows the calibrated area."
        self.menu = ui.Scrolling_Menu('Calibration Results',pos=(300,500),size=(300,300))
        self.menu.append(ui.TextInput('inlier_count',self,getter=lambda: str(self.inlier_count), label='Number of used samples'))
        self.menu.append(ui.TextInput('inlier_ratio',self,getter=lambda: str(self.inlier_ratio)), label='Fraction of used data points')
        self.menu.append(ui.TextInput('calib_area_ratio',self,getter=lambda: str(self.calib_area_ratio)), label='Fraction of calibrated screen area')
        self.menu.append(ui.Button("Close Plugin", self.close))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)


    def gl_display(self):
        if self.inliers is not None:
            draw_gl_polyline_norm(self.inliers,(1.,0.5,0.,.5),type='Lines')
            draw_gl_polyline_norm(self.outliers,(1.,0.,0.,.5),type='Lines')
            draw_gl_polyline_norm(self.calib_bounds[:,0],(.0,1.,0,.5),type='Loop')

    def close(self):
        self.alive = False

    def cleanup(self):
        self.deinit_gui()



if __name__ == '__main__':
    cal_pt_cloud = np.load("cal_pt_cloud.npy")
    map_fn,inlier_map = get_map_from_cloud(cal_pt_cloud,(1280,720),return_inlier_map=True)
    # print cal_pt_cloud[inlier_map][:,0:2].shape
    # print cal_pt_cloud[inlier_map][0,2:4]
    inlier = np.concatenate((cal_pt_cloud[inlier_map][:,0:2],cal_pt_cloud[inlier_map][:,2:4]),axis=1)
    print inlier
    print inlier.reshape(-1,2)