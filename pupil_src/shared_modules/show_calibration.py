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
from plugin import Plugin
from calibration_routines.calibrate import get_map_from_cloud

from pyglui import ui
from pyglui.cygl.utils import RGBA
from pyglui.cygl.utils import draw_polyline_norm
from OpenGL.GL import GL_LINES,GL_LINE_LOOP

#logging
import logging
logger = logging.getLogger(__name__)

class Show_Calibration(Plugin):
    """Calibration results visualization plugin"""
    def __init__(self,g_pool):
        super(Show_Calibration, self).__init__(g_pool)
        self.menu=None

        width,height = self.g_pool.capture.frame_size

        if g_pool.app == 'capture':
            cal_pt_path =  os.path.join(g_pool.user_dir,"cal_pt_cloud.npy")
        else:
            cal_pt_path =  os.path.join(g_pool.rec_dir,"cal_pt_cloud.npy")

        try:
            cal_pt_cloud = np.load(cal_pt_path)
        except:
            logger.warning("Please calibrate first")
            self.close()
            return

        if self.g_pool.binocular:
            map_fns,inlier_map = get_map_from_cloud(cal_pt_cloud,(width, height),binocular=True,return_inlier_map=True)
            idx_0 = cal_pt_cloud[:,4] == 0
            idx_1 = cal_pt_cloud[:,4] == 1
            cal_pt_cloud[idx_0][:,0:2] = np.array(map_fns[0](cal_pt_cloud[idx_0][:,0:2].transpose())).transpose()
            cal_pt_cloud[idx_1][:,0:2] = np.array(map_fns[1](cal_pt_cloud[idx_1][:,0:2].transpose())).transpose()
        else:
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
        self.menu = ui.Scrolling_Menu('Calibration Results',pos=(300,300),size=(300,300))
        self.info = ui.Info_Text("Yellow: calibration error; Red: discarded outliers; Outline: calibrated area.")
        self.menu.append(self.info)
        self.menu.append(ui.Text_Input('inlier_count',self, label='Number of used samples'))
        self.menu.elements[-1].read_only=True
        self.menu.append(ui.Text_Input('inlier_ratio',self, label='Fraction of used data points'))
        self.menu.elements[-1].read_only=True
        self.menu.append(ui.Text_Input('calib_area_ratio',self, label='Fraction of calibrated screen area'))
        self.menu.elements[-1].read_only=True

        self.menu.append(ui.Button('Close', self.close))
        self.g_pool.gui.append(self.menu)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)


    def gl_display(self):
        if self.inliers is not None:
            draw_polyline_norm(self.inliers,1,RGBA(1.,.5,0.,.5),line_type=GL_LINES)
            draw_polyline_norm(self.outliers,1,RGBA(1.,0.,0.,.5),line_type=GL_LINES)
            draw_polyline_norm(self.calib_bounds[:,0],1,RGBA(.0,1.,0,.5),line_type=GL_LINE_LOOP)

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