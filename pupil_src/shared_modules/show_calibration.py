'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
from plugin import Plugin
from calibration_routines.calibrate import calibrate_2d_polynomial
from file_methods import load_object

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
        super().__init__(g_pool)

        self.menu=None

        logger.error("This will be implemented as part of gaze mapper soon.")
        self.alive= False
        return


        width,height = self.g_pool.capture.frame_size

        if g_pool.app == 'capture':
            cal_pt_path =  os.path.join(g_pool.user_dir,"user_calibration_data")
        else:
            cal_pt_path =  os.path.join(g_pool.rec_dir,"user_calibration_data")

        try:
            user_calibration_data = load_object(cal_pt_path)
        except:
            logger.warning("Please calibrate first")
            self.close()
            return

        if self.g_pool.binocular:

            fn_input_eye1 = cal_pt_cloud[:,2:4].transpose()
            cal_pt_cloud[:,0:2] =  np.array(map_fn(fn_input_eye0, fn_input_eye1)).transpose()
            cal_pt_cloud[:,2:4] = cal_pt_cloud[:,4:6]
        else:
            fn_input = cal_pt_cloud[:,0:2].transpose()
            cal_pt_cloud[:,0:2] =  np.array(map_fn(fn_input)).transpose()

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
        self.menu.append(ui.Button('Close', self.close))
        self.info = ui.Info_Text("Yellow: calibration error; Red: discarded outliers; Outline: calibrated area.")
        self.menu.append(self.info)
        self.menu.append(ui.Text_Input('inlier_count',self, label='Number of used samples'))
        self.menu.elements[-1].read_only=True
        self.menu.append(ui.Text_Input('inlier_ratio',self, label='Fraction of used data points'))
        self.menu.elements[-1].read_only=True
        self.menu.append(ui.Text_Input('calib_area_ratio',self, label='Fraction of calibrated screen area'))
        self.menu.elements[-1].read_only=True

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
    map_fn,inlier_map = calibrate_2d_polynomial(cal_pt_cloud,(1280,720),return_inlier_map=True)
    # print cal_pt_cloud[inlier_map][:,0:2].shape
    # print cal_pt_cloud[inlier_map][0,2:4]
    inlier = np.concatenate((cal_pt_cloud[inlier_map][:,0:2],cal_pt_cloud[inlier_map][:,2:4]),axis=1)
    print(inlier)
    print(inlier.reshape(-1,2))
