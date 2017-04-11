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
import scipy.spatial as sp



from methods import normalize,denormalize
from gl_utils import adjust_gl_view,clear_gl_screen,basic_gl_setup
import OpenGL.GL as gl
from glfw import *
from . import calibrate

import audio

from pyglui import ui
from pyglui.cygl.utils import draw_points, draw_points_norm, draw_polyline, draw_polyline_norm, RGBA

from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
from . calibration_plugin_base import Calibration_Plugin
from . screen_marker_calibration import Screen_Marker_Calibration

#logging
import logging
logger = logging.getLogger(__name__)



class Accuracy_Test(Screen_Marker_Calibration,Calibration_Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites not between
    """
    def __init__(self, g_pool,fullscreen=True,marker_scale=1.0,sample_duration=40):
        super().__init__(g_pool,fullscreen,marker_scale)

        #result calculation variables:
        self.fov = 90. #taken from c930e specsheet, confirmed though mesurement within ~10deg.
        self.res =  np.sqrt(self.g_pool.capture.frame_size[0]**2+self.g_pool.capture.frame_size[1]**2)
        self.outlier_thresh = 5.
        self.accuracy = 0
        self.precision = 0

        try:
            self.pt_cloud = np.load(os.path.join(self.g_pool.user_dir,'accuracy_test_pt_cloud.npy'))
            gaze,ref = self.pt_cloud[:,0:2],self.pt_cloud[:,2:4]
            error_lines = np.array([[g,r] for g,r in zip(gaze,ref)])
            self.error_lines = error_lines.reshape(-1,2)
        except Exception:
            self.error_lines = None
            self.pt_cloud = None


    def init_gui(self):
        self.monitor_idx = 0
        self.monitor_names = [glfwGetMonitorName(m) for m in glfwGetMonitors()]

        #primary_monitor = glfwGetPrimaryMonitor()
        self.info = ui.Info_Text("Measure gaze mapping accuracy and precision using a screen based animation: After having calibrated on the screen run this test. To compute results set your world cam FOV and click 'calculate results'.")
        self.g_pool.calibration_menu.append(self.info)

        self.menu = ui.Growing_Menu('Controls')
        self.g_pool.calibration_menu.append(self.menu)
        self.menu.append(ui.Selector('monitor_idx',self,selection = range(len(self.monitor_names)),labels=self.monitor_names,label='Monitor'))
        self.menu.append(ui.Switch('fullscreen',self,label='Use Fullscreen'))


        submenu = ui.Growing_Menu('Error Calculation')
        submenu.append(ui.Text_Input('fov',self,'diagonal camera FV'))
        submenu.append(ui.Text_Input('res',self,'diagonal resolution'))
        submenu[-1].read_only = True
        submenu.append(ui.Slider('outlier_thresh',self,label='outlier threshold deg',min=0,max=10))
        submenu.append(ui.Button('calculate result',self.calc_result))

        accuracy_help ='''Accuracy is calculated as the average angular
                        offset (distance) (in degrees of visual angle)
                        between fixations locations and the corresponding
                        locations of the fixation targets.'''.replace("\n"," ").replace("    ",'')

        precision_help = '''Precision is calculated as the Root Mean Square (RMS)
                            of the angular distance (in degrees of visual angle)
                            between successive samples during a fixation.'''.replace("\n"," ").replace("    ",'')

        submenu.append(ui.Info_Text(accuracy_help))
        submenu.append(ui.Text_Input('accuracy',self,'angular accuracy'))
        submenu.append(ui.Info_Text(precision_help))
        submenu.append(ui.Text_Input('precision',self,'angluar precision'))
        self.menu.append(submenu)


        submenu = ui.Growing_Menu('Advanced Detector Settings')
        submenu.collapsed = True
        submenu.append(ui.Slider('sample_duration',self,step=1,min=10,max=100,label='Sample duration'))
        self.menu.append(submenu)


        self.button = ui.Thumb('active',self,label='C',setter=self.toggle,hotkey='c')
        self.button.on_color[:] = (.3,.2,1.,.9)
        self.g_pool.quickbar.insert(0,self.button)


    def deinit_gui(self):
        if self.menu:
            self.g_pool.calibration_menu.remove(self.menu)
            self.g_pool.calibration_menu.remove(self.info)
            self.menu = None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None



    def start(self):
        audio.say("Starting Accuracy Test")
        logger.info("Starting Accuracy_Test")
        self.sites = [  (.5, .5), (0,.5),
                        (0.,1),(.5,1),(1.,1.),
                        (1,.5),
                        (1., 0),(.5, 0),(0,0.),
                        (.5,.5),(.5,.5)]
        self.sites = np.random.random((10,2)).tolist() + self.sites
        self.active_site = self.sites.pop(0)
        self.active = True
        self.ref_list = []
        self.pupil_list = [] #we dont use it only here becasue we use update fn from parent
        self.gaze_list = []
        self.open_window("Accuracy_Test")

    def recent_events(self, events):
        super().recent_events(events)
        if self.active:
            # always save gaze positions as opposed to pupil positons during calibration
            for pt in events.get('gaze_positions', []):
                if pt['confidence'] > self.pupil_confidence_threshold:
                    #we add an id for the calibration preprocess data to work as is usually expects pupil data.
                    pt['id'] = 0
                    self.gaze_list.append(pt)


    def stop(self):
        audio.say("Stopping Accuracy Test")
        logger.info('Stopping Accuracy_Test')
        self.screen_marker_state = 0
        self.active = False
        self.close_window()

        matched_data = calibrate.closest_matches_monocular(self.gaze_list,self.ref_list)
        pt_cloud = calibrate.preprocess_2d_data_monocular(matched_data)
        logger.info("Collected {} data points.".format(len(pt_cloud)))

        if len(pt_cloud) < 20:
            logger.warning("Did not collect enough data.")
            return

        pt_cloud = np.array(pt_cloud)
        np.save(os.path.join(self.g_pool.user_dir,'accuracy_test_pt_cloud.npy'),pt_cloud)
        gaze,ref = pt_cloud[:,0:2],pt_cloud[:,2:4]
        error_lines = np.array([[g,r] for g,r in zip(gaze,ref)])
        self.error_lines = error_lines.reshape(-1,2)
        self.pt_cloud = pt_cloud


    def calc_result(self):
        #lets denormalize:
        # test world cam resolution
        if self.pt_cloud is None:
            logger.warning("Please run test first!")
            return

        pt_cloud = self.pt_cloud.copy()
        res = self.g_pool.capture.frame_size
        pt_cloud[:,0:3:2] *= res[0]
        pt_cloud[:,1:4:2] *= res[1]

        field_of_view = self.fov
        px_per_degree = self.res/field_of_view
        # Accuracy is calculated as the average angular
        # offset (distance) (in degrees of visual angle)
        # between fixations locations and the corresponding
        # locations of the fixation targets.

        gaze,ref = pt_cloud[:,0:2],pt_cloud[:,2:4]
        # site = pt_cloud[:,4]
        error_lines = np.array([[g,r] for g,r in zip(gaze,ref)])
        error_lines = error_lines.reshape(-1,2)
        error_mag = sp.distance.cdist(gaze,ref).diagonal().copy()
        accuracy_pix = np.mean(error_mag)
        logger.info("Gaze error mean in world camera pixel: {:f}".format(accuracy_pix))
        error_mag /= px_per_degree
        logger.info('Error in degrees: {}'.format(error_mag))
        logger.info('Outliers: {}'.format(np.where(error_mag >=self.outlier_thresh)))
        self.accuracy = np.mean(error_mag[error_mag < self.outlier_thresh])
        logger.info('Angular accuracy: {}'.format(self.accuracy))


        #lets calculate precision:  (RMS of distance of succesive samples.)
        # This is a little rough as we do not compensate headmovements in this test.

        # Precision is calculated as the Root Mean Square (RMS)
        # of the angular distance (in degrees of visual angle)
        # between successive samples during a fixation
        succesive_distances_gaze = sp.distance.cdist(gaze[:-1],gaze[1:]).diagonal().copy()
        succesive_distances_ref = sp.distance.cdist(ref[:-1],ref[1:]).diagonal().copy()
        succesive_distances_gaze /=px_per_degree
        succesive_distances_ref /=px_per_degree
        # if the ref distance is to big we must have moved to a new fixation or there is headmovement,
        # if the gaze dis is to big we can assume human error
        # both times gaze data is not valid for this mesurement
        succesive_distances =  succesive_distances_gaze[np.logical_and(succesive_distances_gaze< 1., succesive_distances_ref< .1)]
        self.precision = np.sqrt(np.mean(succesive_distances**2))
        logger.info("Angular precision: {}".format(self.precision))

    def gl_display(self):
        super().gl_display()

        if not self.active and self.error_lines is not None:
            draw_polyline_norm(self.error_lines,color=RGBA(1.,0.5,0.,.5),line_type=gl.GL_LINES)
            draw_points_norm(self.error_lines[1::2],color=RGBA(.0,0.5,0.5,.5),size=3)
            draw_points_norm(self.error_lines[0::2],color=RGBA(.5,0.0,0.0,.5),size=3)
