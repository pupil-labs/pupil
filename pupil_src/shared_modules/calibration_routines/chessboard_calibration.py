'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
from file_methods import save_object,load_object
from gl_utils import adjust_gl_view,clear_gl_screen,basic_gl_setup,make_coord_system_pixel_based,make_coord_system_norm_based
from methods import normalize, undistord


import OpenGL.GL as gl
from gl_utils.drawing_utils import apply_gl_texture
from pyglui import ui
from pyglui.cygl.utils import draw_polyline,draw_points,RGBA,draw_gl_texture
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
from glfw import *

from plugin import Calibration_Plugin

#logging
import logging
logger = logging.getLogger(__name__)


#these are calibration we recorded. They are estimates and generalize our setup. Its always better to calibrate each camera.
pre_recorded_calibrations = {
                            'Pupil Cam1 ID2':{
                                (1280, 720):{
                                'dist_coefs': np.array([[-0.6746215 ,  0.46527537,  0.01448595, -0.00070578, -0.17128751]]),
                                'camera_name': 'Pupil Cam1 ID2',
                                'resolution': (1280, 720),
                                'error': 0,
                                'camera_matrix': np.array([[  1.08891909e+03,   0.00000000e+00,   6.67944178e+02],
                                                             [  0.00000000e+00,   1.03230180e+03,   3.52772854e+02],
                                                             [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
                                    }
                                },
                            'Logitech Webcam C930e':{
                                (1280, 720):{
                                    'dist_coefs': np.array([[ 0.06330768, -0.17328079,  0.00074967,  0.000353  ,  0.07648477]]),
                                    'camera_name': 'Logitech Webcam C930e',
                                    'resolution': (1280, 720),
                                    'error': 0,
                                    'camera_matrix': np.array([[ 739.72227378,    0.        ,  624.44490772],
                                                                [   0.        ,  717.84832227,  350.46000651],
                                                                [   0.        ,    0.        ,    1.        ]])
                                    }
                                },
                            }


def load_camera_calibration(g_pool):
    if g_pool.app == 'capture':
        try:
            camera_calibration = load_object(os.path.join(g_pool.user_dir,'camera_calibration'))
        except:
            camera_calibration = None
        else:
            same_name = camera_calibration['camera_name'] == g_pool.capture.name
            same_resolution =  camera_calibration['resolution'] == g_pool.capture.frame_size
            if not (same_name and same_resolution):
                logger.warning('Loaded camera calibration but camera name and/or resolution has changed.')
                camera_calibration = None
            else:
                logger.info("Loaded user calibrated calibration for %s@%s."%(g_pool.capture.name,g_pool.capture.frame_size))

        if not camera_calibration:
            logger.debug("Trying to load pre recorded calibration.")
            try:
                camera_calibration = pre_recorded_calibrations[g_pool.capture.name][g_pool.capture.frame_size]
            except KeyError:
                logger.info("Pre recorded calibration for %s@%s not found."%(g_pool.capture.name,g_pool.capture.frame_size))
            else:
                logger.info("Loaded pre recorded calibration for %s@%s."%(g_pool.capture.name,g_pool.capture.frame_size))


        if not camera_calibration:
            logger.warning("Camera calibration not found please run Camera_Intrinsics_Estimation or Chessboard_Calibration to calibrate camera.")


        return camera_calibration

    else:
        raise NotImplementedError()







# window calbacks
def on_resize(window,w, h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)

class Chessboard_Calibration(Calibration_Plugin):
    """Chessboard_Calibration
        This method is not a gaze calibration.
        This method is used to calculate camera intrinsics.
    """
    def __init__(self,g_pool,fullscreen = False):
        super(Chessboard_Calibration, self).__init__(g_pool)

        self.menu = None
        self.button = None
        self.clicks_to_close = 5
        self.window_should_close = False
        self.fullscreen = fullscreen
        self.monitor_idx = 0
        self.nb_img = 10
        self.nb_cols = 10
        self.nb_rows = 7
        self.last_nb_img = self.nb_img
        self.last_nb_cols = self.nb_cols
        self.last_nb_rows = self.nb_rows


        self.collect_new = False
        self.calculated = False
        self.count = self.nb_img
        self.obj_grid = _gen_pattern_grid((self.nb_rows-1,self.nb_cols-1))
        self.img_points = []
        self.obj_points = []
        #self.display_grid = _make_grid()

        self._window = None

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans',get_opensans_font_path())
        self.glfont.set_size(32)
        self.glfont.set_color_float((0.2,0.5,0.9,1.0))
        self.glfont.set_align_string(v_align='center')



        self.undist_img = None
        self.calibrate_img = None
        self.nb_frame = 10
        self.show_undistortion = False
        self.show_undistortion_switch = None


        self.camera_calibration = load_camera_calibration(self.g_pool)
        if self.camera_calibration:
            logger.info('Loaded camera calibration. Click show undistortion to verify.')
            self.camera_intrinsics = self.camera_calibration['camera_matrix'],self.camera_calibration['dist_coefs'],self.camera_calibration['resolution'],self.camera_calibration['error']
        else:
            self.camera_intrinsics = None


    def init_gui(self):

        monitor_names = [glfwGetMonitorName(m) for m in glfwGetMonitors()]
        #primary_monitor = glfwGetPrimaryMonitor()
        self.info = ui.Info_Text("Estimate Camera intrinsics of the world camera. Using an rowsXcols chessboard. Click 'C' to capture a pattern.")
        self.g_pool.calibration_menu.append(self.info)

        self.menu = ui.Growing_Menu('Controls')
        self.menu.append(ui.Button('show Pattern',self.open_window))
        self.menu.append(ui.Selector('monitor_idx',self,selection = range(len(monitor_names)),labels=monitor_names,label='Monitor'))
        self.menu.append(ui.Switch('fullscreen',self,label='Use Fullscreen'))
        self.menu.append(ui.Slider('nb_img',self,step=1,min=10,max=30))
        self.menu.append(ui.Text_Input('nb_cols',self,'nb_cols'))
        self.menu.append(ui.Text_Input('nb_rows',self,'nb_rows'))
        self.show_undistortion_switch = ui.Switch('show_undistortion',self,label='show undistorted image')
        self.menu.append(self.show_undistortion_switch)
        if not self.camera_intrinsics:
            self.show_undistortion_switch.read_only=True
        self.g_pool.calibration_menu.append(self.menu)

        self.button = ui.Thumb('collect_new',self,setter=self.advance,label='Capture',hotkey='c')
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


    def do_open(self):
        if not self._window:
            self.window_should_open = True

    def get_count(self):
        return self.count

    def advance(self,_):
        if self.count == self.nb_img:
            logger.info("Capture %s calibration patterns."%self.nb_img)
            self.button.status_text = "%i to go" %(self.count)
            self.calculated = False
            self.img_points = []
            self.obj_points = []


        self.collect_new = True

    def open_window(self):
        if not self._window:
            if self.fullscreen:
                monitor = glfwGetMonitors()[self.monitor_idx]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width = 640,480

            self._window = glfwCreateWindow(height, width, "Calibration", monitor=monitor, share=glfwGetCurrentContext())
            if not self.fullscreen:
                glfwSetWindowPos(self._window,200,0)


            #Register callbacks
            glfwSetFramebufferSizeCallback(self._window,on_resize)
            glfwSetKeyCallback(self._window,self.on_key)
            glfwSetWindowCloseCallback(self._window,self.on_close)
            glfwSetMouseButtonCallback(self._window,self.on_button)

            on_resize(self._window,*glfwGetFramebufferSize(self._window))


            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()
            glfwMakeContextCurrent(active_window)

            self.clicks_to_close = 5



    def on_key(self,window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                self.on_close()


    def on_button(self,window,button, action, mods):
        if action ==GLFW_PRESS:
            self.clicks_to_close -=1
        if self.clicks_to_close ==0:
            self.on_close()


    def on_close(self,window=None):
        self.window_should_close = True

    def close_window(self):
        self.window_should_close=False
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None


    def calculate(self):
        self.count = 10
        rms, camera_matrix, dist_coefs, rot_vectors, trans_vectors = cv2.calibrateCamera(np.array(self.obj_points), np.array(self.img_points), self.g_pool.capture.frame_size,None,None)
        logger.info("Calibrated Camera, RMS:%s"%rms)

        tot_error = 0
        for i in xrange(len(self.obj_points)):
            img_points2, _ = cv2.projectPoints(self.obj_points[i], rot_vectors[i], trans_vectors[i], camera_matrix, dist_coefs)
            current_error = cv2.norm(self.img_points[i],img_points2, cv2.NORM_L2)/len(img_points2)
            tot_error += current_error
        error = tot_error/len(self.obj_points)
        logger.info("Error:%s"%error)
        print error, rms

        camera_calibration = {'camera_matrix':camera_matrix,'dist_coefs':dist_coefs,'camera_name':self.g_pool.capture.name,'resolution':self.g_pool.capture.frame_size,'error':error}
        save_object(camera_calibration,os.path.join(self.g_pool.user_dir,"camera_calibration"))
        logger.info("Calibration saved to user folder")
        self.camera_intrinsics = camera_matrix,dist_coefs,self.g_pool.capture.frame_size,error
        self.show_undistortion_switch.read_only=False
        self.calibrate_img = None
        self.calculated = True


    def update(self,frame,events):
        #re-init if the user change one of the parameters
        if self.last_nb_cols!=self.nb_cols or self.last_nb_rows!=self.nb_rows or self.last_nb_img!=self.nb_img:
            self.count = self.nb_img
            self.img_points = []
            self.obj_points = []
            self.last_nb_img = self.nb_img
            self.last_nb_cols = self.nb_cols
            self.last_nb_rows = self.nb_rows
            self.obj_grid = _gen_pattern_grid((self.nb_rows-1,self.nb_cols-1))
            self.collect_new = False
            self.button.status_text = ''
            

        if self.collect_new:
            self.calibrate_img = frame.img
            gray = frame.gray
            ret, corners = cv2.findChessboardCorners(gray, (self.nb_rows-1,self.nb_cols-1),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                self.obj_points.append(self.obj_grid)
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                self.img_points.append(corners)
                self.collect_new = False
                self.count -=1
                self.button.status_text = "%i to go"%(self.count)

            cv2.drawChessboardCorners(self.calibrate_img, (self.nb_rows-1,self.nb_cols-1), corners,ret)


        elif self.nb_frame>0 :
            self.nb_frame-=1
        else : 
            self.calibrate_img = None
            self.nb_frame = 10


        if self.count<=0 and not self.calculated:
            self.calculate()
            self.button.status_text = ''

        if self.window_should_close:
            self.close_window()

        if self.show_undistortion:
            h,w = frame.img.shape[:2]
            self.undist_img = undistord(img=frame.img, cm=self.camera_intrinsics[0], dist_coef=self.camera_intrinsics[1], size=(w, h))


    def gl_display(self):

        for grid_points in self.img_points:
            calib_bounds =  cv2.convexHull(grid_points)[:,0] #we dont need that extra encapsulation that opencv likes so much
            draw_polyline(calib_bounds,1,RGBA(0.,0.,1.,.5),line_type=gl.GL_LINE_LOOP)

        if self._window:
            self.gl_display_in_window()

        if self.show_undistortion:
            gl.glPushMatrix()
            make_coord_system_norm_based()
            apply_gl_texture(self.undist_img)
            gl.glPopMatrix()
        elif self.calibrate_img != None:
            gl.glPushMatrix()
            make_coord_system_norm_based()
            draw_gl_texture(self.calibrate_img)
            gl.glPopMatrix()


    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(self._window)

        clear_gl_screen()

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        p_window_size = glfwGetWindowSize(self._window)
        r = p_window_size[0]/15.
        # compensate for radius of marker
        gl.glOrtho(-r,p_window_size[0]+r,p_window_size[1]+r,-r ,-1,1)
        # Switch back to Model View Matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        #hacky way of scaling and fitting in different window rations/sizes
        grid = _make_grid()*min((p_window_size[0],p_window_size[1]*5.5/4.))
        #center the pattern
        grid -= np.mean(grid)
        grid +=(p_window_size[0]/2-r,p_window_size[1]/2+r)

        #draw_points(grid,size=r,color=RGBA(0.,0.,0.,1),sharpness=0.95)

        if self.clicks_to_close <5:
            self.glfont.set_size(int(p_window_size[0]/30.))
            self.glfont.draw_text(p_window_size[0]/2.,p_window_size[1]/4.,'Touch %s more times to close window.'%self.clicks_to_close)

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)


    def get_init_dict(self):
        return {}


    def cleanup(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        if you have a gui or glfw window destroy it here.
        """
        if self._window:
            self.close_window()
        self.deinit_gui()


def _gen_pattern_grid(size=(6,9)):
    objp = np.zeros((size[0]*size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)
    return np.asarray(objp, dtype='f4')


#def _make_grid(dim=(6,9)):
#    """
#    this function generates the structure for an asymmetrical circle grid
#    """
#    x,y = range(dim[0]),range(dim[1])
#    p = np.array([[[s,i] for s in x] for i in y], dtype=np.float32)
#    p[:,1::2,1] += 0.5
#    p = np.reshape(p, (-1,2), 'F')
#
#    # scale height = 1
#    x_scale =  1./(np.amax(p[:,0])-np.amin(p[:,0]))
#    y_scale =  1./(np.amax(p[:,1])-np.amin(p[:,1]))
#
#    p *=x_scale,x_scale/.5
#
#    return p


