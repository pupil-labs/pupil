'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os,platform
import cv2
import numpy as np
from file_methods import Persistent_Dict,load_object

import OpenGL.GL as gl
from gl_utils import make_coord_system_norm_based, make_coord_system_pixel_based
from gl_utils.drawing_utils import *
from pyglui.cygl.utils import draw_polyline,RGBA,draw_gl_texture
from pyglui import ui
from methods import gen_square_pattern_grid, is_inside_simple_polygone, undistord_with_roi, distance, distortPoints
from glfw import *
from plugin import Plugin
from OpenGL.GLU import *
#logging
import logging
logger = logging.getLogger(__name__)

from square_marker_detect import detect_markers_robust,detect_markers, draw_markers,m_marker_to_screen, init_prev_img
from reference_surface import Reference_Surface

"""
#threading like this is not good enough
class myThread (threading.Thread):
    def __init__(self, threadID, name, counter, camera_intrinsics, img):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.camera_intrinsics = camera_intrinsics
        self.img = img
    def run(self):
        global frame_img 
        frame_img = cv2.undistort(self.img, self.camera_intrinsics[0], self.camera_intrinsics[1],newCameraMatrix=self.camera_intrinsics[4])
"""

class World_Interaction(Plugin):
    """docstring
    """
    def __init__(self,g_pool,mode="Show marker IDs",min_marker_perimeter = 40):
        super(World_Interaction, self).__init__(g_pool)
        self.order = .2

        # all markers that are detected in the most recent frame
        self.markers = []
        init_prev_img()

        #load camera intrinsics
        try:
            camera_calibration = load_object(os.path.join(self.g_pool.user_dir,'camera_calibration'))
        except:
            self.camera_intrinsics = None
        else:
            same_name = camera_calibration['camera_name'] == self.g_pool.capture.name
            same_resolution =  camera_calibration['resolution'] == self.g_pool.capture.frame_size
            if same_name and same_resolution:
                logger.info('Loaded camera calibration.')
                A = camera_calibration['camera_matrix']
                dist_coefs = camera_calibration['dist_coefs']
                resolution = camera_calibration['resolution']
                error = camera_calibration['error']
                #roi = region of interrest, where the image have been crop
                new_cm, self.roi = cv2.getOptimalNewCameraMatrix(cameraMatrix= A, distCoeffs=dist_coefs, imageSize=resolution, alpha=0.7, newImgSize=resolution, centerPrincipalPoint=1)
                self.roi += 0., 0.
                self.affine_roi()
                self.camera_intrinsics = A,dist_coefs,resolution,error,new_cm
            else:
                logger.info('Loaded camera calibration but camera name and/or resolution has changed. Please re-calibrate.')
                self.camera_intrinsics = None

        # all registered markers
        self.markers_definitions = Persistent_Dict(os.path.join(g_pool.user_dir,'markers_definitions') )
        self.markers = [d for d in  self.markers_definitions.get('realtime_square_marker',[]) if isinstance(d,dict)]

        #quit if no calibrations
        if self.camera_intrinsics == None:
            logger.info('Please calibrate first')
            self.close()

        #plugin state
        self.mode = mode
        self.running = True

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        #self.axis = np.float32([[40,0,0], [0,40,0], [0,0,40]]).reshape(-1,3)

        self.robust_detection = 1
        self.min_marker_perimeter = min_marker_perimeter
        self.nb_frame_detection = 10

        #for display
        self.only_visible_m = True
        self.config_markers = 0
        self.glIsInit = False
        self.show_undistord = False
        self.resize_distord = False

        #debug vars
        self.img_shape = None
        self.frame_img = None
        self.camera_coord = None  #3D position
        self.current_gaze_pos = None # in pixel coordinate

        self.menu= None
        self.button=  None


    def close(self):
        self.alive = False

    def remove_marker(self,i):
        del self.markers[i]
        logger.info('Marker removed')
        self.update_gui_markers()

    def remove_non_visible_marker(self):
        list_to_keep = []
        for i, m in enumerate(self.markers):
            if m['visible'] == True:
                list_to_keep = list_to_keep + [m]
        self.markers = list_to_keep
        logger.info('Markers removed')
        self.update_gui_markers()

    def show_markers_conf(self):
        self.config_markers = not self.config_markers
        self.update_gui_markers()

    def get_gaze_pos(self, events):
        #return the current gaze position
        if events.get('gaze_positions',[]) == []:
            return None
        else:  
            cumul_x = 0.
            cumul_y = 0.
            count = 0
            min_x, min_y, w, h, offset_x, offset_y = self.roi
            camera_matrix = self.camera_intrinsics[0]

            #coord need to be rectify only if there is an important gap beetween the image shape and the shape given by camera_matrix
            if (camera_matrix[0][2] * 2 - self.img_shape[1]) > 5:
                rectify_x = (camera_matrix[0][2] * 2) - self.img_shape[1]
            else:
                rectify_x = 0
            if (camera_matrix[1][2] * 2 - self.img_shape[0]) > 5:
                rectify_y = (camera_matrix[1][2] * 2) - self.img_shape[0]
            else:
                rectify_y = 0

            for pt in events.get('gaze_positions',[]):
                x_coord = (pt['norm_pos'][0]*self.img_shape[1])  #in pixels
                y_coord = ((1-pt['norm_pos'][1])*self.img_shape[0])

                if(self.show_undistord):
                    #x_coord = (float(x_coord) / float(w)) * (self.img_shape[1])
                    #y_coord = (float(y_coord) / float(h)) * (self.img_shape[0])
                    #x_coord = float((x_coord) * (self.img_shape[1])) / float(self.img_shape[1] + rectify_x)
                    #y_coord = float((y_coord) * (self.img_shape[0])) / float(self.img_shape[0] + rectify_y)

                    pos = np.array([[[x_coord, y_coord]]], dtype = np.float64)
                    pos = cv2.undistortPoints( pos, camera_matrix, self.camera_intrinsics[1], P=self.camera_intrinsics[4] )

                    pos[0][0][0] = (float(pos[0][0][0]) / float(w)) * (self.img_shape[1]) - (float(self.img_shape[1])/float(w)) * min_x
                    pos[0][0][1] = (float(pos[0][0][1]) / float(h)) * (self.img_shape[0]) - (float(self.img_shape[0])/float(h)) * min_y
                else:
                    pos = np.array([[[x_coord, y_coord]]], dtype = np.float64)

                pt['norm_pos'] = (pos[0][0][0]/self.img_shape[1]), 1 - (pos[0][0][1]/self.img_shape[0])
                count += 1
                cumul_x += pt['norm_pos'][0]
                cumul_y += pt['norm_pos'][1]

            return np.array([cumul_x/float(count), cumul_y/float(count)])

    def affine_roi(self):
        """
        crop a little more to be sure to have the same crop ration between height/width
        offset_x and offset_y represent the values that had been subtract on respectively width and height
        """
        x, y, w, h, offset_x, offset_y = self.roi
        if h  == 0 or w == 0:
            return
        if (y/float(h) > x/float(w)):
            offset_x = (float(y*w)/float(h)) - x
            x += offset_x /2
            w -= offset_x
        else:
            offset_y = (float(x*h)/float(w)) - y
            y += offset_y /2
            h -= offset_y

        self.roi = x, y, w, h, offset_x, offset_y


    def is_looked_up(self, marker):
        """
        for the given marker at index in markers, detect if the marker is fixed since at least nb_frame_detection frames
        """
        tolerance_treshold = 20
        already_fixed_count = marker["frames_looked_up_count"]
        gp = self.current_gaze_pos
        polygone = marker["verts_norm"]

        if is_inside_simple_polygone(gp, polygone, -0.0015):
            already_fixed_count += 1
        else :
            already_fixed_count -= 1

        if already_fixed_count > (self.nb_frame_detection + tolerance_treshold):   #max clamping
            marker["frames_looked_up_count"] = self.nb_frame_detection + tolerance_treshold
            return True
        elif already_fixed_count >= self.nb_frame_detection:
            marker["frames_looked_up_count"] = already_fixed_count
            return True
        elif already_fixed_count < 0:  #min clamping
            marker["frames_looked_up_count"] = 0
            return False
        else :
            marker["frames_looked_up_count"] = already_fixed_count
            return False

    def find_main_marker(self, visible_markers):
        marker = None
        current_score = 0
        #the more dist_coeff is, the less main_marker will be distord
        dist_coeff = 5

        for m in visible_markers:
            verts = m['verts']
            size = (distance(verts[0], verts[1]) + distance(verts[1], verts[2]) + distance(verts[2], verts[3]) + distance(verts[3], verts[0])) / 4.

            vertical_distord = min(distance(verts[0], verts[1])/float(distance(verts[3], verts[2])),distance(verts[3], verts[2])/float(distance(verts[0], verts[1])))
            vertical_distord = vertical_distord * vertical_distord

            horizontal_distord = min(distance(verts[0], verts[3])/float(distance(verts[1], verts[2])),distance(verts[1], verts[2])/float(distance(verts[0], verts[3])))
            horizontal_distord = horizontal_distord * horizontal_distord
            distord_res = float(vertical_distord + horizontal_distord)/2
            distord_res = np.power(distord_res, dist_coeff)

            score = size * distord_res
            if score > current_score:
                current_score = score
                marker = m

        if marker != None:
            objp = gen_square_pattern_grid(marker['height'])
            # Find the rotation and translation vectors.

            if not self.show_undistord:
                verts = cv2.undistortPoints( marker['verts'], self.camera_intrinsics[0], self.camera_intrinsics[1], P=self.camera_intrinsics[0] )
            else:
                verts = np.zeros((4,1,2), dtype = np.float64)
                for i in range(4):
                    verts[i][0][0] = marker['verts'][i][0][0] + float(self.roi[0])
                    verts[i][0][1] = marker['verts'][i][0][1] + float(self.roi[1])

                verts = distortPoints(verts, self.camera_intrinsics[0], self.camera_intrinsics[1], new_cm=self.camera_intrinsics[4])
                verts = cv2.undistortPoints( verts, self.camera_intrinsics[0], self.camera_intrinsics[1], P=self.camera_intrinsics[0] )

            #deduce coordinate of the camera
            _, rvecs, tvecs = cv2.solvePnP(objp, verts, self.camera_intrinsics[0], None) #Already undistord, no need to give dist coeffs
            self.camera_coord = self.get_camera_coordinate(rvecs, tvecs)


        return marker

    def create_environnement(self, origin_marker, visible_marker):
        pass

    def init_gui(self):
        self.menu = ui.Growing_Menu('World Interaction')
        self.g_pool.sidebar.append(self.menu)

        self.button = ui.Thumb('running',self,label='Track',hotkey='t')
        self.button.on_color[:] = (.1,.2,1.,.8)
        self.g_pool.quickbar.append(self.button)
        self.update_gui_markers()

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu= None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None

    def update_gui_markers(self):
        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text('This plugin detects and tracks fiducial markers visible in the scene and, given the size of a marker, deduce world coordinate of the camera'))
        def change_undist_mode(val):
            self.show_undistord = val
            init_prev_img()

        def change_resize_undist_mode(val):
            self.resize_distord = val
            if val:  #find the right value to crop
                self.affine_roi()
            else:    #find the original roi
                _, self.roi = cv2.getOptimalNewCameraMatrix(cameraMatrix= self.camera_intrinsics[0], distCoeffs=self.camera_intrinsics[1], imageSize=self.camera_intrinsics[2], alpha=0.7,newImgSize=self.camera_intrinsics[2],centerPrincipalPoint=1)
                self.roi += 0., 0.
            init_prev_img()

        self.show_undistortion_switch = ui.Switch('show_undistord',self,label='show undistorted image',setter=change_undist_mode)
        self.menu.append(self.show_undistortion_switch)
        self.resize_distord_switch = ui.Switch('resize_distord',self,label='resize undistorted image',setter=change_resize_undist_mode)
        self.menu.append(self.resize_distord_switch)
        if not self.camera_intrinsics:
            self.show_undistord = False
            self.show_undistortion_switch.read_only=True

        self.menu.append(ui.Switch('robust_detection',self,label='Robust detection'))
        self.menu.append(ui.Slider('min_marker_perimeter',self,step=1,min=10,max=80))
        self.menu.append(ui.Slider('nb_frame_detection',self,step=1,min=1,max=60))
        self.menu.append(ui.Selector('mode',self,label="Mode",selection=['Show marker IDs','Draw obj']))
        self.menu.append(ui.Button('Configure markers',self.show_markers_conf))
        def set_var(val,i,target):
            self.markers[i][target] = val
            try:
                self.markers[i]['obj'] = OBJ("../ressources/"+self.markers[i]['obj_name'], self.markers[i]['mult'], swapyz=True)
                logger.info('File:%s loaded'%self.markers[i]['obj_name'])
            except IOError:
                self.markers[i]['obj_name'] = "None"
                logger.warning('File:%s does not exist'%self.markers[i]['obj_name'])
                self.markers[i]['obj'] = None

        def change_bool_val(new_val):
            self.only_visible_m = new_val
            self.update_gui_markers()

        if self.config_markers:
            self.menu.append(ui.Button('Remove all non-visible markers',self.remove_non_visible_marker))
            self.menu.append(ui.Switch('only_visible_m',self,label='only visible markers',setter=change_bool_val))
            for index, m in enumerate(self.markers):
                if(not self.only_visible_m) or m['visible']:
                    s_menu = ui.Growing_Menu("Marker %s"%m['id'])
                    def make_remove_s(i):
                        return lambda: self.remove_marker(i)
                    remove_s = make_remove_s(index)
                    s_menu.append(ui.Button('Remove',remove_s))
                    s_menu.append(ui.Text_Input('height',m,'Height (mm) :'))

                    def make_setter_obj(index):
                        return lambda val: set_var(val,index,'obj_name')
                    setter_obj = make_setter_obj(index)
                    s_menu.append(ui.Text_Input('obj_name',m,'Obj file :',setter=setter_obj))

                    def make_setter_mult(index):
                        return lambda val: set_var(val,index,'mult')
                    setter_mult = make_setter_mult(index)
                    s_menu.append(ui.Text_Input('mult',m,'Multiplicative coef :',setter=setter_mult))
                    s_menu.collapsed=True
                    self.menu.append(s_menu)


    def update(self,frame,events):
        self.img_shape = frame.height,frame.width,3

        if self.running:
            self.current_gaze_pos = self.get_gaze_pos(events)

            #drawing only in undistord image
            if self.show_undistord:
                self.frame_img = undistord_with_roi(img=frame.img, cm=self.camera_intrinsics[0], dist_coef=self.camera_intrinsics[1], roi=self.roi, new_cm=self.camera_intrinsics[4])
                gray = cv2.cvtColor(self.frame_img, cv2.COLOR_BGR2GRAY)
                cv2.imshow("test", self.frame_img)
            else:
                self.frame_img = frame.img
                gray = frame.gray


            if self.robust_detection:
                visible_markers = detect_markers_robust(gray,
                                                    grid_size = 5,
                                                    prev_markers=self.markers,
                                                    min_marker_perimeter=self.min_marker_perimeter,
                                                    aperture = 11, 
                                                    visualize=0,
                                                    true_detect_every_frame=3)
            else:
                visible_markers = detect_markers(gray,
                                                grid_size = 5,
                                                min_marker_perimeter=self.min_marker_perimeter,
                                                aperture = 11, 
                                                visualize=0)

            for m in self.markers:
                m['visible'] = False

            self.find_main_marker(visible_markers )

            for vm in visible_markers:
                #find the index of the visible marker in self.markers
                index = -1
                for indexList,m in enumerate(self.markers):
                    if m['id'] == vm['id']:
                        index = indexList
                        break

                if index == -1:  #marker is not registered already
                    index = len(self.markers)
                    new_marker = {'id':vm['id'],'verts':vm['verts'],'verts_norm':vm['verts_norm'],'centroid':vm['centroid'],'frames_since_true_detection':0,'height':76,'frames_looked_up_count':0,'obj_name':"None",'obj':None,'mult':1}
                    self.markers.append(new_marker)
                marker = self.markers[index]

                marker['verts'] = vm['verts']
                marker['verts_norm'] = vm['verts_norm']
                marker['centroid'] = vm['centroid']
                marker['frames_since_true_detection'] = vm['frames_since_true_detection']
                marker['visible'] = True
                objp = gen_square_pattern_grid(marker['height'])

                # Find the rotation and translation vectors.
                _, rvecs, tvecs = cv2.solvePnP(objp, marker['verts'], self.camera_intrinsics[0], None) #Already undistord, no need to give dist coeffs

                #if the marker is fixed by the gaze
                if self.is_looked_up(marker):
                    #get the obj to draw
                    if self.mode == "Draw obj":
                        if marker['obj'] == None and marker['obj_name'] != "None":
                            marker['obj'] = OBJ("../ressources/"+marker['obj_name'], marker['mult'], swapyz=True)

                        marker['rot'] = rvecs
                        marker['trans'] = tvecs
                        marker['to_draw'] = True

                else :  #not fixed
                    if self.mode == "Draw obj":
                       marker['to_draw'] = False

        if not self.running:
            self.button.status_text = 'tracking paused'

    def gl_display(self):
        
        if self.running:
            #apply undistord image to draw
            glMatrixMode( GL_MODELVIEW )
            make_coord_system_norm_based()
            apply_gl_texture(self.frame_img)

            if self.show_undistord:
                make_coord_system_pixel_based([self.roi[3], self.roi[2], 3])
            else:
                make_coord_system_pixel_based(self.img_shape)

            if self.mode == "Show marker IDs":
                visible = [m for m in self.markers if m['visible']]
                if self.show_undistord:
                    draw_markers(self.img_shape, visible, self.roi)
                else:
                    draw_markers(self.img_shape, visible)

            if self.mode == "Draw obj" and self.frame_img != None:

                if not self.glIsInit :
                    glInit()
                    self.glIsInit = True

                for m in self.markers:
                    if m['visible']:
                        if m['to_draw']:

                            #drawIn2D( self.camera_intrinsics[0], self.camera_intrinsics[1], m )
                            if self.show_undistord:
                                glDrawFromCamera( self.camera_intrinsics[0], self.camera_intrinsics[1], m['rot'], m['trans'], self.img_shape, self.roi, m['obj'] )
                            else:
                                glDrawFromCamera( self.camera_intrinsics[0], self.camera_intrinsics[1], m['rot'], m['trans'], self.img_shape, None, m['obj'] )

                self.frame_img = None


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        Keep all informations about markers.
        """
        def data_to_save(m):
            return {'id':m['id'],'verts':m['verts'],'verts_norm':m['verts_norm'],'centroid':m['centroid'],'frames_since_true_detection':0,
                    'height':m['height'],'frames_looked_up_count':0,'obj_name':m['obj_name'],'obj':None,'mult':m['mult']}


        self.markers_definitions["realtime_square_marker"] = [data_to_save(m) for m in self.markers]
        self.markers_definitions.close()

        self.deinit_gui()


    def get_camera_coordinate(self, rvecs, tvecs):
        #deduce coordinate of the camera
        #with (0, 0, 0) is the marker
        rmat = cv2.Rodrigues(rvecs)[0]
        transformMat = -1 * np.matrix(rmat).T * np.matrix(tvecs)

        return transformMat.getA()

