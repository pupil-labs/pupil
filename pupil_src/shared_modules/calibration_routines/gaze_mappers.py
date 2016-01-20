'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Gaze_Mapping_Plugin
import cv2
from calibrate import make_map_function
from methods import project_distort_pts , normalize, spherical_to_cart
from copy import deepcopy
import numpy as np
from pyglui import ui

from visualizer_calibration import *

class Dummy_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Dummy_Gaze_Mapper"""
    def __init__(self, g_pool):
        super(Dummy_Gaze_Mapper, self).__init__(g_pool)

    def update(self,frame,events):
        gaze_pts = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                gaze_pts.append({'norm_pos':p['norm_pos'][:],'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})

        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {}


class Simple_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Simple_Gaze_Mapper"""
    def __init__(self, g_pool,params):
        super(Simple_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.map_fn = make_map_function(*self.params)

    def update(self,frame,events):
        gaze_pts = []

        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                gaze_point = self.map_fn(p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})

        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params}


class Binocular_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self, g_pool,params,params_eye0,params_eye1):
        super(Gaze_Mapping_Plugin, self).__init__(g_pool)
        self.params = params
        self.params_eye0 = params_eye0
        self.params_eye1 = params_eye1
        self.multivariate = True
        self.map_fn = make_map_function(*self.params)
        self.map_fn_fallback = []
        self.map_fn_fallback.append(make_map_function(*self.params_eye0))
        self.map_fn_fallback.append(make_map_function(*self.params_eye1))

    def init_gui(self):
        self.menu = ui.Growing_Menu('Binocular Gaze Mapping')
        self.g_pool.sidebar.insert(3,self.menu)
        self.menu.append(ui.Switch('multivariate',self,on_val=True,off_val=False,label='Multivariate Mode'))

    def update(self,frame,events):

        pupil_pts_0 = []
        pupil_pts_1 = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                if p['id'] == 0:
                    pupil_pts_0.append(p)
                else:
                    pupil_pts_1.append(p)

        # try binocular mapping (needs at least 1 pupil position in each list)
        gaze_pts = []
        if len(pupil_pts_0) > 0 and len(pupil_pts_1) > 0:
            gaze_pts = self._map_binocular(pupil_pts_0, pupil_pts_1, self.multivariate)
        # fallback to monocular if something went wrong
        else:
            for p in pupil_pts_0:
                gaze_point = self.map_fn_fallback[0](p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})
            for p in pupil_pts_1:
                gaze_point = self.map_fn_fallback[1](p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})

        events['gaze_positions'] = gaze_pts

    def _map_binocular(self, pupil_pts_0, pupil_pts_1,multivariate=True):
        # maps gaze with binocular mapping
        # requires each list to contain at least one item!
        # returns 1 gaze point at minimum
        gaze_pts = []
        p0 = pupil_pts_0.pop(0)
        p1 = pupil_pts_1.pop(0)
        while True:
            if multivariate:
                gaze_point = self.map_fn(p0['norm_pos'], p1['norm_pos'])
            else:
                gaze_point_eye0 = self.map_fn_fallback[0](p0['norm_pos'])
                gaze_point_eye1 = self.map_fn_fallback[1](p1['norm_pos'])
                gaze_point = (gaze_point_eye0[0] + gaze_point_eye1[0])/2. , (gaze_point_eye0[1] + gaze_point_eye1[1])/2.
            confidence = (p0['confidence'] + p1['confidence'])/2.
            ts = (p0['timestamp'] + p1['timestamp'])/2.
            gaze_pts.append({'norm_pos':gaze_point,'confidence':confidence,'timestamp':ts,'base':[p0, p1]})

            # keep sample with higher timestamp and increase the one with lower timestamp
            if p0['timestamp'] <= p1['timestamp'] and pupil_pts_0:
                p0 = pupil_pts_0.pop(0)
                continue
            elif p1['timestamp'] <= p0['timestamp'] and pupil_pts_1:
                p1 = pupil_pts_1.pop(0)
                continue
            elif pupil_pts_0 and not pupil_pts_1:
                p0 = pupil_pts_0.pop(0)
            elif pupil_pts_1 and not pupil_pts_0:
                p1 = pupil_pts_1.pop(0)
            else:
                break

        return gaze_pts

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def cleanup(self):
        self.deinit_gui()

    def get_init_dict(self):
        return {'params':self.params, 'params_eye0':self.params_eye0, 'params_eye1':self.params_eye1}



class Vector_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Vector_Gaze_Mapper"""
    def __init__(self, g_pool, eye_to_world_matrix , camera_intrinsics , cal_ref_points_3d = [], cal_gaze_points_3d = [] , gaze_distance = 500 ):
        super(Vector_Gaze_Mapper, self).__init__(g_pool)
        self.eye_to_world_matrix  =  eye_to_world_matrix
        self.rotation_vector = cv2.Rodrigues( self.eye_to_world_matrix[:3,:3]  )[0]
        self.translation_vector  = self.eye_to_world_matrix[:3,3]
        self.camera_matrix = camera_intrinsics['camera_matrix']
        self.dist_coefs = camera_intrinsics['dist_coefs']
        self.camera_intrinsics = camera_intrinsics
        self.cal_ref_points_3d = cal_ref_points_3d
        self.cal_gaze_points_3d = cal_gaze_points_3d
        self.visualizer = Calibration_Visualizer(g_pool, camera_intrinsics , cal_ref_points_3d,eye_to_world_matrix, cal_gaze_points_3d)
        self.g_pool = g_pool
        self.visualizer.open_window()
        self.gaze_pts_debug = []
        self.sphere = None
        self.gaze_distance = gaze_distance

    def update(self,frame,events):
        gaze_pts = []
        for p in events['pupil_positions']:
            if p['method'] == '3D c++' and p['confidence'] > self.g_pool.pupil_confidence_threshold:

                gaze_point =  np.array(p['circle3D']['normal'] ) * self.gaze_distance  + np.array( p['sphere']['center'] )

                gaze_point *= 1.0,-1.0,1.0
                self.gaze_pts_debug.append( gaze_point )
                image_point, _  =  cv2.projectPoints( np.array([gaze_point]) , self.rotation_vector, self.translation_vector , self.camera_matrix , self.dist_coefs )
                image_point = image_point.reshape(-1,2)
                image_point = normalize( image_point[0], (frame.width, frame.height) , flip_y = True)
                gaze_pts.append({'norm_pos':image_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})

                self.sphere = p['sphere']


        events['gaze_positions'] = gaze_pts

    def gl_display(self):
        self.visualizer.update_window( self.g_pool , self.gaze_pts_debug , self.sphere)
        self.gaze_pts_debug = []

    def get_init_dict(self):
       return {'eye_to_world_matrix':self.eye_to_world_matrix ,'cal_ref_points_3d':self.cal_ref_points_3d, 'cal_gaze_points_3d':self.cal_gaze_points_3d,  "camera_intrinsics":self.camera_intrinsics,'gaze_distance':self.gaze_distance}

    def cleanup(self):
        self.visualizer.close_window()


class Binocular_Vector_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Vector_Gaze_Mapper"""
    def __init__(self, g_pool, eye_to_world_matrix0, eye_to_world_matrix1 , camera_intrinsics , cal_ref_points_3d = [], cal_gaze_points0_3d = [], cal_gaze_points1_3d = [], gaze_distance = 500 ):
        super(Binocular_Vector_Gaze_Mapper, self).__init__(g_pool)

        self.eye_to_world_matrix0  =  eye_to_world_matrix0
        self.rotation_vector0 = cv2.Rodrigues( self.eye_to_world_matrix0[:3,:3]  )[0]
        self.translation_vector0  = self.eye_to_world_matrix0[:3,3]

        self.eye_to_world_matrix1  =  eye_to_world_matrix1
        self.rotation_vector1 = cv2.Rodrigues( self.eye_to_world_matrix1[:3,:3]  )[0]
        self.translation_vector1  = self.eye_to_world_matrix1[:3,3]

        self.cal_ref_points_3d = cal_ref_points_3d

        self.cal_gaze_points0_3d = cal_gaze_points0_3d #save for debug window
        self.cal_gaze_points1_3d = cal_gaze_points1_3d #save for debug window

        self.camera_matrix = camera_intrinsics['camera_matrix']
        self.dist_coefs = camera_intrinsics['dist_coefs']
        self.camera_intrinsics = camera_intrinsics
        self.visualizer = Calibration_Visualizer(g_pool, camera_intrinsics ,cal_ref_points_3d, eye_to_world_matrix0, cal_gaze_points0_3d, eye_to_world_matrix1,  cal_gaze_points1_3d)
        self.g_pool = g_pool
        self.visualizer.open_window()
        self.gaze_pts_debug0 = []
        self.gaze_pts_debug1 = []
        self.intersection_points_debug = []
        self.sphere0 = None
        self.sphere1 = None
        self.gaze_distance = gaze_distance

    def update(self,frame,events):

        pupil_pts_0 = []
        pupil_pts_1 = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                if p['id'] == 0:
                    pupil_pts_0.append(p)
                else:
                    pupil_pts_1.append(p)

        # try binocular mapping (needs at least 1 pupil position in each list)
        gaze_pts = []
        if len(pupil_pts_0) > 0 and len(pupil_pts_1) > 0:
            gaze_pts = self.map_binocular(pupil_pts_0, pupil_pts_1 ,frame )
        # fallback to monocular if something went wrong
        else:
            print 'not implemented yet'
            #raise NotImplementedError()
            # for p in pupil_pts_0:
            #     gaze_point = self.map_fn_fallback[0](p['norm_pos'])
            #     gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})
            # for p in pupil_pts_1:
            #     gaze_point = self.map_fn_fallback[1](p['norm_pos'])
            #     gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})

        events['gaze_positions'] = gaze_pts


    def map_binocular(self, pupil_pts_0, pupil_pts_1 ,frame ):
        # maps gaze with binocular mapping
        # requires each list to contain at least one item!
        # returns 1 gaze point at minimum
        gaze_pts = []
        p0 = pupil_pts_0.pop(0)
        p1 = pupil_pts_1.pop(0)
        while True:


            #find the nearest intersection point of the two gaze lines
            # a line is defined by two point
            gaze_line0 = [np.zeros(4), np.zeros(4)]
            gaze_line0[0][:3] =  np.array( p0['sphere']['center'] )
            gaze_line0[1][:3] =  np.array( p0['sphere']['center'] ) + np.array(p0['circle3D']['normal'] ) * 10000.0

            gaze_line1 = [np.zeros(4), np.zeros(4)]
            gaze_line1[0][:3] =   np.array( p1['sphere']['center'] )
            gaze_line1[1][:3] = np.array( p1['sphere']['center'] ) + np.array( p1['circle3D']['normal'] ) * 10000.0

            gaze_line0[0][1] *= -1.0
            gaze_line0[1][1] *= -1.0
            gaze_line1[0][1] *= -1.0
            gaze_line1[1][1] *= -1.0

            #transform lines to world-coordinate system
            gaze_line_world0 = [np.zeros(3), np.zeros(3)]
            gaze_line_world0[0] = np.squeeze(np.asarray(self.eye_to_world_matrix0.dot(gaze_line0[0])))[:3]
            gaze_line_world0[1] = np.squeeze(np.asarray(self.eye_to_world_matrix0.dot(gaze_line0[1])))[:3]

            gaze_line_world1 = [np.zeros(3), np.zeros(3)]
            gaze_line_world1[0] = np.squeeze(np.asarray( self.eye_to_world_matrix1.dot(gaze_line1[0])))[:3]
            gaze_line_world1[1] = np.squeeze(np.asarray( self.eye_to_world_matrix1.dot(gaze_line1[1])))[:3]

            nearest_intersection_point = self.nearest_intersection( gaze_line_world0, gaze_line_world1 )

            image_point, _  =  cv2.projectPoints( np.array([nearest_intersection_point]) ,  np.array([0.0,0.0,0.0]) ,  np.array([0.0,0.0,0.0]) , self.camera_matrix , self.dist_coefs )
            image_point = image_point.reshape(-1,2)
            image_point = normalize( image_point[0], (frame.width, frame.height) , flip_y = True)

            confidence = (p0['confidence'] + p1['confidence'])/2.
            ts = (p0['timestamp'] + p1['timestamp'])/2.
            gaze_pts.append({'norm_pos':image_point,'confidence':confidence,'timestamp':ts,'base':[p0, p1]})

            self.sphere0 = p0['sphere']
            self.sphere1 = p1['sphere']

            # for debug
            self.gaze_pts_debug0.append(  gaze_line0[1][:3] )
            self.gaze_pts_debug1.append(  gaze_line1[1][:3] )
            self.intersection_points_debug.append( nearest_intersection_point )
            # keep sample with higher timestamp and increase the one with lower timestamp
            if p0['timestamp'] <= p1['timestamp'] and pupil_pts_0:
                p0 = pupil_pts_0.pop(0)
                continue
            elif p1['timestamp'] <= p0['timestamp'] and pupil_pts_1:
                p1 = pupil_pts_1.pop(0)
                continue
            elif pupil_pts_0 and not pupil_pts_1:
                p0 = pupil_pts_0.pop(0)
            elif pupil_pts_1 and not pupil_pts_0:
                p1 = pupil_pts_1.pop(0)
            else:
                break

        return gaze_pts


    def gl_display(self):
        self.visualizer.update_window( self.g_pool , self.gaze_pts_debug0 , self.sphere0, self.gaze_pts_debug1, self.sphere1, self.intersection_points_debug )
        self.gaze_pts_debug0 = []
        self.gaze_pts_debug1 = []
        self.intersection_points_debug = []

    def get_init_dict(self):
       return {'eye_to_world_matrix0':self.eye_to_world_matrix0 ,'eye_to_world_matrix1':self.eye_to_world_matrix1 ,'cal_ref_points_3d':self.cal_ref_points_3d, 'cal_gaze_points0_3d':self.cal_gaze_points0_3d, 'cal_gaze_points1_3d':self.cal_gaze_points1_3d,  "camera_intrinsics":self.camera_intrinsics,'gaze_distance':self.gaze_distance}

    def cleanup(self):
        self.visualizer.close_window()

    def nearest_intersection( self, line0 , line1 ):

        A1 = line0[0]
        A2 = line0[1]
        B1 = line1[0]
        B2 = line1[1]

        nA = np.dot(np.cross(B2-B1,A1-B1),np.cross(A2-A1,B2-B1));
        nB = np.dot(np.cross(A2-A1,A1-B1),np.cross(A2-A1,B2-B1));
        d = np.dot(np.cross(A2-A1,B2-B1),np.cross(A2-A1,B2-B1));
        A0 = A1 + (nA/d)*(A2-A1);
        B0 = B1 + (nB/d)*(B2-B1);

        nAB = A0 - B0;
        return B0 + nAB * 0.5


