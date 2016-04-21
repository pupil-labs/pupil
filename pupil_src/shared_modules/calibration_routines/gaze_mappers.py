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
import calibrate
from methods import project_distort_pts , normalize, spherical_to_cart
from copy import deepcopy
import numpy as np
from pyglui import ui
import math_helper

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
    def __init__(self, g_pool, eye_camera_to_world_matrix , camera_intrinsics ,cal_points_3d, cal_ref_points_3d, cal_gaze_points_3d, gaze_distance = 500 ):
        super(Vector_Gaze_Mapper, self).__init__(g_pool)
        self.eye_camera_to_world_matrix  =  eye_camera_to_world_matrix
        self.rotation_matrix = self.eye_camera_to_world_matrix[:3,:3]
        self.rotation_vector = cv2.Rodrigues( self.rotation_matrix  )[0]
        self.translation_vector  = self.eye_camera_to_world_matrix[:3,3]
        self.camera_matrix = camera_intrinsics['camera_matrix']
        self.dist_coefs = camera_intrinsics['dist_coefs']
        self.camera_intrinsics = camera_intrinsics
        self.cal_points_3d = cal_points_3d
        self.cal_ref_points_3d = cal_ref_points_3d
        self.cal_gaze_points_3d = cal_gaze_points_3d
        self.visualizer = Calibration_Visualizer(g_pool, camera_intrinsics ,cal_points_3d, cal_ref_points_3d,eye_camera_to_world_matrix, cal_gaze_points_3d)
        self.g_pool = g_pool
        self.gaze_pts_debug = []
        self.sphere = {}
        self.gaze_distance = gaze_distance
        self.visualizer.open_window()

    def open_close_window(self,new_state):
        if new_state:
            self.visualizer.open_window()
        else:
            self.visualizer.close_window()

    def toWorld(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matrix , point)[:3]

    def init_gui(self):
        self.menu = ui.Growing_Menu('Monocular 3D gaze mapper')
        self.g_pool.sidebar.insert(3,self.menu)
        self.menu.append(ui.Switch('debug window',setter=self.open_close_window, getter=lambda: bool(self.visualizer.window) ))
        self.menu.append(ui.Slider('gaze_distance',self,min=50,max=2000,label='gaze distance mm'))


    def update(self,frame,events):

        gaze_pts = []
        for p in events['pupil_positions']:
            if p['method'] == '3d c++' and p['confidence'] > self.g_pool.pupil_confidence_threshold:

                gaze_point =  np.array(p['circle_3d']['normal'] ) * self.gaze_distance  + np.array( p['sphere']['center'] )

                image_point, _  =  cv2.projectPoints( np.array([gaze_point]) , self.rotation_vector, self.translation_vector , self.camera_matrix , self.dist_coefs )
                image_point = image_point.reshape(-1,2)
                image_point = normalize( image_point[0], (frame.width, frame.height) , flip_y = True)

                eye_center = self.toWorld(p['sphere']['center'])
                gaze_3d = self.toWorld(gaze_point)
                normal_3d = np.dot( self.rotation_matrix, np.array( p['circle_3d']['normal'] ) )

                gaze_pts.append({   'norm_pos':image_point,
                                    'eye_center_3d':eye_center.tolist(),
                                    'gaze_normal_3d':normal_3d.tolist(),
                                    'gaze_point_3d':gaze_3d.tolist(),
                                    'confidence':p['confidence'],
                                    'timestamp':p['timestamp'],
                                    'base':[p]})

                if self.visualizer.window:
                    self.gaze_pts_debug.append( gaze_3d )
                    self.sphere['center'] = eye_center #eye camera coordinates
                    self.sphere['radius'] = p['sphere']['radius']

        events['gaze_positions'] = gaze_pts

    def gl_display(self):
        self.visualizer.update_window( self.g_pool , self.gaze_pts_debug , self.sphere)
        self.gaze_pts_debug = []

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
       return {'eye_camera_to_world_matrix':self.eye_camera_to_world_matrix ,'cal_points_3d':self.cal_points_3d,'cal_ref_points_3d':self.cal_ref_points_3d, 'cal_gaze_points_3d':self.cal_gaze_points_3d,  "camera_intrinsics":self.camera_intrinsics,'gaze_distance':self.gaze_distance}

    def cleanup(self):
        self.deinit_gui()
        self.visualizer.close_window()


class Binocular_Vector_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Vector_Gaze_Mapper"""
    def __init__(self, g_pool, eye_camera_to_world_matrix0, eye_camera_to_world_matrix1 , camera_intrinsics , cal_points_3d = [],cal_ref_points_3d = [], cal_gaze_points0_3d = [], cal_gaze_points1_3d = [] ):
        super(Binocular_Vector_Gaze_Mapper, self).__init__(g_pool)

        self.eye_camera_to_world_matrix0  =  eye_camera_to_world_matrix0
        self.rotation_matrix0  =  eye_camera_to_world_matrix0[:3,:3]
        self.rotation_vector0 = cv2.Rodrigues( self.eye_camera_to_world_matrix0[:3,:3]  )[0]
        self.translation_vector0  = self.eye_camera_to_world_matrix0[:3,3]

        self.eye_camera_to_world_matrix1  =  eye_camera_to_world_matrix1
        self.rotation_matrix1  =  eye_camera_to_world_matrix1[:3,:3]
        self.rotation_vector1 = cv2.Rodrigues( self.eye_camera_to_world_matrix1[:3,:3]  )[0]
        self.translation_vector1  = self.eye_camera_to_world_matrix1[:3,3]

        self.cal_points_3d = cal_points_3d
        self.cal_ref_points_3d = cal_ref_points_3d

        self.cal_gaze_points0_3d = cal_gaze_points0_3d #save for debug window
        self.cal_gaze_points1_3d = cal_gaze_points1_3d #save for debug window

        self.camera_matrix = camera_intrinsics['camera_matrix']
        self.dist_coefs = camera_intrinsics['dist_coefs']
        self.camera_intrinsics = camera_intrinsics
        self.visualizer = Calibration_Visualizer(g_pool,
                                                world_camera_intrinsics=camera_intrinsics ,
                                                cal_ref_points_3d = cal_points_3d,
                                                cal_observed_points_3d = cal_ref_points_3d,
                                                eye_camera_to_world_matrix0=  eye_camera_to_world_matrix0,
                                                cal_gaze_points0_3d = cal_gaze_points0_3d,
                                                eye_camera_to_world_matrix1=  eye_camera_to_world_matrix1,
                                                cal_gaze_points1_3d =  cal_gaze_points1_3d)
        self.g_pool = g_pool
        self.visualizer.open_window()
        self.gaze_pts_debug0 = []
        self.gaze_pts_debug1 = []
        self.intersection_points_debug = []
        self.sphere0 = {}
        self.sphere1 = {}
        self.last_gaze_distance = 0.0



    def open_close_window(self,new_state):
        if new_state:
            self.visualizer.open_window()
        else:
            self.visualizer.close_window()


    def init_gui(self):
        self.menu = ui.Growing_Menu('Binocular 3D gaze mapper')
        self.g_pool.sidebar.insert(3,self.menu)
        # self.menu.append(ui.Slider('last_gaze_distance',self,min=50,max=2000,label='gaze distance mm'))
        self.menu.append(ui.Switch('debug window',setter=self.open_close_window, getter=lambda: bool(self.visualizer.window) ))


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
            gaze_pts = self.map_binocular_intersect(pupil_pts_0, pupil_pts_1 ,frame )
        # fallback to monocular if something went wrong
        else:
            for p in pupil_pts_0:

                gaze_point =  np.array(p['circle_3d']['normal'] ) * self.last_gaze_distance  + np.array( p['sphere']['center'] )

                image_point, _  =  cv2.projectPoints( np.array([gaze_point]) , self.rotation_vector0, self.translation_vector0 , self.camera_matrix , self.dist_coefs )
                image_point = image_point.reshape(-1,2)
                image_point = normalize( image_point[0], (frame.width, frame.height) , flip_y = True)
                gaze_pts.append({'norm_pos':image_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})

                eye_center = self.eye0_to_World(p['sphere']['center'])
                gaze_3d = self.eye0_to_World(gaze_point)
                normal_3d = np.dot( self.rotation_matrix0, np.array( p['circle_3d']['normal'] ) )
                gaze_pts.append({   'norm_pos':image_point,
                                    'eye_centers_3d':{p['id']:eye_center.tolist()},
                                    'gaze_normals_3d':{p['id']:normal_3d.tolist()},
                                    'gaze_point_3d':gaze_3d.tolist(),
                                    'confidence':p['confidence'],
                                    'timestamp':p['timestamp'],
                                    'base':[p]})


                if self.visualizer.window:
                    self.gaze_pts_debug0.append(gaze_3d)
                    self.sphere0['center'] = eye_center
                    self.sphere0['radius'] = p['sphere']['radius']


            for p in pupil_pts_1:

                gaze_point =  np.array(p['circle_3d']['normal'] ) * self.last_gaze_distance  + np.array( p['sphere']['center'] )

                image_point, _  =  cv2.projectPoints( np.array([gaze_point]) , self.rotation_vector1, self.translation_vector1 , self.camera_matrix , self.dist_coefs )
                image_point = image_point.reshape(-1,2)
                image_point = normalize( image_point[0], (frame.width, frame.height) , flip_y = True)
                eye_center = self.eye1_to_World(p['sphere']['center'])
                gaze_3d = self.eye1_to_World(gaze_point)
                normal_3d = np.dot( self.rotation_matrix1, np.array( p['circle_3d']['normal'] ) )
                gaze_pts.append({   'norm_pos':image_point,
                                    'eye_centers_3d':{p['id']:eye_center.tolist()},
                                    'gaze_normals_3d':{p['id']:normal_3d.tolist()},
                                    'gaze_point_3d':gaze_3d.tolist(),
                                    'confidence':p['confidence'],
                                    'timestamp':p['timestamp'],
                                    'base':[p]})

                if self.visualizer.window:
                    self.gaze_pts_debug0.append(gaze_3d)
                    self.sphere0['center'] = eye_center
                    self.sphere0['radius'] = p['sphere']['radius']

        events['gaze_positions'] = gaze_pts


    def eye0_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matrix0 , point)[:3]

    def eye1_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matrix1 , point)[:3]

    def map_binocular_intersect(self, pupil_pts_0, pupil_pts_1 ,frame ):
        # maps gaze with binocular mapping
        # requires each list to contain at least one item!
        # returns 1 gaze point at minimum
        gaze_pts = []
        p0 = pupil_pts_0.pop(0)
        p1 = pupil_pts_1.pop(0)
        while True:

            #find the nearest intersection point of the two gaze lines
            # a line is defined by two point
            s0_center = self.eye0_to_World( np.array( p0['sphere']['center'] ) )
            s1_center = self.eye1_to_World( np.array( p1['sphere']['center'] ) )

            s0_normal = np.dot( self.rotation_matrix0, np.array( p0['circle_3d']['normal'] ) )
            s1_normal = np.dot( self.rotation_matrix1, np.array( p1['circle_3d']['normal'] ) )

            gaze_line0 = [ s0_center, s0_center + s0_normal ]
            gaze_line1 = [ s1_center, s1_center + s1_normal ]

            nearest_intersection_point , intersection_distance = math_helper.nearest_intersection( gaze_line0, gaze_line1 )

            if nearest_intersection_point is not None :

                self.last_gaze_distance = np.sqrt( nearest_intersection_point.dot( nearest_intersection_point ) )

                image_point, _  =  cv2.projectPoints( np.array([nearest_intersection_point]) ,  np.array([0.0,0.0,0.0]) ,  np.array([0.0,0.0,0.0]) , self.camera_matrix , self.dist_coefs )
                image_point = image_point.reshape(-1,2)
                image_point = normalize( image_point[0], (frame.width, frame.height) , flip_y = True)

                confidence = (p0['confidence'] + p1['confidence'])/2.
                ts = (p0['timestamp'] + p1['timestamp'])/2.
                gaze_pts.append({   'norm_pos':image_point,
                                    'eye_centers_3d':{0:s0_center.tolist(),1:s1_center.tolist()},
                                    'gaze_normals_3d':{0:s0_normal.tolist(),1:s1_normal.tolist()},
                                    'gaze_point_3d':nearest_intersection_point.tolist(),
                                    'confidence':confidence,
                                    'timestamp':ts,
                                    'base':[p0,p1]})
            else:
                logger.debug('No gaze line intersection point found')


            if self.visualizer.window:

                gaze0_3d =  s0_normal * self.last_gaze_distance  + s0_center
                gaze1_3d =  s1_normal * self.last_gaze_distance  + s1_center
                self.gaze_pts_debug0.append(  gaze0_3d)
                self.gaze_pts_debug1.append(  gaze1_3d)
                if nearest_intersection_point is not None:
                    self.intersection_points_debug.append( nearest_intersection_point )

                self.sphere0['center'] = s0_center #eye camera coordinates
                self.sphere0['radius'] = p0['sphere']['radius']

                self.sphere1['center'] = s1_center #eye camera coordinates
                self.sphere1['radius'] = p1['sphere']['radius']



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
       return {'eye_camera_to_world_matrix0':self.eye_camera_to_world_matrix0 ,'eye_camera_to_world_matrix1':self.eye_camera_to_world_matrix1 ,'cal_ref_points_3d':self.cal_ref_points_3d, 'cal_gaze_points0_3d':self.cal_gaze_points0_3d, 'cal_gaze_points1_3d':self.cal_gaze_points1_3d,  "camera_intrinsics":self.camera_intrinsics}


    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None


    def cleanup(self):
        self.deinit_gui()
        self.visualizer.close_window()
