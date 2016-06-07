'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
import cv2
from calibrate import make_map_function
import calibrate
from methods import project_distort_pts , normalize, spherical_to_cart
from copy import deepcopy
import numpy as np
from pyglui import ui
import math_helper
import zmq_tools
import threading
from multiprocessing import Process as Thread

from visualizer_calibration import *


class Gaze_Mapping_Plugin(Plugin):
    '''base class for all gaze mapping routines'''
    uniqueness = 'by_base_class'
    def __init__(self,g_pool):
        super(Gaze_Mapping_Plugin, self).__init__(g_pool)
        self.g_pool.active_gaze_mapping_plugin = self

    def on_pupil_datum(self, p):
        raise NotImplementedError()

    # def start_map_loop(self):
    #     self._map_process = Thread(target=self._map_loop)
    #     self._map_process.start()

    # def stop_map_loop(self):
    #     n = {'subject':'stop_gaze_mapper'}
    #     self.g_pool.ipc_pub.notify(n)

    # def _map_loop(self):
    #     zmq_ctx = zmq_tools.zmq.Context()
    #     socket = zmq_tools.Msg_Receiver(zmq_ctx,self.g_pool.ipc_sub_url)
    #     socket.subscribe('notify.stop_gaze_mapper')
    #     socket.subscribe('pupil')
    #     out_socket = zmq_tools.Msg_Dispatcher(zmq_ctx,self.g_pool.ipc_pub_url)

    #     while True:
    #         topic, p = socket.recv()
    #         if topic == 'notify.stop_gaze_mapper':
    #             break
    #         gaze_data = self.on_pupil_datum(p)
    #         for g in gaze_data:
    #             out_socket.send('gaze.%s'%p['id'],g)

    # def cleanup(self):
    #     self.stop_map_loop()


class Monocular_Gaze_Mapper_Base(Gaze_Mapping_Plugin):
    """Base class to implement the map callback"""
    def __init__(self, g_pool):
        super(Monocular_Gaze_Mapper_Base, self).__init__(g_pool)
        self.min_pupil_confidence = 0.0

    def on_pupil_datum(self, p):
        if p['confidence'] > self.min_pupil_confidence:
            g = self._map_monocular(p)
            if g:
                return [g,]
            else:
                return []
        else:
            return []

class Binocular_Gaze_Mapper_Base(Gaze_Mapping_Plugin):
    """Base class to implement the map callback"""
    def __init__(self, g_pool):
        super(Binocular_Gaze_Mapper_Base, self).__init__(g_pool)

        self.min_pupil_confidence = 0.7
        self._caches = ([],[])
        self.temportal_cutoff = 0.3
        self.sample_cutoff = 10

    def on_pupil_datum(self, p):
        if p['confidence'] > self.min_pupil_confidence:
            self._caches[p['id']].append(p)

        if self._caches[0] and self._caches[1]:
            #we have binocular data

            if self._caches[0][0]['timestamp'] < self._caches[1][0]['timestamp']:
                p0 = self._caches[0].pop(0)
                p1 = self._caches[1][0]
                older_pt = p0
            else:
                p0 = self._caches[0][0]
                p1 = self._caches[1].pop(0)
                older_pt = p1

            if abs(p0['timestamp'] - p1['timestamp']) < self.temportal_cutoff:
                gaze_datum = self._map_binocular(p0,p1)
            else:
                gaze_datum = self._map_monocular(older_pt)

        elif len(self._caches[0])>self.sample_cutoff:
            p = self._caches[0].pop(0)
            gaze_datum = self._map_monocular(p)
        elif len(self._caches[1])>self.sample_cutoff:
            p = self._caches[1].pop(0)
            gaze_datum = self._map_monocular(p)
        else:
            gaze_datum = None

        if gaze_datum:
            return [gaze_datum,]
        else:
            return []



class Dummy_Gaze_Mapper(Monocular_Gaze_Mapper_Base):
    """docstring for Dummy_Gaze_Mapper"""
    def __init__(self, g_pool):
        super(Dummy_Gaze_Mapper, self).__init__(g_pool)

    def _map_monocular(self,p):
        return {'norm_pos':p['norm_pos'],'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]}

    def get_init_dict(self):
        return {}



class Simple_Gaze_Mapper(Monocular_Gaze_Mapper_Base):
    """docstring for Simple_Gaze_Mapper"""
    def __init__(self, g_pool,params):
        super(Simple_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.map_fn = make_map_function(*self.params)

    def _map_monocular(self,p):
        gaze_point = self.map_fn(p['norm_pos'])
        return {'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]}


    def get_init_dict(self):
        return {'params':self.params}


class Binocular_Gaze_Mapper(Binocular_Gaze_Mapper_Base):
    def __init__(self, g_pool,params,params_eye0,params_eye1):
        super(Binocular_Gaze_Mapper, self).__init__(g_pool)
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
        self.menu.append(ui.Switch('multivariate',self,label='Multivariate Mode'))


    def _map_binocular(self, p0, p1):
        if self.multivariate:
            gaze_point = self.map_fn(p0['norm_pos'], p1['norm_pos'])
        else:
            gaze_point_eye0 = self.map_fn_fallback[0](p0['norm_pos'])
            gaze_point_eye1 = self.map_fn_fallback[1](p1['norm_pos'])
            gaze_point = (gaze_point_eye0[0] + gaze_point_eye1[0])/2. , (gaze_point_eye0[1] + gaze_point_eye1[1])/2.
        confidence = (p0['confidence'] + p1['confidence'])/2.
        ts = (p0['timestamp'] + p1['timestamp'])/2.
        return {'norm_pos':gaze_point,'confidence':confidence,'timestamp':ts,'base':[p0, p1]}

    def _map_monocular(self,p):
        gaze_point = self.map_fn_fallback[p['id']](p['norm_pos'])
        return {'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]}

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def cleanup(self):
        super(Binocular_Gaze_Mapper, self).cleanup()
        self.deinit_gui()

    def get_init_dict(self):
        return {'params':self.params, 'params_eye0':self.params_eye0, 'params_eye1':self.params_eye1}



class Vector_Gaze_Mapper(Monocular_Gaze_Mapper_Base):
    """docstring for Vector_Gaze_Mapper"""
    def __init__(self, g_pool, eye_camera_to_world_matrix , camera_intrinsics ,cal_points_3d, cal_ref_points_3d, cal_gaze_points_3d, gaze_distance = 500 ):
        super(Vector_Gaze_Mapper, self).__init__(g_pool)
        self.eye_camera_to_world_matrix  =  eye_camera_to_world_matrix
        self.rotation_matrix = self.eye_camera_to_world_matrix[:3,:3]
        self.rotation_vector = cv2.Rodrigues( self.rotation_matrix  )[0]
        self.translation_vector  = self.eye_camera_to_world_matrix[:3,3]
        self.camera_matrix = camera_intrinsics['camera_matrix']
        self.dist_coefs = camera_intrinsics['dist_coefs']
        self.world_frame_size = camera_intrinsics['resolution']
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


    def toWorld(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matrix , point)[:3]

    def init_gui(self):

        def open_close_window(new_state):
            if new_state:
                self.visualizer.open_window()
            else:
                self.visualizer.close_window()

        self.menu = ui.Growing_Menu('Monocular 3D gaze mapper')
        self.g_pool.sidebar.insert(3,self.menu)
        self.menu.append(ui.Switch('debug window',setter=open_close_window, getter=lambda: bool(self.visualizer.window) ))
        self.menu.append(ui.Slider('gaze_distance',self,min=50,max=2000,label='gaze distance mm'))


    def _map_monocular(self,p):
        if '3d' not in p['method']:
            return None

        gaze_point =  np.array(p['circle_3d']['normal'] ) * self.gaze_distance  + np.array( p['sphere']['center'] )

        image_point, _  =  cv2.projectPoints( np.array([gaze_point]) , self.rotation_vector, self.translation_vector , self.camera_matrix , self.dist_coefs )
        image_point = image_point.reshape(-1,2)
        image_point = normalize( image_point[0], self.world_frame_size , flip_y = True)

        eye_center = self.toWorld(p['sphere']['center'])
        gaze_3d = self.toWorld(gaze_point)
        normal_3d = np.dot( self.rotation_matrix, np.array( p['circle_3d']['normal'] ) )

        g = {   'norm_pos':image_point,
                'eye_center_3d':eye_center.tolist(),
                'gaze_normal_3d':normal_3d.tolist(),
                'gaze_point_3d':gaze_3d.tolist(),
                'confidence':p['confidence'],
                'timestamp':p['timestamp'],
                'base':[p]}

        if self.visualizer.window:
            self.gaze_pts_debug.append( gaze_3d )
            self.sphere['center'] = eye_center #eye camera coordinates
            self.sphere['radius'] = p['sphere']['radius']
        return g


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
        super(Vector_Gaze_Mapper,self).cleanup()
        self.deinit_gui()
        self.visualizer.close_window()


class Binocular_Vector_Gaze_Mapper(Binocular_Gaze_Mapper_Base):
    """docstring for Vector_Gaze_Mapper"""
    def __init__(self, g_pool, eye_camera_to_world_matrix0, eye_camera_to_world_matrix1 , camera_intrinsics , cal_points_3d = [],cal_ref_points_3d = [], cal_gaze_points0_3d = [], cal_gaze_points1_3d = [] ):
        super(Binocular_Vector_Gaze_Mapper, self).__init__(g_pool)

        self.eye_camera_to_world_matricies  =  eye_camera_to_world_matrix0 , eye_camera_to_world_matrix1
        self.rotation_matricies  =  eye_camera_to_world_matrix0[:3,:3],eye_camera_to_world_matrix1[:3,:3]
        self.rotation_vectors = cv2.Rodrigues( eye_camera_to_world_matrix0[:3,:3]  )[0] , cv2.Rodrigues( eye_camera_to_world_matrix1[:3,:3]  )[0]
        self.translation_vectors  = eye_camera_to_world_matrix0[:3,3] , eye_camera_to_world_matrix1[:3,3]


        self.cal_points_3d = cal_points_3d
        self.cal_ref_points_3d = cal_ref_points_3d

        self.cal_gaze_points0_3d = cal_gaze_points0_3d #save for debug window
        self.cal_gaze_points1_3d = cal_gaze_points1_3d #save for debug window

        self.camera_matrix = camera_intrinsics['camera_matrix']
        self.dist_coefs = camera_intrinsics['dist_coefs']
        self.world_frame_size = camera_intrinsics['resolution']
        self.camera_intrinsics = camera_intrinsics
        self.visualizer = Calibration_Visualizer(g_pool,
                                                world_camera_intrinsics=camera_intrinsics ,
                                                cal_ref_points_3d = cal_points_3d,
                                                cal_observed_points_3d = cal_ref_points_3d,
                                                eye_camera_to_world_matrix0=  self.eye_camera_to_world_matricies[0],
                                                cal_gaze_points0_3d = cal_gaze_points0_3d,
                                                eye_camera_to_world_matrix1=  self.eye_camera_to_world_matricies[1],
                                                cal_gaze_points1_3d =  cal_gaze_points1_3d)
        self.g_pool = g_pool
        self.visualizer.open_window()
        self.gaze_pts_debug0 = []
        self.gaze_pts_debug1 = []
        self.intersection_points_debug = []
        self.sphere0 = {}
        self.sphere1 = {}
        self.last_gaze_distance = 500.


    def eye0_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matricies[0] , point)[:3]

    def eye1_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matricies[1] , point)[:3]


    def init_gui(self):

        def open_close_window(new_state):
            if new_state:
                self.visualizer.open_window()
            else:
                self.visualizer.close_window()

        self.menu = ui.Growing_Menu('Binocular 3D gaze mapper')
        self.g_pool.sidebar.insert(3,self.menu)
        # self.menu.append(ui.Slider('last_gaze_distance',self,min=50,max=2000,label='gaze distance mm'))
        self.menu.append(ui.Switch('debug window',setter=open_close_window, getter=lambda: bool(self.visualizer.window) ))

    def _map_monocular(self,p):
        if '3d' not in p['method']:
            return None

        p_id = p['id']
        gaze_point =  np.array(p['circle_3d']['normal'] ) * self.last_gaze_distance  + np.array( p['sphere']['center'] )
        image_point, _  =  cv2.projectPoints( np.array([gaze_point]) , self.rotation_vectors[p_id], self.translation_vectors[p_id] , self.camera_matrix , self.dist_coefs )
        image_point = image_point.reshape(-1,2)
        image_point = normalize( image_point[0], self.world_frame_size , flip_y = True)

        if p_id == 0:
            eye_center = self.eye0_to_World(p['sphere']['center'])
            gaze_3d = self.eye0_to_World(gaze_point)
        else:
            eye_center = self.eye1_to_World(p['sphere']['center'])
            gaze_3d = self.eye1_to_World(gaze_point)

        normal_3d = np.dot( self.rotation_matricies[p_id], np.array( p['circle_3d']['normal'] ) )

        g = {   'norm_pos':image_point,
                'eye_centers_3d':{p['id']:eye_center.tolist()},
                'gaze_normals_3d':{p['id']:normal_3d.tolist()},
                'gaze_point_3d':gaze_3d.tolist(),
                'confidence':p['confidence'],
                'timestamp':p['timestamp'],
                'base':[p]}


        if self.visualizer.window:
            if p_id == 0:
                self.gaze_pts_debug0.append(gaze_3d)
                self.sphere0['center'] = eye_center
                self.sphere0['radius'] = p['sphere']['radius']
            else:
                self.gaze_pts_debug1.append(gaze_3d)
                self.sphere1['center'] = eye_center
                self.sphere1['radius'] = p['sphere']['radius']

        return g

    def _map_binocular(self, p0, p1):

        if '3d'not in p0['method'] or '3d' not in p1['method']:
            return None

        #find the nearest intersection point of the two gaze lines
        # a line is defined by two point
        s0_center = self.eye0_to_World( np.array( p0['sphere']['center'] ) )
        s1_center = self.eye1_to_World( np.array( p1['sphere']['center'] ) )

        s0_normal = np.dot( self.rotation_matricies[0], np.array( p0['circle_3d']['normal'] ) )
        s1_normal = np.dot( self.rotation_matricies[1], np.array( p1['circle_3d']['normal'] ) )

        gaze_line0 = [ s0_center, s0_center + s0_normal ]
        gaze_line1 = [ s1_center, s1_center + s1_normal ]

        nearest_intersection_point , intersection_distance = math_helper.nearest_intersection( gaze_line0, gaze_line1 )

        if nearest_intersection_point is not None :
            self.last_gaze_distance = np.sqrt( nearest_intersection_point.dot( nearest_intersection_point ) )
            image_point, _  =  cv2.projectPoints( np.array([nearest_intersection_point]) ,  np.array([0.0,0.0,0.0]) ,  np.array([0.0,0.0,0.0]) , self.camera_matrix , self.dist_coefs )
            image_point = image_point.reshape(-1,2)
            image_point = normalize( image_point[0], self.world_frame_size , flip_y = True)


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

        if nearest_intersection_point is None :
            return None


        confidence = (p0['confidence'] + p1['confidence'])/2.
        ts = (p0['timestamp'] + p1['timestamp'])/2.
        g = {   'norm_pos':image_point,
                'eye_centers_3d':{0:s0_center.tolist(),1:s1_center.tolist()},
                'gaze_normals_3d':{0:s0_normal.tolist(),1:s1_normal.tolist()},
                'gaze_point_3d':nearest_intersection_point.tolist(),
                'confidence':confidence,
                'timestamp':ts,
                'base':[p0,p1]}
        return g

    def gl_display(self):
        self.visualizer.update_window( self.g_pool , self.gaze_pts_debug0 , self.sphere0, self.gaze_pts_debug1, self.sphere1, self.intersection_points_debug )
        self.gaze_pts_debug0 = []
        self.gaze_pts_debug1 = []
        self.intersection_points_debug = []

    def get_init_dict(self):
       return {'eye_camera_to_world_matrix0':self.eye_camera_to_world_matricies[0] ,'eye_camera_to_world_matrix1':self.eye_camera_to_world_matricies[1] ,'cal_ref_points_3d':self.cal_ref_points_3d, 'cal_gaze_points0_3d':self.cal_gaze_points0_3d, 'cal_gaze_points1_3d':self.cal_gaze_points1_3d,  "camera_intrinsics":self.camera_intrinsics}


    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None


    def cleanup(self):
        super(Binocular_Vector_Gaze_Mapper, self).cleanup()
        self.deinit_gui()
        self.visualizer.close_window()
