'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from visualizer import Visualizer
from OpenGL.GL import *
from pyglui.cygl.utils import RGBA
from pyglui.cygl import utils as glutils
from gl_utils.trackball import Trackball
import numpy as np
import math


class Calibration_Visualizer(Visualizer):
	def __init__(self, g_pool, world_camera_intrinsics , cal_ref_points_3d, cal_observed_points_3d, eye_camera_to_world_matrix0 , cal_gaze_points0_3d, eye_camera_to_world_matrix1 = np.eye(4) , cal_gaze_points1_3d = [],  run_independently = False , name = "Calibration Visualizer" ):
		super().__init__( g_pool,name,  run_independently)

		self.image_width = 640 # right values are assigned in update
		self.focal_length = 620
		self.image_height = 480

		self.eye_camera_to_world_matrix0 = np.asarray(eye_camera_to_world_matrix0)
		self.eye_camera_to_world_matrix1 = np.asarray(eye_camera_to_world_matrix1)

		self.cal_ref_points_3d = cal_ref_points_3d
		self.cal_observed_points_3d = cal_observed_points_3d
		self.cal_gaze_points0_3d = cal_gaze_points0_3d
		self.cal_gaze_points1_3d = cal_gaze_points1_3d

		if world_camera_intrinsics:
			self.world_camera_width = world_camera_intrinsics['resolution'][0]
			self.world_camera_height = world_camera_intrinsics['resolution'][1]
			self.world_camera_focal = (world_camera_intrinsics['camera_matrix'][0][0] +  world_camera_intrinsics['camera_matrix'][1][1] ) / 2.0
		else:
			self.world_camera_width = 0
			self.world_camera_height = 0
			self.world_camera_focal = 0

		camera_fov = math.degrees(2.0 * math.atan( self.window_size[0] / (2.0 * self.focal_length)))
		self.trackball = Trackball(camera_fov)
		self.trackball.distance = [0,0,-80.]
		self.trackball.pitch = 210
		self.trackball.roll = 0


	########### Open, update, close #####################

	def update_window(self, g_pool , gaze_points0 , sphere0 , gaze_points1 = [] , sphere1 = None, intersection_points = []  ):

		if not self.window:
			return

		self.begin_update_window() #sets context

		self.clear_gl_screen()
		self.trackball.push()

		glMatrixMode( GL_MODELVIEW )

		# draw things in world camera coordinate system
		glPushMatrix()
		glLoadIdentity()

		calibration_points_line_color = RGBA(0.5,0.5,0.5,0.1);
		error_line_color = RGBA(1.0,0.0,0.0,0.5)

		self.draw_coordinate_system(200)
		if self.world_camera_width != 0:
			self.draw_frustum( self.world_camera_width/ 10.0 , self.world_camera_height/ 10.0 , self.world_camera_focal / 10.0)

		for p in self.cal_observed_points_3d:
			glutils.draw_polyline( [ (0,0,0), p]  , 1 , calibration_points_line_color, line_type = GL_LINES)
			#draw error lines form eye gaze points to  ref points
		for(cal_point,ref_point) in zip(self.cal_ref_points_3d, self.cal_observed_points_3d):
				glutils.draw_polyline( [ cal_point, ref_point]  , 1 , error_line_color, line_type = GL_LINES)

		#calibration points
		glutils.draw_points( self.cal_ref_points_3d , 4 , RGBA( 0, 1, 1, 1 ) )


		glPopMatrix()

		if sphere0:
			# eye camera
			glPushMatrix()
			glLoadMatrixf( self.eye_camera_to_world_matrix0.T )

			self.draw_coordinate_system(60)
			self.draw_frustum( self.image_width / 10.0, self.image_height / 10.0, self.focal_length /10.)
			glPopMatrix()

			#everything else is in world coordinates

			#eye
			sphere_center0 = list(sphere0['center'])
			sphere_radius0 = sphere0['radius']
			self.draw_sphere(sphere_center0,sphere_radius0,  color = RGBA(1,1,0,1))

			#gazelines
			for p in self.cal_gaze_points0_3d:
				glutils.draw_polyline( [ sphere_center0, p]  , 1 , calibration_points_line_color, line_type = GL_LINES)

			#calibration points
			# glutils.draw_points( self.cal_gaze_points0_3d , 4 , RGBA( 1, 0, 1, 1 ) )

			#current gaze points
			glutils.draw_points( gaze_points0 , 2 , RGBA( 1, 0, 0, 1 ) )
			for p in gaze_points0:
				glutils.draw_polyline( [sphere_center0, p]  , 1 , RGBA(0,0,0,1), line_type = GL_LINES)

			#draw error lines form eye gaze points to  ref points
			for(cal_gaze_point,ref_point) in zip(self.cal_gaze_points0_3d, self.cal_ref_points_3d):
				glutils.draw_polyline( [ cal_gaze_point, ref_point]  , 1 , error_line_color, line_type = GL_LINES)

		#second eye
		if sphere1:
			# eye camera
			glPushMatrix()
			glLoadMatrixf( self.eye_camera_to_world_matrix1.T )

			self.draw_coordinate_system(60)
			self.draw_frustum( self.image_width / 10.0, self.image_height / 10.0, self.focal_length /10.)
			glPopMatrix()

			#everything else is in world coordinates

			#eye
			sphere_center1 = list(sphere1['center'])
			sphere_radius1 = sphere1['radius']
			self.draw_sphere(sphere_center1,sphere_radius1,  color = RGBA(1,1,0,1))

			#gazelines
			for p in self.cal_gaze_points1_3d:
				glutils.draw_polyline( [ sphere_center1, p]  , 4 , calibration_points_line_color, line_type = GL_LINES)

			#calibration points
			glutils.draw_points( self.cal_gaze_points1_3d , 4 , RGBA( 1, 0, 1, 1 ) )

			#current gaze points
			glutils.draw_points( gaze_points1 , 2 , RGBA( 1, 0, 0, 1 ) )
			for p in gaze_points1:
				glutils.draw_polyline( [sphere_center1, p]  , 1 , RGBA(0,0,0,1), line_type = GL_LINES)

			#draw error lines form eye gaze points to  ref points
			for(cal_gaze_point,ref_point) in zip(self.cal_gaze_points1_3d, self.cal_ref_points_3d):
				glutils.draw_polyline( [ cal_gaze_point, ref_point]  , 1 , error_line_color, line_type = GL_LINES)

		self.trackball.pop()

		self.end_update_window() #swap buffers, handle context



	############ window callbacks #################
	def on_resize(self,window,w, h):
		Visualizer.on_resize(self,window,w, h)
		self.trackball.set_window_size(w,h)


	def on_char(self,window,char):
		if char == ord('r'):
			self.trackball.distance = [0,0,-0.1]
			self.trackball.pitch = 0
			self.trackball.roll = 180

	def on_scroll(self,window,x,y):
		self.trackball.zoom_to(y)
