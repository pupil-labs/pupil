
import logging
from glfw import *
from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D

# create logger for the context of this function
logger = logging.getLogger(__name__)
from pyglui import ui

from pyglui.cygl.utils import init
from pyglui.cygl.utils import RGBA
from pyglui.cygl.utils import *
from pyglui.cygl import utils as glutils
from gl_utils.trackball import Trackball
from pyglui.pyfontstash import fontstash as fs
from pyglui.ui import get_opensans_font_path
import numpy as np
import math

circle_xy = [] #this is a global variable
circle_res = 30.0
for i in range(0,int(circle_res+1)):
	temp =  (i)/circle_res *  math.pi * 2.0
	circle_xy.append([np.cos(temp),np.sin(temp)])

class Calibration_Visualizer(object):
	def __init__(self, g_pool, world_camera_intrinsics , cal_ref_points_3d, eye_camera_to_world_matrix0 , cal_gaze_points0_3d, eye_camera_to_world_matrix1 = np.eye(4) , cal_gaze_points1_3d = [],   name = "Debug Calibration Visualizer", run_independently = False):
       # super(Visualizer, self).__init__()

		self.g_pool = g_pool
		self.image_width = 640 # right values are assigned in update
		self.focal_length = 620
		self.image_height = 480

		self.eye_camera_to_world_matrix0 = eye_camera_to_world_matrix0
		self.eye_camera_to_world_matrix1 = eye_camera_to_world_matrix1

		self.cal_ref_points_3d = cal_ref_points_3d
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


		self.name = name
		self.window_size = (640,480)
		self.window = None
		self.input = None
		self.run_independently = run_independently

		camera_fov = math.degrees(2.0 * math.atan( self.window_size[0] / (2.0 * self.focal_length)))
		self.trackball = Trackball(camera_fov)
		self.trackball.distance = [0,0,-0.1]
		self.trackball.pitch = 0
		self.trackball.roll = 180

	############## DRAWING FUNCTIONS ##############################

	def draw_frustum(self, width, height , length):

		W = width/2.0
		H = height/2.0
		Z = length
		# draw it
		glLineWidth(1)
		glColor4f( 1, 0.5, 0, 0.5 )
		glBegin( GL_LINE_LOOP )
		glVertex3f( 0, 0, 0 )
		glVertex3f( -W, H, Z )
		glVertex3f( W, H, Z )
		glVertex3f( 0, 0, 0 )
		glVertex3f( W, H, Z )
		glVertex3f( W, -H, Z )
		glVertex3f( 0, 0, 0 )
		glVertex3f( W, -H, Z )
		glVertex3f( -W, -H, Z )
		glVertex3f( 0, 0, 0 )
		glVertex3f( -W, -H, Z )
		glVertex3f( -W, H, Z )
		glEnd( )

	def draw_coordinate_system(self,l=1):
		# Draw x-axis line. RED
		glLineWidth(2)
		glColor3f( 1, 0, 0 )
		glBegin( GL_LINES )
		glVertex3f( 0, 0, 0 )
		glVertex3f( l, 0, 0 )
		glEnd( )

		# Draw y-axis line. GREEN.
		glColor3f( 0, 1, 0 )
		glBegin( GL_LINES )
		glVertex3f( 0, 0, 0 )
		glVertex3f( 0, l, 0 )
		glEnd( )

		# Draw z-axis line. BLUE
		glColor3f( 0, 0, 1 )
		glBegin( GL_LINES )
		glVertex3f( 0, 0, 0 )
		glVertex3f( 0, 0, l )
		glEnd( )

	def draw_sphere(self,sphere_position, sphere_radius,contours = 45, color =RGBA(.2,.5,0.5,.5) ):
		# this function draws the location of the eye sphere
		glPushMatrix()

		glTranslatef(sphere_position[0],sphere_position[1],sphere_position[2]) #sphere[0] contains center coordinates (x,y,z)
		glTranslatef(0,0,sphere_radius) #sphere[1] contains radius
		for i in xrange(1,contours+1):

			glTranslatef(0,0, -sphere_radius/contours*2)
			position = sphere_radius - i*sphere_radius*2/contours
			draw_radius = np.sqrt(sphere_radius**2 - position**2)
			glPushMatrix()
			glScalef(draw_radius,draw_radius,1)
			draw_polyline((circle_xy),2,color)
			glPopMatrix()

		glPopMatrix()



	def basic_gl_setup(self):
		glEnable(GL_POINT_SPRITE )
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE) # overwrite pointsize
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable(GL_BLEND)
		glClearColor(.8,.8,.8,1.)
		glEnable(GL_LINE_SMOOTH)
		# glEnable(GL_POINT_SMOOTH)
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
		glEnable(GL_LINE_SMOOTH)
		glEnable(GL_POLYGON_SMOOTH)
		glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

	def adjust_gl_view(self,w,h):
		"""
		adjust view onto our scene.
		"""
		glViewport(0, 0, w, h)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0, w, h, 0, -1, 1)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

	def clear_gl_screen(self):
		glClearColor(.9,.9,0.9,1.)
		glClear(GL_COLOR_BUFFER_BIT)

	########### Open, update, close #####################


	def open_window(self):
		if not self.window:
			self.input = {'button':None, 'mouse':(0,0)}

			# get glfw started
			if self.run_independently:
				glfwInit()
				self.window = glfwCreateWindow(self.window_size[0], self.window_size[1], self.name, None  )
			else:
				self.window = glfwCreateWindow(self.window_size[0], self.window_size[1], self.name, None, share=self.g_pool.main_window )

			active_window = glfwGetCurrentContext();
			glfwMakeContextCurrent(self.window)

			glfwSetWindowPos(self.window,0,0)
			# Register callbacks window
			glfwSetFramebufferSizeCallback(self.window,self.on_resize)
			glfwSetWindowIconifyCallback(self.window,self.on_iconify)
			glfwSetKeyCallback(self.window,self.on_key)
			glfwSetCharCallback(self.window,self.on_char)
			glfwSetMouseButtonCallback(self.window,self.on_button)
			glfwSetCursorPosCallback(self.window,self.on_pos)
			glfwSetScrollCallback(self.window,self.on_scroll)

			# get glfw started
			if self.run_independently:
				init()
			self.basic_gl_setup()

			self.glfont = fs.Context()
			self.glfont.add_font('opensans',get_opensans_font_path())
			self.glfont.set_size(22)
			self.glfont.set_color_float((0.2,0.5,0.9,1.0))
			self.on_resize(self.window,*glfwGetFramebufferSize(self.window))
			glfwMakeContextCurrent(active_window)

			# self.gui = ui.UI()

	def update_window(self, g_pool , gaze_points0 , sphere0 , gaze_points1 = [] , sphere1 = None, intersection_points = []  ):
		if self.window:
			if glfwWindowShouldClose(self.window):
				self.close_window()
				return

			active_window = glfwGetCurrentContext()
			glfwMakeContextCurrent(self.window)

			self.clear_gl_screen()
			self.trackball.push()

			# use opencv coordinate system
			#glMatrixMode( GL_PROJECTION )
			#glScalef( 1. ,-1. , -1. )
			glMatrixMode( GL_MODELVIEW )

			# draw things in world camera coordinate system
			glPushMatrix()
			glLoadIdentity()

			calibration_points_line_color = RGBA(0.5,0.5,0.5,0.1);
			error_line_color = RGBA(1.0,0.0,0.0,0.5)

			self.draw_coordinate_system(200)
			if self.world_camera_width != 0:
				self.draw_frustum( self.world_camera_width/ 10.0 , self.world_camera_height/ 10.0 , self.world_camera_focal / 10.0)

			for p in self.cal_ref_points_3d:
				draw_polyline( [ (0,0,0), p]  , 1 , calibration_points_line_color, line_type = GL_LINES)
			#calibration points
			draw_points( self.cal_ref_points_3d , 4 , RGBA( 0, 1, 1, 1 ) )


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
					draw_polyline( [ sphere_center0, p]  , 1 , calibration_points_line_color, line_type = GL_LINES)

				#calibration points
				draw_points( self.cal_gaze_points0_3d , 4 , RGBA( 1, 0, 1, 1 ) )

				#current gaze points
				draw_points( gaze_points0 , 2 , RGBA( 1, 0, 0, 1 ) )
				for p in gaze_points0:
					draw_polyline( [sphere_center0, p]  , 1 , RGBA(0,0,0,1), line_type = GL_LINES)

				#draw error lines form eye gaze points to  ref points
				distance = 0.0
				for(cal_gaze_point,ref_point) in zip(self.cal_gaze_points0_3d, self.cal_ref_points_3d):
					draw_polyline( [ cal_gaze_point, ref_point]  , 1 , error_line_color, line_type = GL_LINES)

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
					draw_polyline( [ sphere_center1, p]  , 1 , calibration_points_line_color, line_type = GL_LINES)

				#calibration points
				draw_points( self.cal_gaze_points1_3d , 4 , RGBA( 1, 0, 1, 1 ) )

				#current gaze points
				draw_points( gaze_points1 , 2 , RGBA( 1, 0, 0, 1 ) )
				for p in gaze_points1:
					draw_polyline( [sphere_center1, p]  , 1 , RGBA(0,0,0,1), line_type = GL_LINES)

				#draw error lines form eye gaze points to  ref points
				distance = 0.0
				for(cal_gaze_point,ref_point) in zip(self.cal_gaze_points1_3d, self.cal_ref_points_3d):
					draw_polyline( [ cal_gaze_point, ref_point]  , 1 , error_line_color, line_type = GL_LINES)

			self.trackball.pop()


			glfwSwapBuffers(self.window)
			glfwPollEvents()
			glfwMakeContextCurrent(active_window)

	def close_window(self):
		if self.window:
			glfwDestroyWindow(self.window)
			self.window = None

	############ window callbacks #################
	def on_resize(self,window,w, h):
		h = max(h,1)
		w = max(w,1)
		self.trackball.set_window_size(w,h)

		self.window_size = (w,h)
		active_window = glfwGetCurrentContext()
		glfwMakeContextCurrent(window)
		self.adjust_gl_view(w,h)
		glfwMakeContextCurrent(active_window)

	def on_char(self,window,char):
		if char == ord('r'):
			self.trackball.distance = [0,0,-0.1]
			self.trackball.pitch = 0
			self.trackball.roll = 180

	def on_button(self,window,button, action, mods):
		# self.gui.update_button(button,action,mods)
		if action == GLFW_PRESS:
			self.input['button'] = button
			self.input['mouse'] = glfwGetCursorPos(window)
		if action == GLFW_RELEASE:
			self.input['button'] = None

	def on_pos(self,window,x, y):
		hdpi_factor = float(glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0])
		x,y = x*hdpi_factor,y*hdpi_factor
		# self.gui.update_mouse(x,y)
		if self.input['button']==GLFW_MOUSE_BUTTON_RIGHT:
			old_x,old_y = self.input['mouse']
			self.trackball.drag_to(x-old_x,y-old_y)
			self.input['mouse'] = x,y
		if self.input['button']==GLFW_MOUSE_BUTTON_LEFT:
			old_x,old_y = self.input['mouse']
			self.trackball.pan_to(x-old_x,y-old_y)
			self.input['mouse'] = x,y

	def on_scroll(self,window,x,y):
		self.trackball.zoom_to(y)

	def on_iconify(self,window,iconified): pass

	def on_key(self,window, key, scancode, action, mods): pass
