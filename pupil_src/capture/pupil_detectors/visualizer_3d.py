
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
import cv2
import random



def convert_fov(fov,width):
	fov = fov* math.pi/180
	focal_length = (width/2)/np.tan(fov/2)
	return focal_length

def get_perpendicular_vector(v):
    """ Finds an arbitrary perpendicular vector to *v*."""
    # http://codereview.stackexchange.com/questions/43928/algorithm-to-get-an-arbitrary-perpendicular-vector
    # for two vectors (x, y, z) and (a, b, c) to be perpendicular,
    # the following equation has to be fulfilled
    #     0 = ax + by + cz

    # x = y = z = 0 is not an acceptable solution
    if v[0] == v[1] == v[2] == 0:
        logger.error('zero-vector')

    # If one dimension is zero, this can be solved by setting that to
    # non-zero and the others to zero. Example: (4, 2, 0) lies in the
    # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
    if v[0] == 0:
        return np.array((1, 0, 0))
    if v[1] == 0:
        return np.array((0, 1, 0))
    if v[2] == 0:
        return np.array((0, 0, 1))

    # arbitrarily set a = b = 1
    # then the equation simplifies to
    #     c = -(x + y)/z
    return np.array([1, 1, -1.0 * (v[0] + v[1]) / v[2]])

circle_xy = [] #this is a global variable
circle_res = 30.0
for i in range(0,int(circle_res+1)):
	temp =  (i)/circle_res *  math.pi * 2.0
	circle_xy.append([np.cos(temp),np.sin(temp)])

class Visualizer(object):
	def __init__(self,focal_length, name = "Debug Visualizer", run_independently = False):
       # super(Visualizer, self).__init__()
		self.focal_length = focal_length
		self.image_width = 640 # right values are assigned in update
		self.image_height = 480
		# transformation matrices
		self.anthromorphic_matrix = self.get_anthropomorphic_matrix()
		self.adjusted_pixel_space_matrix = self.get_adjusted_pixel_space_matrix(1)

		self.name = name
		self.window_size = (640,480)
		self._window = None
		self.input = None
		self.run_independently = run_independently

		camera_fov = math.degrees(2.0 * math.atan( self.window_size[0] / (2.0 * self.focal_length)))
		self.trackball = Trackball(camera_fov)

	############## MATRIX FUNCTIONS ##############################

	def get_anthropomorphic_matrix(self):
		temp =  np.identity(4)
		temp[2,2] *= -1
		return temp

	def get_adjusted_pixel_space_matrix(self):
		temp =  np.identity(4)
		temp[2,2] *= -1
		return temp

	def get_adjusted_pixel_space_matrix(self,scale):
		# returns a homoegenous matrix
		temp = self.get_anthropomorphic_matrix()
		temp[3,3] *= scale
		return temp

	def get_image_space_matrix(self,scale=1.):
		temp = self.get_adjusted_pixel_space_matrix(scale)
		temp[1,1] *=-1 #image origin is top left
		temp[0,3] = -self.image_width/2.0
		temp[1,3] = self.image_height/2.0
		temp[2,3] = -self.focal_length
		return temp.T

	def get_pupil_transformation_matrix(self,circle_normal,circle_center, circle_scale = 1.0):
		"""
			OpenGL matrix convention for typical GL software
			with positive Y=up and positive Z=rearward direction
			RT = right
			UP = up
			BK = back
			POS = position/translation
			US = uniform scale

			float transform[16];

			[0] [4] [8 ] [12]
			[1] [5] [9 ] [13]
			[2] [6] [10] [14]
			[3] [7] [11] [15]

			[RT.x] [UP.x] [BK.x] [POS.x]
			[RT.y] [UP.y] [BK.y] [POS.y]
			[RT.z] [UP.z] [BK.z] [POS.Z]
			[    ] [    ] [    ] [US   ]
		"""
		temp = self.get_anthropomorphic_matrix()
		right = temp[:3,0]
		up = temp[:3,1]
		back = temp[:3,2]
		translation = temp[:3,3]
		back[:] = np.array(circle_normal)
		back[2] *=-1 #our z axis is inverted

		if np.linalg.norm(back) != 0:
			back[:] /= np.linalg.norm(back)
			right[:] = get_perpendicular_vector(back)/np.linalg.norm(get_perpendicular_vector(back))
			up[:] = np.cross(right,back)/np.linalg.norm(np.cross(right,back))
			right[:] *= circle_scale
			back[:] *=circle_scale
			up[:] *=circle_scale
			translation[:] = np.array(circle_center)
			translation[2] *= -1
		return   temp.T

	############## DRAWING FUNCTIONS ##############################

	def draw_frustum(self ):

		W = self.image_width/2.0
		H = self.image_height/2.0
		Z = self.focal_length
		glPushMatrix()
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
		glPopMatrix()

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


	def draw_circle(self, circle_center, circle_normal, circle_radius, color=RGBA(1.1,0.2,.8), num_segments = 20):
		vertices = []
		vertices.append( (0,0,0) )  # circle center

		#create circle vertices in the xy plane
		for i in np.linspace(0.0, 2.0*math.pi , num_segments ):
			x = math.sin(i)
			y = math.cos(i)
			z = 0
			vertices.append((x,y,z))

		glPushMatrix()
		glMatrixMode(GL_MODELVIEW )
		glLoadMatrixf(self.get_pupil_transformation_matrix(circle_normal,circle_center, circle_radius))
		draw_polyline((vertices),color=color, line_type = GL_TRIANGLE_FAN) # circle
		draw_polyline( [ (0,0,0), (0,0, 4) ] ,color=RGBA(0,0,0), line_type = GL_LINES) #normal
		glPopMatrix()



	def draw_contours_on_screen(self,contours, color = RGBA(0.,0.,0.,0.5)):
		#this function displays the contours on the 2D video stream within the visualizer module
		glPushMatrix()
		glLoadMatrixf(self.get_image_space_matrix(30))
		for contour in contours:
			draw_polyline(contour,color)
		glPopMatrix()

	def draw_contours(self, contours, thickness = 1, color = RGBA(0.,0.,0.,0.5) ):
		glPushMatrix()
		glLoadMatrixf(self.get_anthropomorphic_matrix())
		for contour in contours:
			draw_polyline(contour, thickness, color = color )
		glPopMatrix()

	def draw_contour(self, contour, thickness = 1, color = RGBA(0.,0.,0.,0.5) ):
		glPushMatrix()
		glLoadMatrixf(self.get_anthropomorphic_matrix())
		draw_polyline(contour,thickness, color)
		glPopMatrix()


	def draw_debug_info(self, result  ):
		models = result['models']
		eye = models[0]['sphere'];
		direction = result['circle'][1];
		pupil_radius = result['circle'][2];

		status = ' Eyeball center : X: %.2fmm Y: %.2fmm Z: %.2fmm\n Pupil direction:  X: %.2f Y: %.2f Z: %.2f\n Pupil Diameter: %.2fmm\n  ' \
		%(eye[0][0], eye[0][1],eye[0][2],
		direction[0], direction[1],direction[2], pupil_radius*2)

		self.glfont.push_state()
		self.glfont.set_color_float( (0,0,0,1) )

		self.glfont.draw_multi_line_text(5,20,status)


		#draw model info for each model
		delta_y = 20
		for model in models:
			modelStatus =	('Model: %d \n ' %  model['modelID'] ,
							'    maturity: %.3f\n' % model['maturity'] ,
							'    fit: %.6f\n' % model['fit'] ,
							'    performance: %.6f\n' % model['performance'] ,
							'    perf.Grad.: %.3e\n' % model['performanceGradient'] ,
							)
			modeltext = ''.join( modelStatus )
			self.glfont.draw_multi_line_text(self.window_size[0] - 200 ,delta_y, modeltext)

			delta_y += 100

		self.glfont.pop_state()


	########## Setup functions I don't really understand ############

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
		if not self._window:
			self.input = {'button':None, 'mouse':(0,0)}

			# get glfw started
			if self.run_independently:
				glfwInit()
			window = glfwGetCurrentContext()
			self._window = glfwCreateWindow(self.window_size[0], self.window_size[1], self.name, None, window)
			glfwMakeContextCurrent(self._window)

			if not self._window:
				exit()

			glfwSetWindowPos(self._window,0,0)
			# Register callbacks window
			glfwSetFramebufferSizeCallback(self._window,self.on_resize)
			glfwSetWindowIconifyCallback(self._window,self.on_iconify)
			glfwSetKeyCallback(self._window,self.on_key)
			glfwSetCharCallback(self._window,self.on_char)
			glfwSetMouseButtonCallback(self._window,self.on_button)
			glfwSetCursorPosCallback(self._window,self.on_pos)
			glfwSetScrollCallback(self._window,self.on_scroll)

			# get glfw started
			if self.run_independently:
				init()
			self.basic_gl_setup()

			self.glfont = fs.Context()
			self.glfont.add_font('opensans',get_opensans_font_path())
			self.glfont.set_size(22)
			self.glfont.set_color_float((0.2,0.5,0.9,1.0))
			self.on_resize(self._window,*glfwGetFramebufferSize(self._window))
			glfwMakeContextCurrent(window)

			# self.gui = ui.UI()

	def update_window(self, g_pool, result  ):

		if not result:
			return

		if glfwWindowShouldClose(self._window):
			self.close_window()
			return

		if self._window != None:
			glfwMakeContextCurrent(self._window)
		else:
			return

		self.image_width , self.image_height = g_pool.capture.frame_size

		latest_circle = result['circle']
		predicted_circle = result['predictedCircle']
		edges =  result['edges']
		sphere_models = result['models']

		self.clear_gl_screen()
		self.trackball.push()

		# 2. in pixel space draw video frame
		glLoadMatrixf(self.get_image_space_matrix())
		g_pool.image_tex.draw( quad=((0,self.image_height),(self.image_width,self.image_height),(self.image_width,0),(0,0)) ,alpha=0.5)

		glLoadMatrixf(self.get_anthropomorphic_matrix())

		self.draw_frustum()

		model_count = 0;
		sphere_color = RGBA( 0,147/255.,147/255.,0.2)
		initial_sphere_color = RGBA( 0,147/255.,147/255.,0.2)

		alternative_sphere_color = RGBA( 1,0.5,0.5,0.05)
		alternative_initial_sphere_color = RGBA( 1,0.5,0.5,0.05)

		for model in sphere_models:
			bin_positions = model['binPositions']
			sphere = model['sphere']
			initial_sphere = model['initialSphere']

			if model_count == 0:
				# self.draw_sphere(initial_sphere[0],initial_sphere[1], color = sphere_color )
				self.draw_sphere(sphere[0],sphere[1],  color = initial_sphere_color )
				draw_points(bin_positions, 3 , RGBA(0.6,0.0,0.6,0.5) )

			else:
				#self.draw_sphere(initial_sphere[0],initial_sphere[1], color = alternative_sphere_color )
				self.draw_sphere(sphere[0],sphere[1],  color = alternative_initial_sphere_color )

			model_count += 1


		self.draw_circle( latest_circle[0], latest_circle[1], latest_circle[2], RGBA(0.0,1.0,1.0,0.4))
		self.draw_circle( predicted_circle[0], predicted_circle[1], predicted_circle[2], RGBA(1.0,0.0,0.0,0.4))

		draw_points(edges, 2 , RGBA(1.0,0.0,0.6,0.5) )

		glLoadMatrixf(self.get_anthropomorphic_matrix())
		self.draw_coordinate_system(4)

		self.trackball.pop()

		self.draw_debug_info(result)

		glfwSwapBuffers(self._window)
		glfwPollEvents()
		return True

	def close_window(self):
		if self._window:
			active_window = glfwGetCurrentContext();
			glfwDestroyWindow(self._window)
			self._window = None
			glfwMakeContextCurrent( active_window)

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
			self.trackball.roll = 0

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
