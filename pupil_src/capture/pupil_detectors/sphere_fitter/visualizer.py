"""
	Andrew Xia working on visualizing data.
	I want to use opengl to display the 3d sphere and lines that connect to it.
	This file is in pupil-labs-andrew/sphere_fitter, so it is the prototype version
	July 6 2015

"""
import logging
from glfw import *
from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D
from OpenGL.GLUT import glutWireSphere, glutInit

# create logger for the context of this function
logger = logging.getLogger(__name__)
from pyglui import ui

from pyglui.cygl.utils import init
from pyglui.cygl.utils import RGBA
from pyglui.cygl.utils import *
from pyglui.cygl import utils as glutils
from trackball import Trackball
from pyglui.pyfontstash import fontstash as fs
from pyglui.ui import get_opensans_font_path

import numpy as np
import scipy
import geometry 
from __init__ import Sphere_Fitter
import cv2

def convert_fov(fov,width):
	fov = fov*scipy.pi/180
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

rad = [] #this is a global variable
for i in xrange(45 + 1): #so go to 45
	temp = i*16*scipy.pi/360.
	rad.append([np.cos(temp),np.sin(temp)])

class Visualizer():
	def __init__(self,name = "unnamed", focal_length = 554.25625, intrinsics = None, run_independently = False):
		# self.video_frame = (np.linspace(0,1,num=(400*400*4))*255).astype(np.uint8).reshape((400,400,4)) #the randomized image, should be video frame
		# self.screen_points = [] #collection of points

		if intrinsics == None:
			intrinsics = np.identity(3)
			if focal_length != None:
				intrinsics[0,0] = focal_length
				intrinsics[1,1] = focal_length
				logger.warning('no camera intrinsic input, set to focal length')
			else:
				logger.warning('no camera intrinsic input, set to default identity matrix')
		# transformation matrices
		self.intrinsics = intrinsics #camera intrinsics of our webcam.
		self.anthromorphic_matrix = self.get_anthropomorphic_matrix()
		self.adjusted_pixel_space_matrix = self.get_adjusted_pixel_space_matrix(1)


		self.name = name
		self._window = None
		self.input = None
		self.trackball = None
		self.run_independently = run_independently

		self.window_should_close = False

	############## MATRIX FUNCTIONS ##############################

	def get_anthropomorphic_matrix(self):
		temp =  np.identity(4)
		temp[2,2] *=-1 #consistent with our 3d coord system
		return temp

	def get_adjusted_pixel_space_matrix(self,scale):
		# returns a homoegenous matrix
		temp = self.get_anthropomorphic_matrix()
		temp[3,3] *= scale
		return temp

	def get_image_space_matrix(self,scale=1.):
		temp = self.get_adjusted_pixel_space_matrix(scale)
		temp[1,1] *=-1 #image origin is top left
		temp[0,3] = -self.intrinsics[0,2] #cx
		temp[1,3] = self.intrinsics[1,2] #cy
		temp[2,3] = -self.intrinsics[0,0] #focal length
		return temp.T

	def get_pupil_transformation_matrix(self,circle):
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
		back[:] = np.array(circle.normal)
		back[-2] *=-1 #our z axis is inverted
		back[-0] *=-1 #our z axis is inverted
		back[:] /= np.linalg.norm(back)
		right[:] = get_perpendicular_vector(back)/np.linalg.norm(get_perpendicular_vector(back))
		up[:] = np.cross(right,back)/np.linalg.norm(np.cross(right,back))
		translation[:] = np.array((circle.center[0],circle.center[1],-circle.center[2]))
		return temp.T

	def get_rotated_sphere_matrix(self,circle,sphere):
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
		back[:] = np.array(circle.normal)
		back[-2] *=-1 #our z axis is inverted
		back[-0] *=-1 #our z axis is inverted
		back[:] /= np.linalg.norm(back)
		right[:] = get_perpendicular_vector(back)/np.linalg.norm(get_perpendicular_vector(back))
		up[:] = np.cross(right,back)/np.linalg.norm(np.cross(right,back))
		translation[:] = np.array((sphere.center[0],sphere.center[1],-sphere.center[2]))
		return temp.T

	############## DRAWING FUNCTIONS ##############################

	def draw_frustum(self, scale=1):
		# average focal length
		#f = (K[0, 0] + K[1, 1]) / 2
		# compute distances for setting up the camera pyramid
		W = self.intrinsics[0,2]
		H = self.intrinsics[1,2]
		Z = self.intrinsics[0,0]
		# scale the pyramid
		W *= scale
		H *= scale
		Z *= scale
		# draw it
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
		glColor3f( 0, 0,1 )
		glBegin( GL_LINES )
		glVertex3f( 0, 0, 0 )
		glVertex3f( 0, 0, l )
		glEnd( )

	def draw_sphere(self,circle,sphere,contours = 45):
		# this function draws the location of the eye sphere
		glPushMatrix()
		glLoadMatrixf(self.get_rotated_sphere_matrix(circle,sphere))

		glTranslatef(0,0,sphere.radius)
		draw_points(((0,0),),color=RGBA(0,1,0.2,.5))
		for i in xrange(1,contours+1):
			glTranslatef(0,0,-sphere.radius/contours*2)
			position = sphere.radius- i*sphere.radius*2/contours
			draw_radius = np.sqrt(sphere.radius**2 - position**2)
			glPushMatrix()
			glScalef(draw_radius,draw_radius,1)
			draw_polyline((rad),5,color=RGBA(0,1,0.2,.5))
			glPopMatrix()
			# draw_points(((0,0),),color=RGBA(0,1,0.2,.5))

		glPopMatrix()

	def draw_all_ellipses(self,model,number = 10):
		# draws all ellipses in model. numder determines last x amt of ellipses to show
		glPushMatrix()
		for observation in model.observations[-number:]:
			ellipse = observation.ellipse
			glColor3f(0.0, 1.0, 0.0)  #set color to green
			pts = cv2.ellipse2Poly( (int(ellipse.center[0]),int(ellipse.center[1])),
                                        (int(ellipse.major_radius),int(ellipse.minor_radius)),
                                        int(ellipse.angle*180/scipy.pi),0,360,15)
			draw_polyline(pts,4,color = RGBA(0,1,1,.5))
		glPopMatrix()

	def draw_all_circles(self,model,number = 10):
		for pupil in model.observations[-number:]: #draw the last 10
			self.draw_circle(pupil.circle)

	def draw_circle(self,circle):
		glPushMatrix()
		glLoadMatrixf(self.get_pupil_transformation_matrix(circle))
		draw_points(((0,0),),color=RGBA(1.1,0.2,.8))
		glScalef(circle.radius,circle.radius,1)
		draw_polyline((rad),color=RGBA(0.,0.,0.,.5), line_type = GL_POLYGON)
		glColor4f(0.0, 0.0, 0.0,0.5)  #set color to green
		glBegin(GL_POLYGON) #draw circle
		glEnd()
		glPopMatrix()

	def draw_eye_model_text(self, model):
		glLoadIdentity()
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		gluOrtho2D(0., 640.,0.0, 480.)
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()

		glTranslatef(5,35,0)
		glScalef(1,-1,0)
		self.glfont.draw_multi_line_text(0,0,'Eye model center: \n %s'%model.eye.center)
		glTranslatef(0,-20,0)
		self.glfont.draw_multi_line_text(0,0,'View: %s'%self.trackball.distance)

		glMatrixMode(GL_MODELVIEW)
		glPopMatrix()
		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glEnable(GL_TEXTURE_2D)

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
			self.trackball = Trackball()

			# get glfw started
			if self.run_independently:
				glfwInit()
			window = glfwGetCurrentContext()
			self._window = glfwCreateWindow(640, 480, self.name, None, window)
			glfwMakeContextCurrent(self._window)

			if not self._window:
				exit()

			glfwSetWindowPos(self._window,2000,0)
			# Register callbacks window
			glfwSetFramebufferSizeCallback(self._window,self.on_resize)
			glfwSetWindowIconifyCallback(self._window,self.on_iconify)
			glfwSetKeyCallback(self._window,self.on_key)
			glfwSetCharCallback(self._window,self.on_char)
			glfwSetMouseButtonCallback(self._window,self.on_button)
			glfwSetCursorPosCallback(self._window,self.on_pos)
			glfwSetScrollCallback(self._window,self.on_scroll)
			glfwSetWindowCloseCallback(self._window,self.on_close)

			# get glfw started
			if self.run_independently:
				init()
			glutInit() #can delete later
			self.basic_gl_setup()

			self.glfont = fs.Context()
			self.glfont.add_font('opensans',get_opensans_font_path())
			self.glfont.set_size(22)
			self.glfont.set_color_float((0.2,0.5,0.9,1.0))


			# self.gui = ui.UI()
			self.on_resize(self._window,*glfwGetFramebufferSize(self._window))

	def update_window(self, g_pool = None,model = None):
		if self.window_should_close:
			self.close_window()
		if self._window != None:
			glfwMakeContextCurrent(self._window)

			# glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			# glClearDepth(1.0)
			# glDepthFunc(GL_LESS)
			# glEnable(GL_DEPTH_TEST)
			# glAlphaFunc(GL_GREATER, 0)
			self.clear_gl_screen()

			self.trackball.push()

			#THINGS I NEED TO DRAW

			# 1. in anthromorphic space, draw pupil sphere and circles on it
			glLoadMatrixf(self.get_anthropomorphic_matrix())

			if model: #if we are feeding in spheres to draw
				self.draw_all_circles(model,10)
				self.draw_sphere(model.observations[-1].circle,model.eye) #draw the eyeball

			self.draw_coordinate_system(4)

			# 1b. draw frustum in pixel scale, but retaining origin
			glLoadMatrixf(self.get_adjusted_pixel_space_matrix(30))
			self.draw_frustum()

			# 2. in pixel space, draw ellipses, and video frame
			glLoadMatrixf(self.get_image_space_matrix(30))
			if g_pool: #if display eye camera frames
				draw_named_texture(g_pool.image_tex,quad=((0,480),(640,480),(640,0),(0,0)),alpha=0.5)
			self.draw_all_ellipses(model,10)

			# 3. draw eye model text
			self.draw_eye_model_text(model)

			self.trackball.pop()
			glfwSwapBuffers(self._window)
			glfwPollEvents()
			return True

	def close_window(self):
		if self.window_should_close == True:
			glfwDestroyWindow(self._window)
			if self.run_independently:
				glfwTerminate()
			self._window = None
			self.window_should_close = False
			logger.debug("Process done")

	############ window callbacks #################
	def on_resize(self,window,w, h):
		h = max(h,1)
		w = max(w,1)
		self.trackball.set_window_size(w,h)

		active_window = glfwGetCurrentContext()
		glfwMakeContextCurrent(window)
		self.adjust_gl_view(w,h)
		glfwMakeContextCurrent(active_window)

	def on_button(self,window,button, action, mods):
		# self.gui.update_button(button,action,mods)
		if action == GLFW_PRESS:
			self.input['button'] = button
			self.input['mouse'] = glfwGetCursorPos(window)
		if action == GLFW_RELEASE:
			self.input['button'] = None

		# pos = normalize(pos,glfwGetWindowSize(window))
		# pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels

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
		# self.gui.update_scroll(x,y)
		self.trackball.zoom_to(y)

	def on_close(self,window=None):
		self.window_should_close = True

	def on_iconify(self,window,x,y): pass
	def on_key(self,window, key, scancode, action, mods): pass #self.gui.update_key(key,scancode,action,mods)
	def on_char(window,char): pass # self.gui.update_char(char)

if __name__ == '__main__':
	intrinsics = np.matrix('879.193 0 320; 0 879.193 240; 0 0 1')
	huding = Visualizer("huding", run_independently = True, intrinsics = intrinsics)

	ellipse1 = geometry.Ellipse((419.14,181.08),44.28,33.03,1.32)
	ellipse2 = geometry.Ellipse((406.03,134.45),45.75,31.87,1.02)
	ellipse3 = geometry.Ellipse((224.99,177.82),50.97,46.14,2.04)
	ellipse4 = geometry.Ellipse((299.65,93.53),47.17,40.54,0.33)
	intrinsics = np.matrix('879.193 0 320; 0 -879.193 240; 0 0 1')
	eye_model = Sphere_Fitter(intrinsics = intrinsics)
	eye_model.add_observation(ellipse1)
	eye_model.add_observation(ellipse2)
	eye_model.add_observation(ellipse3)
	eye_model.add_observation(ellipse4)

	eye_model.unproject_observations()
	eye_model.initialize_model()

	huding.open_window()
	a = 0
	while huding.update_window(model = eye_model):
		a += 1
	huding.close_window()
	print a