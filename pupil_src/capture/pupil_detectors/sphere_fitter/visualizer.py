"""
	Andrew Xia working on visualizing data.
	I want to use opengl to display the 3d sphere and lines that connect to it.
	This file is in pupil-labs-andrew/sphere_fitter, so it is the prototype version
	July 6 2015

"""
import logging
from glfw import *
from OpenGL.GL import *

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
import geometry #how do I find this folder?
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

class Visualizer():
	def __init__(self,name = "unnamed", focal_length = 554.25625, intrinsics = None, run_independently = False):
		self.sphere = geometry.Sphere((0,0,0),0) #the eyeball, initialized as something random
		self.ellipses = [] #collection of ellipses
		self.circles = [] #collection of all 3D circles on the sphere
		self.video_frame = (np.linspace(0,1,num=(400*400*4))*255).astype(np.uint8).reshape((400,400,4)) #the randomized image, should be video frame
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
		self.scale = 1

		self.name = name
		self._window = None
		self.input = None
		self.trackball = None
		self.run_independently = run_independently

		self.window_should_close = False

	############## MATRIX FUNCTIONS ##############################

	def get_pixel_space_matrix(self):
		# returns a homoegenous matrix
		temp = self.get_anthropomorphic_matrix()
		temp[3,3] *= self.scale
		return temp

	def get_adjusted_pixel_space_matrix(self,scale):
		# returns a homoegenous matrix
		temp = self.get_pixel_space_matrix()
		temp[3,3] *= scale
		return temp

	def get_anthropomorphic_matrix(self):
		temp =  np.identity(4)
		temp[2,2] *=-1 #consistent with our 3d coord system
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

	def draw_rect(self):
		glBegin(GL_QUADS)
		glColor4f(0.0, 0.0, 0.5,0.2)  #set color to light blue
		glVertex2f(0,0)
		glVertex2f(0 + 32, 0)
		glVertex2f(0 + 32, 0 + 16)
		glVertex2f(0, 0 + 16)
		glEnd()

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

	def draw_sphere(self,circle):
		# this function draws the location of the eye sphere
		glPushMatrix()
		# glColor4f(0.0, 1.0, 1.0,0.2)  #set color to blue
		glLoadMatrixf(self.get_rotated_sphere_matrix(circle,self.sphere))
		# glutSolidSphere(self.sphere.radius,20,10)
		glColor4f(0.0, 0.0, 1.0,0.8)  #set color to blue
		# glutWireSphere(self.sphere.radius,20,10)
		glPopMatrix()

	def draw_all_ellipses(self):
		# draws all ellipses in self.ellipses.
		glPushMatrix()
		for ellipse in self.ellipses[-10:]:
			glColor3f(0.0, 1.0, 0.0)  #set color to green
			pts = cv2.ellipse2Poly( (int(ellipse.center[0]),int(ellipse.center[1])),
                                        (int(ellipse.major_radius),int(ellipse.minor_radius)),
                                        int(ellipse.angle*180/scipy.pi),0,360,15)
			draw_polyline(pts,2,color = RGBA(0,1,1,.5))
		glPopMatrix()

	def draw_circle(self,circle):
		glPushMatrix()
		glLoadMatrixf(self.get_pupil_transformation_matrix(circle))
		draw_points(((0,0),),color=RGBA(1.1,0.2,.8))
		glColor4f(0.0, 0.0, 0.0,0.5)  #set color to green
		glBegin(GL_POLYGON) #draw circle
		for i in xrange(45):
			rad = i*16*scipy.pi/360.
			glVertex2f(np.cos(rad)*circle.radius,np.sin(rad)*circle.radius)
		glEnd()
		glPopMatrix()

	def draw_video_screen(self):
		# Function to draw self.video_frame.
		# Need to scale
		glPushMatrix()
		tex_id = create_named_texture(self.video_frame.shape)
		update_named_texture(tex_id,self.video_frame) #since image doesn't change, do not need to put in while loop
		draw_named_texture(tex_id)
		glPopMatrix()

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
				for pupil in model.observations[-10:]: #draw the last 10
					self.draw_circle(pupil.circle)

			self.draw_sphere(pupil.circle) #draw the eyeball

			self.draw_coordinate_system(4)
			glTranslatef(*self.sphere.center)
			glScalef(0.1,0.1,0.1)
			self.glfont.draw_multi_line_text(0,0,'Eyeball location: \n %s'%self.sphere.center)

			# 1a. draw frustum in pixel scale, but retaining origin
			glLoadMatrixf(self.get_adjusted_pixel_space_matrix(30))
			self.draw_frustum()

			# 2. in pixel space, draw ellipses, and video frame
			glLoadMatrixf(self.get_image_space_matrix(30))
			if g_pool: #if display eye camera frames
				draw_named_texture(g_pool.image_tex,quad=((0,480),(640,480),(640,0),(0,0)),alpha=0.5)
			self.draw_all_ellipses()

			# 3. display text on screen
			# glWindowPos2i(100, 100)
			# glutPrint("SAMPLE TEXT")


			# glFontBegin(font);
			# glScalef(8.0, 8.0, 8.0);
			# glTranslatef(30, 30, 0);
			# glFontTextOut("Test", 5, 5, 0);
			# glFontEnd();
			# glFlush();


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

	def on_iconify(self,window,x,y): pass

	def on_key(self,window, key, scancode, action, mods): pass
		#self.gui.update_key(key,scancode,action,mods)

	def on_char(window,char): pass
		# self.gui.update_char(char)

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

if __name__ == '__main__':
	intrinsics = np.matrix('879.193 0 320; 0 879.193 240; 0 0 1')
	huding = Visualizer("huding", run_independently = True, intrinsics = intrinsics)

	huding.ellipses.append(geometry.Ellipse((389.67633057,340.16584778), 44.97, 24, 2*scipy.pi/3))
	huding.ellipses.append(geometry.Ellipse((389.67633057,340.16584778), 44.97, 24, -2*scipy.pi/3))
	# huding.ellipses.append(geometry.Ellipse((370.1632843,225.21165466), 42.16, 9.7, 1.7))
	# huding.ellipses.append(geometry.Ellipse((390.1632843,225.21165466), 42.16, 29.7, 3.7))
	# huding.ellipses.append(geometry.Ellipse([ 504.64957428 , 298.43984222],42.28, 14.13, 4.38*scipy.pi))
	# huding.ellipses.append(geometry.Ellipse((4.86,-48.18)   ))

	# huding.ellipses.append(geometry.Ellipse((232,58),45,26,4.62))

	huding.circles.append(geometry.Circle3D((1,1,2),(1,0,0),3)) #pointing in x dir
	huding.circles.append(geometry.Circle3D((0,0,0),(0,1,0),1)) #pointing in y dir
	huding.sphere.center = [11.63136843 , 16.53159359 , 40.12095085]
	# huding.sphere.center = [5,5,10]
	huding.sphere.radius = 5.
	huding.open_window()
	a = 0
	while huding.update_window():
		a += 1
	huding.close_window()
	print a

	# # print convert_fov(60,640)
	b = np.array((1,1,1))
	a = get_perpendicular_vector(b)
	c = np.cross(a,b)
	print np.dot(a,b)
	print np.dot(a,c)


	circ = geometry.Circle3D((1,1,2),(0,0,1),3)
	print huding.get_pupil_transformation_matrix(circ).T
