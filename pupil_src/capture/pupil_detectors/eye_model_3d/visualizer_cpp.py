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
# from intersect import sphere_intersect #not gonna work
# from geometry  import Line3D #not gonna work
import numpy as np
import scipy
# import geometry #not gonna work
# from __init__ import Sphere_Fitter #not gonna work
import cv2

import build_test

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

		if intrinsics is None:
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

	def get_adjusted_pixel_space_matrix(self):
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

	def get_pupil_transformation_matrix(self,circle_normal,circle_center):
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
		back[-2] *=-1 #our z axis is inverted
		back[-0] *=-1 #our x axis is inverted
		# if np.linalg.norm(back) != 0:
		back[:] /= np.linalg.norm(back)
		right[:] = get_perpendicular_vector(back)/np.linalg.norm(get_perpendicular_vector(back))
		up[:] = np.cross(right,back)/np.linalg.norm(np.cross(right,back))
		translation[:] = np.array(circle_center)
		translation[2] *= -1
		return temp.T

	def get_rotated_sphere_matrix(self,circle_normal,sphere_center):
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
		back[-2] *=-1 #our z axis is inverted
		back[-0] *=-1 #our x axis is inverted
		back[:] /= np.linalg.norm(back)
		right[:] = get_perpendicular_vector(back)/np.linalg.norm(get_perpendicular_vector(back))
		up[:] = np.cross(right,back)/np.linalg.norm(np.cross(right,back))
		translation[:] = np.array(sphere_center)
		translation[2] *= -1 #not sure if this is correct, maybe depends on camera?
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

	def draw_sphere(self,pupil_observation,sphere,contours = 45):
		# this function draws the location of the eye sphere
		glPushMatrix()
		circle_normal = pupil_observation[2][1]
		glLoadMatrixf(self.get_anthropomorphic_matrix())
		# glLoadMatrixf(self.get_rotated_sphere_matrix(circle_normal,sphere[0]))
		glTranslatef(sphere[0][0],sphere[0][1],sphere[0][2]) #sphere[0] contains center coordinates (x,y,z)
		glTranslatef(0,0,sphere[1]) #sphere[1] contains radius
		draw_points(((0,0),),color=RGBA(0,1,0.2,.5))
		for i in xrange(1,contours+1):
			glTranslatef(0,0,-sphere[1]/contours*2)
			position = sphere[1]- i*sphere[1]*2/contours
			draw_radius = np.sqrt(sphere[1]**2 - position**2)
			glPushMatrix()
			glScalef(draw_radius,draw_radius,1)
			draw_polyline((rad),2,color=RGBA(.2,.5,0.5,.5))
			glPopMatrix()
			# draw_points(((0,0),),color=RGBA(0,1,0.2,.5))

		glPopMatrix()

	def draw_all_ellipses(self,cpp_model,number = 0):
		# draws all ellipses in model. numder determines last x amt of ellipses to show
		glPushMatrix()
		if number == 0 or number > cpp_model.num_observations: #print everything. or statement in case try to access too many
			for pupil in cpp_model.get_all_pupil_observations():
				#ellipse is pupil[0]. ellipse is ([x,y], major, minor, angle)
				glColor3f(0.0, 1.0, 0.0)  #set color to green
				pts = cv2.ellipse2Poly( (int(pupil[0][0][0]),int(pupil[0][0][1])),
	                                        (int(pupil[0][1]),int(pupil[0][2])),
	                                        int(pupil[0][3]*180/scipy.pi),0,360,15)
				draw_polyline(pts,4,color = RGBA(0,1,1,.5))
		else:
			for pupil in cpp_model.get_last_pupil_observations(number):
				#ellipse is pupil[0]. ellipse is ([x,y], major, minor, angle)
				glColor3f(0.0, 1.0, 0.0)  #set color to green
				pts = cv2.ellipse2Poly( (int(pupil[0][0][0]),int(pupil[0][0][1])),
	                                        (int(pupil[0][1]),int(pupil[0][2])),
	                                        int(pupil[0][3]*180/scipy.pi),0,360,15)
				draw_polyline(pts,4,color = RGBA(0,1,1,.5))
		glPopMatrix()

	def draw_all_circles(self,cpp_model,number = 0):
		if number == 0 or number > cpp_model.num_observations: #print everything. or statement in case try to access too many
			for pupil in cpp_model.get_all_pupil_observations():
				#circle is pupil[2]. circle is (center[x,y,z], normal[x,y,z], radius)
				glPushMatrix()
				glLoadMatrixf(self.get_pupil_transformation_matrix(pupil[2][1],pupil[2][0])) #circle normal, center
				draw_points(((0,0),),color=RGBA(1.1,0.2,.8))
				glScalef(pupil[2][2],pupil[2][2],1) #scale by pupil radius
				draw_polyline((rad),color=RGBA(0.,0.,0.,.5), line_type = GL_POLYGON)
				glColor4f(0.0, 0.0, 0.0,0.5)  #set color to green
				glBegin(GL_POLYGON) #draw circle
				glEnd()
				glPopMatrix()
		else:
			for pupil in cpp_model.get_last_pupil_observations(number):
				#circle is pupil[2]. circle is (center[x,y,z], normal[x,y,z], radius)
				glPushMatrix()
				glLoadMatrixf(self.get_pupil_transformation_matrix(pupil[2][1],pupil[2][0])) #circle normal, center
				draw_points(((0,0),),color=RGBA(1.1,0.2,.8))
				glScalef(pupil[2][2],pupil[2][2],1) #scale by pupil radius
				draw_polyline((rad),color=RGBA(0.,0.,0.,.5), line_type = GL_POLYGON)
				glColor4f(0.0, 0.0, 0.0,0.5)  #set color to green
				glBegin(GL_POLYGON) #draw circle
				glEnd()
				glPopMatrix()


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

	def draw_contours_on_screen(self,contours):
		#this function displays the contours on the 2D video stream within the visualizer module
		glPushMatrix()
		glLoadMatrixf(self.get_image_space_matrix(30))
		for contour in contours:
			contour_2d = [point[0] for point in contour]
			draw_polyline(contour_2d,color=RGBA(0,0,0,0.5))
		glPopMatrix()

	def draw_contours(self,contours,cpp_model):
		# for contour in contours:
		# 	intersect_contour = [sphere_intersect(Line3D((0,0,0),geometry.unproject_point(c[0],20,self.intrinsics)),model.eye) for c in contour]
		# 	intersect_contour = [c[0] for c in intersect_contour if c is not None]
		# 	draw_polyline3d(np.array(intersect_contour),color=RGBA(0.,0.,0.,.5))
		# num = 0
		for contour in contours:
			# draw_polyline3d(cpp_model.intersect_contour_with_eye(contour),color=RGBA(0.,0.,0.,.5))
			intersect_contour = [self.sphere_intersect(point[0],cpp_model.eye,cpp_model) for point in contour]
			intersect_contour = [c for c in intersect_contour if c is not None]
			draw_polyline3d(np.array(intersect_contour),color=RGBA(0.,0.,0.,.5))
			# num += len(intersect_contour) 
		# print num #see how many points are inside contours

	def draw_eye_model_text(self, model):
		params = model.get_pupil_observation(0)[1] #0 is temporary, should be -1 but can't do that in cpp
		status = 'Eyeball center : X%.2fmm Y%.2fmm Z%.2fmm\nGaze vector (currently WRONG): Theta: %.3f Psi %.3f\nPupil Diameter: %.2fmm'%(model.eye[0][0],
			model.eye[0][1],model.eye[0][2],
			params[0], params[1], params[2]*2)
		self.glfont.draw_multi_line_text(5,20,status)
		self.glfont.draw_multi_line_text(440,20,'View: %.2f %.2f %.2f'%(self.trackball.distance[0],self.trackball.distance[1],self.trackball.distance[2]))

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
			self.on_resize(self._window,*glfwGetFramebufferSize(self._window))
			glfwMakeContextCurrent(window)

			# self.gui = ui.UI()

	def update_window(self, g_pool = None, cpp_model = None, contours = None):
		if self.window_should_close:
			self.close_window()
		if self._window != None:
			glfwMakeContextCurrent(self._window)
			# print glGetDoublev(GL_MODELVIEW_MATRIX), glGetDoublev(GL_PROJECTION_MATRIX)

			# glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			# glClearDepth(1.0)
			# glDepthFunc(GL_LESS)
			# glEnable(GL_DEPTH_TEST)
			# glAlphaFunc(GL_GREATER, 0)
			self.clear_gl_screen()
			self.trackball.push()

			# 1. in anthromorphic space, draw pupil sphere and circles on it
			glLoadMatrixf(self.get_anthropomorphic_matrix())

			if cpp_model and cpp_model.num_observations > 3:
				# self.draw_all_circles(cpp_model, 10)
				self.draw_sphere(cpp_model.get_last_pupil_observation(),cpp_model.eye) #draw CPP

			glLoadMatrixf(self.get_anthropomorphic_matrix())

			self.draw_coordinate_system(4)
			if contours:
				self.draw_contours_on_screen(contours)
				self.draw_contours(contours,cpp_model)
			# 1b. draw frustum in pixel scale, but retaining origin
			glLoadMatrixf(self.get_adjusted_pixel_space_matrix(30))
			self.draw_frustum()

			# 2. in pixel space, draw ellipses, and video frame
			glLoadMatrixf(self.get_image_space_matrix(30))
			if g_pool: #if display eye camera frames
				draw_named_texture(g_pool.image_tex,quad=((0,480),(640,480),(640,0),(0,0)),alpha=0.5)
			self.draw_all_ellipses(cpp_model, 5)

			self.trackball.pop()
			# 3. draw eye model text
			if cpp_model and cpp_model.num_observations > 3:
				self.draw_eye_model_text(cpp_model)

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

	def sphere_intersect(self,point,sphere,cpp_model):
		#auxiliary function originally in intersect.py, used here to find sphere intersect.
		#intersection between a line and a sphere, originally called intersect(line,sphere)
		#we only return the closer value

		#first, need to unproject point to 3d. We put z position at 1 arbitrarily.
		# direction= np.array([(point[0]-self.intrinsics[0,2])/ self.intrinsics[0,0],
	 #                    (point[1]-self.intrinsics[1,2])/ self.intrinsics[1,1],1])
		# # return direction #this prints unprojected at z=20. Not on sphere
		# origin= np.array([0,0,0])
		# s_center= np.array(sphere[0]) #the center
		# s_radius= sphere[1] #the radius

		# # print origin, direction, s_center, s_radius

		# vcvc_cc_rr = direction.dot(s_center)**2 - s_center.dot(s_center) + s_radius**2 # from wikipedia :)
		# if (vcvc_cc_rr < 0):
		# 	# logger.warning("NO INTERSECTION between line and sphere")
		# 	return None
		# s1 = direction.dot(s_center) - np.sqrt(vcvc_cc_rr)
		# # s2 = direction.dot(s_center) + np.sqrt(vcvc_cc_rr)
		# p1 = origin + s1*direction
		# # p2 = origin + s2*direction
		# return p1 #, p2 a line intersects a sphere at two points

		# point[0] = point[0]*cpp_model.scale
		# point[1] = point[1]*cpp_model.scale #not sure if this works

		v = np.array([(point[0]-self.intrinsics[0,2])/ self.intrinsics[0,0],
	                    (point[1]-self.intrinsics[1,2])/ self.intrinsics[1,1],1])
		p = (0,0,0) #put p at origin
		c = np.array(sphere[0]) #sphere center
		r = sphere[1] # /cpp_model.scale #sphere radius

		vcvc_cc_rr = v.dot(c)**2 - c.dot(c) + r**2 # from wikipedia :)
		if (vcvc_cc_rr < 0):
			# logger.warning("NO INTERSECTION between line and sphere")
			return None
		s1 = v.dot(c) - np.sqrt(vcvc_cc_rr)
		# s2 = v.dot(c) + np.sqrt(vcvc_cc_rr)
		p1 = p + s1*v
		# p2 = p + s2*v
		return p1 #,p2 #a line intersects a sphere at two points

if __name__ == '__main__':
	print "done"

	huding = build_test.eye_model_3d.PyEyeModelFitter(focal_length=879.193, x_disp = 320, y_disp = 240)
	# print model
	huding.add_observation([422.255,255.123],40.428,30.663,1.116)
	huding.add_observation([442.257,365.003],44.205,32.146,1.881)
	huding.add_observation([307.473,178.163],41.29,22.765,0.2601)
	huding.add_observation([411.339,290.978],51.663,41.082,1.377)
	huding.add_observation([198.128,223.905],46.852,34.949,2.659)
	huding.add_observation([299.641,177.639],40.133,24.089,0.171)
	huding.add_observation([211.669,212.248],46.885,33.538,2.738)
	huding.add_observation([196.43,236.69],47.094,38.258,2.632)
	huding.add_observation([317.584,189.71],42.599,27.721,0.3)
	huding.add_observation([482.762,315.186],38.397,23.238,1.519)
	huding.update_model()

	# print huding.print_eye()	
	# for pupil in huding.get_all_pupil_observations():
	# 		#circle is pupil[2]. circle is (center[x,y,z], normal[x,y,z], radius)
	# 		print pupil[2]

	# contours = [[[[38, 78]], [[39, 78]]],
	# 	[[[ 65, 40]], [[ 66, 39]], [[ 67, 40]], [[ 68, 40]], [[ 69, 41]], [[ 70, 41]], [[ 71, 40]], [[ 72, 41]], [[ 73, 41]], [[ 74, 41]], [[ 75, 42]], [[ 76, 42]], [[ 77, 42]], [[ 78, 42]], [[ 79, 43]], [[ 80, 44]], [[ 81, 45]], [[ 82, 45]], [[ 83, 45]], [[ 84, 46]], [[ 85, 46]], [[ 86, 47]], [[ 87, 47]], [[ 88, 48]], [[ 89, 49]], [[ 90, 50]], [[ 90, 51]],[[ 91, 52]],[[ 92, 53]],[[ 93, 53]],[[ 94, 54]],[[ 95, 55]],[[ 96, 56]],[[ 96, 57]],[[ 97, 58]],[[ 98, 59]],[[ 99, 60]],[[ 99, 61]],[[100, 62]],[[100, 63]],[[100, 64]], [[101, 65]],[[101, 66]],[[102, 67]],[[102, 68]],[[103, 69]], [[103, 70]],[[103, 71]], [[104, 72]],[[104, 73]],[[104, 74]],[[104, 75]],[[104, 76]],[[104, 77]],[[104, 78]],[[104, 79]], [[104, 80]],[[104, 81]],[[104, 82]],[[104, 83]],[[104, 84]],[[104, 85]], [[103, 86]],[[103, 87]],[[103, 88]], [[103, 89]],[[103, 90]],[[103, 91]],[[102, 92]], [[101, 93]],[[101, 94]],[[100, 95]], [[100, 96]],[[ 99, 97]],[[ 98, 98]],[[ 97, 99]], [[ 96, 100]],[[ 95, 101]],[[ 94, 101]],[[ 93, 102]],[[ 92, 103]],[[ 91, 103]], [[ 90, 104]],[[ 89, 104]],[[ 88, 104]],[[ 87, 104]],[[ 86, 104]],[[ 85, 105]],[[ 84, 105]],[[ 83, 105]],[[ 82, 105]],[[ 81, 105]],[[ 80, 106]],[[ 79, 106]],[[ 78, 106]],[[ 77, 106]],[[ 76, 106]],[[ 75, 105]],[[ 74, 105]],[[ 73, 105]],[[ 72, 105]],[[ 71, 104]],[[ 70, 104]],[[ 69, 104]],[[ 68, 104]],[[ 67, 103]],[[ 66, 103]],[[ 65, 102]],[[ 64, 101]],[[ 63, 100]],[[ 62, 100]], [[ 61, 99]],[[ 60, 99]],[[ 59, 98]],[[ 58, 97]],[[ 57, 97]], [[ 56, 96]],[[ 55, 95]],[[ 54, 94]],[[ 53, 93]],[[ 52, 92]],[[ 51, 91]],[[ 51, 90]], [[ 50, 89]], [[ 49, 88]],[[ 48, 87]],[[ 47, 86]],[[ 47, 85]],[[ 47, 84]], [[ 46, 83]], [[ 45, 82]],[[ 45, 81]],[[ 45, 80]],[[ 45, 79]],[[ 44, 78]],[[ 44, 77]],[[ 43, 76]],[[ 43, 75]],[[ 43, 74]],[[ 42, 73]],[[ 42, 72]],[[ 42, 71]], [[ 42, 70]],[[ 41, 69]],[[ 41, 68]],[[ 41, 67]],[[ 41, 66]],[[ 41, 65]], [[ 41, 64]],[[ 41, 63]],[[ 41, 62]], [[ 41, 61]],[[ 41, 60]], [[ 41, 59]], [[ 41, 58]],[[ 42, 57]],[[ 42, 56]],[[ 43, 55]],[[ 43, 54]],[[ 44, 53]],[[ 44, 52]],[[ 45, 51]],[[ 45, 50]],[[ 46, 49]],[[ 47, 48]],[[ 48, 47]],[[ 49, 47]],[[ 50, 46]],[[ 51, 45]],[[ 52, 45]],[[ 53, 44]],[[ 54, 43]],[[ 55, 42]],[[ 56, 42]],[[ 57, 41]],[[ 58, 41]],[[ 59, 41]],[[ 60, 41]],[[ 61, 41]],[[ 62, 40]],[[ 63, 40]],[[ 64, 40]]]
	# 	[[[ 66, 39]],[[ 65, 40]],[[ 64, 40]],[[ 63, 40]],[[ 62, 40]],[[ 61, 41]],[[ 60, 41]],[[ 59, 41]],[[ 58, 41]],[[ 57, 41]],[[ 56, 42]],[[ 55, 42]],[[ 54, 42]],[[ 53, 43]],[[ 52, 44]],[[ 51, 45]],[[ 50, 45]],[[ 49, 46]],[[ 48, 47]],[[ 47, 47]],[[ 46, 48]],[[ 45, 49]],[[ 45, 50]],[[ 44, 51]],[[ 44, 52]],[[ 43, 53]],[[ 43, 54]],[[ 42, 55]],[[ 42, 56]],[[ 42, 57]],[[ 41, 58]],[[ 41, 59]],[[ 41, 60]],[[ 41, 61]],[[ 41, 62]],[[ 41, 63]],[[ 41, 64]],[[ 41, 65]],[[ 41, 66]],[[ 41, 67]],[[ 41, 68]],[[ 41, 69]],[[ 42, 70]],[[ 42, 71]], [[ 42, 72]],[[ 42, 73]],[[ 43, 74]],[[ 43, 75]],[[ 43, 76]],[[ 44, 77]],[[ 44, 78]],[[ 44, 79]],[[ 45, 80]],[[ 45, 81]],[[ 45, 82]],[[ 46, 83]],[[ 46, 84]],[[ 47, 85]],[[ 47, 86]],[[ 48, 87]],[[ 48, 88]],[[ 49, 89]],[[ 50, 90]],[[ 51, 91]],[[ 51, 92]],[[ 52, 93]],[[ 53, 94]],[[ 54, 95]],[[ 55, 96]],[[ 56, 97]],[[ 57, 97]],[[ 58, 98]],[[ 59, 99]],[[ 60, 99]],[[ 61, 100]],[[ 62, 100]],[[ 63, 101]],[[ 64, 102]],[[ 65, 103]],[[ 66, 103]],[[ 67, 103]],[[ 68, 104]],[[ 69, 104]],[[ 70, 104]],[[ 71, 104]],[[ 72, 105]],[[ 73, 105]],[[ 74, 105]],[[ 75, 105]],[[ 76, 106]],[[ 77, 106]],[[ 78, 106]],[[ 79, 106]],[[ 80, 106]],[[ 81, 105]], [[ 82, 105]],[[ 83, 105]],[[ 84, 105]],[[ 85, 105]],[[ 86, 104]],[[ 87, 104]],[[ 88, 104]],[[ 89, 104]],[[ 90, 104]],[[ 91, 103]],[[ 92, 103]],[[ 93, 103]],[[ 94, 102]],[[ 95, 101]],[[ 96, 101]],[[ 97, 100]],[[ 98, 99]],[[ 99, 98]],[[100, 97]],[[100, 96]],[[101, 95]],[[101, 94]],[[102, 93]],[[103, 92]],[[103, 91]],[[103, 90]],[[103, 89]],[[103, 88]],[[103, 87]],[[103, 86]],[[104, 85]],[[104, 84]],[[104, 83]],[[104, 82]],[[104, 81]],[[104, 80]],[[104, 79]],[[104, 78]],[[104, 77]],[[104, 76]],[[104, 75]], [[104, 74]],[[104, 73]],[[104, 72]],[[103, 71]],[[103, 70]],[[103, 69]],[[102, 68]],[[102, 67]],[[102, 66]],[[101, 65]],[[101, 64]],[[100, 63]],[[100, 62]],[[ 99, 61]],[[ 99, 60]],[[ 99, 59]],[[ 98, 58]],[[ 97, 57]],[[ 96, 56]],[[ 96, 55]],[[ 95, 54]],[[ 94, 53]],[[ 93, 53]],[[ 92, 52]],[[ 91, 51]],[[ 90, 50]],[[ 90, 49]],[[ 89, 48]],[[ 88, 47]],[[ 87, 47]],[[ 86, 46]],[[ 85, 46]],[[ 84, 45]],[[ 83, 45]],[[ 82, 44]],[[ 81, 44]],[[ 80, 43]],[[ 79, 42]],[[ 78, 42]],[[ 77, 42]], [[ 76, 42]],[[ 75, 42]],[[ 74, 41]],[[ 73, 41]],[[ 72, 41]],[[ 71, 40]],[[ 70, 41]],[[ 69, 41]],[[ 68, 40]],[[ 67, 40]]]]
	contours = [[[[20,20]],[[21,20]],[[22,23]]]]

	intrinsics = np.matrix('879.193 0 320; 0 -879.193 240; 0 0 1')
	visualhuding = Visualizer("huding", run_independently = True, intrinsics = intrinsics)

	# print visualhuding.sphere_intersect([320,240],((0,0,49),12))

	visualhuding.open_window()
	a = 0
	while visualhuding.update_window(cpp_model = huding, contours = contours):
	# while visualhuding.update_window(cpp_model = huding):
		a += 1
	# visualhuding.update_window(cpp_model = huding)
	visualhuding.close_window()
	print a