import glumpy 
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glumpy.atb as atb
from ctypes import *
import numpy as np

from methods import *
from calibrate import *
from gl_shapes import Point, Ellipse
from methods import Temp,capture
from multiprocessing import Queue, Value

class Bar(atb.Bar):
	"""docstring for Bar"""
	def __init__(self, name,g_pool, defs):
		super(Bar, self).__init__(name,**defs) 
		self.fps = 0.0 
		self.display = c_int(1)
		self.exit = c_bool(0)
		self.spec_lower = c_int(250)
		self.spec_upper = c_int(3)
		self.bin_lower = c_int(0)
		self.bin_upper = c_int(0)
		self.pupil_point = c_bool(1)

		self.add_var("FPS", step=0.01, getter=self.get_fps)
		self.add_var("Display", self.display, step=1,max=4, min=0)
		self.add_var("Show Pupil Point", self.pupil_point)		
		self.add_var("Specular/S_Lower", self.spec_lower, step=1, max=256, min=0)
		self.add_var("Specular/S_Upper",self.spec_upper, step=1, max=256, min=0)
		self.add_var("Binary/B_Lower", self.bin_lower, step=1,max=256, min=0)
		self.add_var("Binary/B_Upper", self.bin_upper, step=1,max=256, min=0)

		self.add_var("Exit", g_pool.quit)

	def update_fps(self, dt):
		temp_fps = 1/dt
		self.fps += 0.1*(temp_fps-self.fps)

	def get_fps(self):
		return self.fps

class Roi(object):
	"""this is a simple Region of Interest class
	it is applied on nump arrays for convinient slicing"""
	def __init__(self, array_shape):
		self.array_shape = array_shape
		self.lX = 10
		self.lY = 10
		self.uX = array_shape[1]-10
		self.uY = array_shape[0]-10
		self.nX = 0
		self.nY = 0

	def setStart(self,(x,y)):
		x,y = max(0,x),max(0,y)
		self.nX,self.nY = x,y

	def setEnd(self,(x,y)):
		x,y = max(0,x),max(0,y)
		if x != self.nX and y != self.nY:
			self.lX = min(x,self.nX)
			self.lY = min(y,self.nY)
			self.uX = max(x,self.nX)
			self.uY = max(y,self.nY)		

	def add_vector(self,(x,y)):
		return (self.lX+x,self.lY+y)

			
def eye(src, g_pool, eye_pipe):
	"""eye
		- Initialize glumpy figure, image, atb controls
		- Execute the glumpy main glut loop
	"""
	# Get image array from queue, initialize glumpy, map img_arr to opengl texture 
	cap = capture(src,(640,480))
	s, img_arr = cap.read_RGB()

	fig = glumpy.figure((img_arr.shape[1], img_arr.shape[0]))
	image = glumpy.Image(img_arr)
	image.x, image.y = 0,0


	# pupil object
	pupil = Temp()
	pupil.norm_coords = (0,0)
	pupil.image_coords = (0,0)
	pupil.screen_coords = (0,0)
	pupil.ellipse = None
	pupil.map_coords = (0,0)
	pupil.coefs = None
	pupil.pt_cloud = None

	r = Roi(img_arr.shape)
	
	# local object
	l_pool = Temp()
	l_pool.calib_running = False
	l_pool.record_running = False
	l_pool.record_positions = []
	l_pool.record_path = None

	# initialize gl shape primitives
	pupil_point = Point()
	pupil_ellipse = Ellipse()
	# Initialize ant tweak bar inherits from atb.Bar (see Bar class)
	atb.init()
	bar = Bar("Eye",g_pool, dict(label="Controls",
			help="Scene controls", color=(50,50,50), alpha=50,
			text='light', position=(10, 10), size=(200, 200)) )



	def on_idle(dt):
		bar.update_fps(dt)
		
		s,img = cap.read_RGB()
		
		###IMAGE PROCESSING 
		gray_img = grayscale(img[r.lY:r.uY,r.lX:r.uX])
		spec_img = erase_specular_new(gray_img, bar.spec_lower.value, bar.spec_upper.value)
		# spec_img = equalize(spec_img)     
		# spec_img = dif_gaus(spec_img, bar.bin_lower, bar.bin_upper)
		# binary_img = adaptive_threshold(spec_img, bar.bin_lower, bar.bin_upper)
		binary_img = extract_darkspot(spec_img, bar.bin_lower.value, bar.bin_upper.value)
		pupil.ellipse = fit_ellipse(binary_img)
		

		if pupil.ellipse:
			pupil.image_coords = r.add_vector(pupil.ellipse['center'])
			# numpy array wants (row,col) for an image this = (height,width)
			# therefore: img.shape[1] = xval, img.shape[0] = yval
			pupil.norm_coords = normalize(pupil.image_coords, img.shape[1], img.shape[0])
			pupil.screen_coords = denormalize(pupil.norm_coords, fig.width, fig.height)
			pupil_point.update(pupil.screen_coords)
			pupil_ellipse.update(pupil.screen_coords, pupil.ellipse)

			# for the world screen
			pupil.map_coords = map_vector(pupil.norm_coords, pupil.coefs)
			g_pool.pupil_x.value, g_pool.pupil_y.value = pupil.map_coords


		if bar.display.value == 0:
			img_arr[...] = img
		elif bar.display.value == 1:
			img_arr[r.lY:r.uY,r.lX:r.uX]=  np.dstack((gray_img,gray_img,gray_img))
		elif bar.display.value == 2:
			img_arr[r.lY:r.uY,r.lX:r.uX] = np.dstack((spec_img, spec_img, spec_img))
		elif bar.display.value == 3:
			img_arr[r.lY:r.uY,r.lX:r.uX] = np.dstack((binary_img, binary_img, binary_img))
		else:
			pass

		###CALIBRATION and MAPPING###
		# Initialize Calibration (setup variables and lists)
		if g_pool.calibrate.value and not l_pool.calib_running:
			l_pool.calib_running = True
			pupil.pt_cloud = [] 
			pupil.coefs = None

		# While Calibrating... 
		if l_pool.calib_running and (g_pool.pattern_x.value or g_pool.pattern_y.value) and pupil.ellipse:
			pupil.pt_cloud.append([pupil.norm_coords[0],pupil.norm_coords[1],
								g_pool.pattern_x.value, g_pool.pattern_y.value])

		# Calculate coefs
		if not g_pool.calibrate.value and l_pool.calib_running:         
			l_pool.calib_running = 0
			if pupil.pt_cloud:
				pupil.coefs = calibrate_poly(pupil.pt_cloud)

		
		###RECORDING###
		# Setup variables and lists for recording
		if g_pool.pos_record.value and not l_pool.record_running:
			l_pool.record_path = eye_pipe.recv()
			print "l_pool.record_path: ", l_pool.record_path
			l_pool.record_positions = []
			l_pool.record_running = True

		# While recording... 
		if l_pool.record_running:
			l_pool.record_positions.append([pupil.map_coords[0], pupil.map_coords[1], dt, g_pool.frame_count_record.value])

		# Save values and flip switch to off for recording
		if not g_pool.pos_record.value and l_pool.record_running:
			np.save(l_pool.record_path, np.asarray(l_pool.record_positions))
			l_pool.record_running = False


		image.update()
		fig.redraw()
		
		if g_pool.quit.value:
			print "EYE Process closing from global or atb"
			fig.window.stop()

	
	def on_draw():
		fig.clear(0.0, 0.0, 0.0, 1.0)
		image.draw(x=image.x, y=image.y, z=0.0, 
					width=fig.width, height=fig.height)
		
		if bar.pupil_point and pupil.ellipse:
			pupil_point.draw()
			pupil_ellipse.draw()


	def on_close():
		g_pool.quit.value = True
		print "EYE Process closed from window"

	def on_mouse_press(x, y, button):
		x,y = denormalize(normalize((x,y),fig.width,fig.height),img_arr.shape[1],img_arr.shape[0],flip_y=True) 
		r.setStart((x,y))
		pass

	def on_mouse_drag(x, y, dx, dy, buttons):
		x,y = denormalize(normalize((x,y),fig.width,fig.height),img_arr.shape[1],img_arr.shape[0],flip_y=True) 
		r.setEnd((x,y))
		pass

	fig.window.push_handlers(on_idle)
	fig.window.push_handlers(atb.glumpy.Handlers(fig.window))
	fig.window.push_handlers(on_draw)
	fig.window.push_handlers(on_mouse_press,on_mouse_drag)
	fig.window.push_handlers(on_close)	
	fig.window.set_title("Eye")
	fig.window.set_position(650,0)    
	glumpy.show()   




