import glumpy 
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glumpy.atb as atb
from ctypes import *
import numpy as np

from methods import *
from calibrate import *
from gl_shapes import Point, Ellipse

from multiprocessing import Queue, Value

class Bar(atb.Bar):
	"""docstring for Bar"""
	def __init__(self, name, defs):
		super(Bar, self).__init__(name,**defs) 
		self.fps = 0.0 
		self.display = 1
		self.exit = c_bool(0)
		self.spec_lower = 125.0
		self.spec_upper = 255.0
		self.bin_lower = 90.0
		self.bin_upper = 64.0
		self.pupil_point = c_bool(1)

		self.add_var("FPS", step=0.01, getter=self.get_fps)
		self.add_var("Display", step=1, getter=self.get_display, setter=self.set_display,
					max=4, min=0)
		self.add_var("Specular/S_Lower", step=1.0, getter=self.get_spec_lower, setter=self.set_spec_lower,
					max=256, min=0)
		self.add_var("Specular/S_Upper", step=1.0, getter=self.get_spec_upper, setter=self.set_spec_upper,
					max=256, min=0)
		self.add_var("Binary/B_Lower", step=1.0, getter=self.get_bin_lower, setter=self.set_bin_lower,
					max=256, min=0)
		self.add_var("Binary/B_Upper", step=1.0, getter=self.get_bin_upper, setter=self.set_bin_upper,
					max=256, min=0)
		self.add_var("Exit", self.exit)

	def update_fps(self, dt):
		temp_fps = 1/dt
		self.fps += 0.1*(temp_fps-self.fps)

	def get_fps(self):
		return self.fps

	def get_display(self):
		return self.display
	def set_display(self, val):
		self.display = val

	def get_spec_lower(self):
		return self.spec_lower
	def set_spec_lower(self, val):
		self.spec_lower = val

	def get_spec_upper(self):
		return self.spec_upper
	def set_spec_upper(self, val):
		self.spec_upper = val

	def get_bin_lower(self):
		return self.bin_lower
	def set_bin_lower(self, val):
		self.bin_lower = val

	def get_bin_upper(self):
		return self.bin_upper
	def set_bin_upper(self, val):
		self.bin_upper = val

class Temp(object):
	"""Temp class to make objects"""
	def __init__(self):
		pass


def eye(q, pupil_x, pupil_y, 
		pattern_x, pattern_y, 
		calibrate, pos_record, frame_count_record, 
		eye_pipe):
	"""eye
		- Initialize glumpy figure, image, atb controls
		- Execute the glumpy main glut loop
	"""
	# Get image array from queue, initialize glumpy, map img_arr to opengl texture 
	img_params = Temp()
	img_params.shape = q.get()
	img_arr = q.get()
	img_arr.shape = img_params.shape

	if len(img_arr.shape) <3:
		img_arr = np.dstack((img_arr,img_arr,img_arr))

	fig = glumpy.figure((img_arr.shape[1], img_arr.shape[0]) )
	image = glumpy.Image(img_arr)
	image.x, image.y = 0,0

	# global pool
	g_pool = Temp()
	g_pool.pupil_x = pupil_x
	g_pool.pupil_y = pupil_y
	g_pool.pattern_x = pattern_x
	g_pool.pattern_y = pattern_y
	g_pool.calibrate = calibrate
	g_pool.pos_record = pos_record
	g_pool.frame_count_record = frame_count_record

	# pupil object
	pupil = Temp()
	pupil.norm_coords = (0,0)
	pupil.image_coords = (0,0)
	pupil.screen_coords = (0,0)
	pupil.ellipse = None
	pupil.map_coords = (0,0)
	pupil.coefs = None
	pupil.pt_cloud = None

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
	bar = Bar("Eye", dict(label="Controls",
			help="Scene controls", color=(50,50,50), alpha=50,
			text='light', position=(10, 10), size=(200, 200)) )

	def draw():
		"""draw
			- place to draw objects to the glumpy window
		"""
		if bar.pupil_point and pupil.ellipse:
			pupil_point.draw()
			pupil_ellipse.draw()

	def on_draw():
		fig.clear(0.0, 0.0, 0.0, 1.0)
		image.draw(x=image.x, y=image.y, z=0.0, 
					width=fig.width, height=fig.height)
		draw()


	def on_idle(dt):
		bar.update_fps(dt)
		img = q.get()
		img.shape = img_params.shape
			
		if len(img.shape) <3:
			gray_img = img
			img = np.dstack((img,img,img))
		else:
			gray_img = grayscale(img)

		gray_img = add_horizontal_gradient(gray_img)
		gray_img = 	add_vertical_gradient(gray_img)

		spec_img = erase_specular(gray_img, bar.spec_lower, bar.spec_upper)
		# spec_img = equalize(spec_img)		

		# binary_img = adaptive_threshold(spec_img, bar.bin_lower, bar.bin_upper)
		binary_img = extract_darkspot(spec_img, bar.bin_lower, bar.bin_upper)
		pupil.ellipse = fit_ellipse(binary_img)
		


		if pupil.ellipse:
			pupil.image_coords = pupil.ellipse['center']
			# numpy array wants (row,col) for an image this = (height,width)
			# therefore: img.shape[1] = xval, img.shape[0] = yval
			pupil.norm_coords = normalize(pupil.image_coords, img.shape[1], img.shape[0])
			pupil.screen_coords = denormalize(pupil.norm_coords, fig.width, fig.height)
			pupil_point.update(pupil.screen_coords)
			pupil_ellipse.update(pupil.screen_coords, pupil.ellipse)

			# for the world screen
			pupil.map_coords = map_vector(pupil.norm_coords, pupil.coefs)
			g_pool.pupil_x.value, g_pool.pupil_y.value = pupil.map_coords


		if bar.display == 0:
			img_arr[...] = img
		elif bar.display == 1:
			img_arr[...] =  np.dstack((gray_img,gray_img,gray_img))
		elif bar.display == 2:
			img_arr[...] = np.dstack((spec_img, spec_img, spec_img))
		elif bar.display == 3:
			img_arr[...] = np.dstack((binary_img, binary_img, binary_img))
		else:
			gray_img = cv2.Scharr(gray_img, -1, 1, 1)
			img_arr[...] = np.dstack((gray_img,gray_img,gray_img))

	
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
		if bar.exit:
			pass
			#fig.window.stop()

	fig.window.push_handlers(on_idle)
	fig.window.push_handlers(atb.glumpy.Handlers(fig.window))
	fig.window.push_handlers(on_draw)
	fig.window.set_title("Eye")
	fig.window.set_position(1280,0)	
	glumpy.show() 	




