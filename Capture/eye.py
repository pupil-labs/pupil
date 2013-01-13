import glumpy
import OpenGL.GL as gl
import glumpy.atb as atb
from ctypes import c_int,c_bool,c_float
import cPickle as pickle
import numpy as np
from time import sleep
from methods import *
from calibrate import *
from gl_shapes import Point, Ellipse
from methods import Temp,capture
from os import path
class Bar(atb.Bar):
	"""docstring for Bar"""
	def __init__(self, name,g_pool, bar_defs):
		super(Bar, self).__init__(name,**bar_defs)
		self.fps = c_float(0.0)
		self.sleep = c_float(0.0)
		self.display = c_int(1)
		self.draw_pupil = c_bool(1)
		self.draw_roi = c_bool(0)
		self.bin_thresh = c_int(60)
		self.blur = c_int(3)
		self.pupil_ratio = c_float(.6)
		self.pupil_angle = c_float(0.0)
		self.pupil_min_size = c_float(50.)
		self.pupil_max_size = c_float(100.)
		self.pupil_size = c_float(80.)
		self.pupil_size_tolerance = c_float(40.)
		self.canny_apture = c_int(7)
		self.canny_thresh = c_int(200)
		self.canny_ratio = c_int(2)

		#add class field here and it will become session persistant
		self.session_save = {'display':self.display,
							'draw_pupil':self.draw_pupil,
							'bin_thresh':self.bin_thresh,
							'pupil_ratio':self.pupil_ratio,
							'pupil_size':self.pupil_size,
							'pupil_min_size':self.pupil_min_size,
							'pupil_max_size':self.pupil_max_size,
							'mean_blur':self.blur,
							'canny_apture':self.canny_apture,
							'canny_thresh':self.canny_thresh,
							'canny_ratio':self.canny_ratio}

		self.load()

		self.add_var("Display/FPS",self.fps, step=1.,readonly=True)
		self.add_var("Display/SlowDown",self.sleep, step=0.01,min=0.0)
		self.add_var("Display/Mode", self.display, step=1,max=3, min=0, help="select the view-mode")
		self.add_var("Display/Show_Pupil_Point", self.draw_pupil)
		self.add_button("Draw_ROI", self.roi, help="drag on screen to select a region of interest", Group="Display")
		self.add_var("Darkspot/Threshold", self.bin_thresh, step=1, max=256, min=0)
		self.add_var("Pupil/Ratio", self.pupil_ratio, step=.05, max=1., min=0.)
		# self.add_var("Pupil/Angle", self.pupil_angle,step=1.0,readonly=True)
		self.add_var("Pupil/Min_Size", self.pupil_min_size, step=1, min=0)
		self.add_var("Pupil/Max_Size", self.pupil_max_size, step=1, min=0)
		self.add_var("Pupil/Size", self.pupil_size, step=1, min=0)
		self.add_var("Pupil/Size_Tolerance", self.pupil_size_tolerance, step=1, min=0)
		self.add_var("Canny/MeanBlur", self.blur,step=2,max=7,min=1)
		self.add_var("Canny/Apture",self.canny_apture, step=2, max=7, min=3)
		self.add_var("Canny/Lower_Threshold", self.canny_thresh, step=1,min=1)
		self.add_var("Canny/LowerUpperRatio", self.canny_ratio, step=1,min=0,help="Canny recommended a ratio between 3/1 and 2/1")
		self.add_var("SaveSettings&Exit", g_pool.quit)



	def update_fps(self, dt):
		temp_fps = 1/dt
		self.fps.value += 0.1*(temp_fps-self.fps.value)


	def save(self):
		new_settings = dict([(key,field.value) for key, field in self.session_save.items()])
		settings_file = open('session_settings','wb')
		pickle.dump(new_settings,settings_file)
		settings_file.close

	def roi(self):
		self.draw_roi.value = 1


	def load(self):
		try:
			settings_file = open('session_settings','rb')
			new_settings = pickle.load(settings_file)
			settings_file.close
		except IOError:
			print "No session_settings file found. Using defaults"
			return

		for key,val in new_settings.items():
			try:
				self.session_save[key].value = val
			except KeyError:
				print "Warning the Sessions file is from a different version, not all fields may be updated"

class Roi(object):
	"""this is a simple 2D Region of Interest class
	it is applied on numpy arrays for convinient slicing
	like this:

	roi_array_slice = full_array[r.lY:r.uY,r.lX:r.uX]
	#do something with roi_array_slice
	full_array[r.lY:r.uY,r.lX:r.uX] = roi_array_slice

	this creates a view, no data coping done
	"""
	def __init__(self, array_shape):
		self.array_shape = array_shape
		self.lX = 0
		self.lY = 0
		self.uX = array_shape[1]-0
		self.uY = array_shape[0]-0
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
		"""
		adds the roi offset to a len2 vector
		"""
		return (self.lX+x,self.lY+y)


def eye(src,size,g_pool):
	"""eye
		- Initialize glumpy figure, image, atb controls
		- Execute the glumpy main glut loop
	"""
	#init capture, initialize glumpy, map img_arr to opengl texture
	cap = capture(src,size)
	if isinstance(cap.src, str): #if using a debug video
		cap.auto_rewind = True
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
	pupil.gaze_coords = (0,0)
	pupil.prev_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
	try:
		pupil.pt_cloud = list(np.load('cal_pt_cloud.npy'))  #LIST just makes it conform with \
															#how our pupil data is captured during calibration
	except:
		pupil.pt_cloud = None
	if pupil.pt_cloud is not None:
		pupil.coefs = calibrate_poly(pupil.pt_cloud)
	else:
		pupil.coefs = None

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

	atb.init()
	bar = Bar("Eye",g_pool, dict(label="Controls",
			help="Scene controls", color=(50,50,50), alpha=50,
			text='light', refresh=.2, position=(10, 10), size=(200, 300)) )



	def on_idle(dt):
		bar.update_fps(dt)

		s,img = cap.read_RGB()
 		sleep(bar.sleep.value)

		###IMAGE PROCESSING
		gray_img = grayscale(img[r.lY:r.uY,r.lX:r.uX])
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
		binary_img = bin_thresholding(gray_img,image_upper=bar.bin_thresh.value)
		cv2.dilate(binary_img, kernel,binary_img, iterations=3)
		spec_mask = bin_thresholding(gray_img, image_upper=250)
		cv2.erode(spec_mask, kernel,spec_mask, iterations=1)

		if bar.blur.value >1:
			gray_img = cv2.medianBlur(gray_img,bar.blur.value)

		edges =  cv2.Canny(gray_img,bar.canny_thresh.value, bar.canny_thresh.value*bar.canny_ratio.value,apertureSize= bar.canny_apture.value)
		# edges = dif_gaus(gray_img,20.,24.)
		edges = cv2.min(edges, spec_mask)
		edges = cv2.min(edges,binary_img)

		result = fit_ellipse(edges,binary_img,bar.bin_thresh.value, ratio=bar.pupil_ratio.value,target_size=bar.pupil_size.value,size_tolerance=bar.pupil_size_tolerance.value)

		overlay_b = cv2.max(edges,gray_img)
		overlay =cv2.cvtColor(overlay_b, cv2.COLOR_GRAY2RGB)
		overlay[:,:,2] = cv2.max(overlay_b,binary_img) #blue channel
		overlay[:,:,1] = cv2.min(overlay_b,spec_mask) #red channel

		if result is not None:
			#display some centers for debugging
			pupil.ellipse, others= result
			for pre,((x,y),axs,ang) in others:
				x,y = int(x),int(y)
				overlay[y,x,:]   = [0,255,0]
				# overlay[y,x+1,:] = [0,255,0]
				# overlay[y+1,x,:] = [0,255,0]
				# overlay[y+1,x+1,:]=[0,255,0]



		if bar.display.value == 0:
			img_arr[...] = img
		elif bar.display.value == 1:
			img_arr[r.lY:r.uY,r.lX:r.uX] = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
		elif bar.display.value == 2:
			if bar.draw_roi.value:
				overlay[0:r.uY-r.lY:2,0,:]= [255,255,255]
				overlay[0:r.uY-r.lY:2,r.uX-r.lX-1,:]= [255,255,255]
				overlay[0,0:r.uX-r.lX:2,:]= [255,255,255]
				overlay[r.uY-r.lY-1,0:r.uX-r.lX:2,:]= [255,255,255]
			img_arr[...] = img
			img_arr[r.lY:r.uY,r.lX:r.uX] = overlay

		elif bar.display.value == 3:
			t_img = np.zeros(gray_img.shape,dtype= gray_img.dtype)
			t_img += 125
			t_img +=  gray_img-pupil.prev_img
			img_arr[...] = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
			pupil.prev_img = gray_img
		else:
			pass
		if result is not None:
			pupil.image_coords = r.add_vector(pupil.ellipse['center'])
			#update pupil size,angle and ratio for the ellipse filter algorithm
			bar.pupil_size.value  = bar.pupil_size.value +  .5*(pupil.ellipse['major']-bar.pupil_size.value)
			bar.pupil_ratio.value = bar.pupil_ratio.value + .7*(pupil.ellipse['ratio']-bar.pupil_ratio.value)
			bar.pupil_angle.value = bar.pupil_angle.value + 1.*(pupil.ellipse['angle']-bar.pupil_angle.value)
			# if pupil found tighten the size tolerance
			bar.pupil_size_tolerance.value -=1
			bar.pupil_size_tolerance.value = max(10,bar.pupil_size_tolerance.value)

			bar.pupil_size.value  = max(bar.pupil_min_size.value,min(bar.pupil_max_size.value,bar.pupil_size.value))


			pupil.norm_coords = normalize(pupil.image_coords, img.shape[1], img.shape[0])# numpy array wants (row,col) for an image this = (height,width)
			pupil.screen_coords = denormalize(pupil.norm_coords, fig.width, fig.height)
			if bar.draw_pupil.value:
				pupil_point.update(pupil.screen_coords)
				pupil_ellipse.update(pupil.screen_coords, pupil.ellipse)

			# for the world screen
			pupil.gaze_coords = map_vector(pupil.norm_coords, pupil.coefs)
			g_pool.gaze_x.value, g_pool.gaze_y.value = pupil.gaze_coords
		else:
			pupil.ellipse = None
			# pupil.gaze_coords = None, None #whithout this line the last know pupil position is recorded if none is found
			# if pupil not found widen the size tolerance
			bar.pupil_size_tolerance.value +=1
			bar.pupil_size_tolerance.value = min(bar.pupil_size_tolerance.value,40)

		###CALIBRATION and MAPPING###
		# Initialize Calibration (setup variables and lists)
		if g_pool.calibrate.value and not l_pool.calib_running:
			l_pool.calib_running = True
			pupil.pt_cloud = []
			pupil.coefs = None

		# While Calibrating...
		if l_pool.calib_running and ((g_pool.pattern_x.value != 0) or (g_pool.pattern_y.value != 0)) and pupil.ellipse:
			pupil.pt_cloud.append([pupil.norm_coords[0],pupil.norm_coords[1],
								g_pool.pattern_x.value, g_pool.pattern_y.value])

		# Calculate coefs
		if not g_pool.calibrate.value and l_pool.calib_running:
			l_pool.calib_running = 0
			if pupil.pt_cloud:
				pupil.coefs = calibrate_poly(pupil.pt_cloud)
				np.save('cal_pt_cloud.npy', np.array(pupil.pt_cloud))



		###RECORDING###
		# Setup variables and lists for recording
		if g_pool.pos_record.value and not l_pool.record_running:
			l_pool.record_path = g_pool.eye_rx.recv()
			print "l_pool.record_path: ", l_pool.record_path
			l_pool.record_positions = []
			l_pool.record_running = True

		# While recording...
		if l_pool.record_running:
			l_pool.record_positions.append([pupil.gaze_coords[0], pupil.gaze_coords[1],pupil.norm_coords[0],pupil.norm_coords[1], dt, g_pool.frame_count_record.value])

		# Save values and flip switch to off for recording
		if not g_pool.pos_record.value and l_pool.record_running:
			positions_path = path.join(l_pool.record_path, "gaze_positions.npy")
			cal_pt_cloud_path = path.join(l_pool.record_path, "cal_pt_cloud.npy")
			np.save(positions_path, np.asarray(l_pool.record_positions))
			try:
				np.save(cal_pt_cloud_path, np.asarray(pupil.pt_cloud))
			except:
				print "Warning: No calibration data associated with this recording."
			l_pool.record_running = False


		image.update()
		fig.redraw()

		if g_pool.quit.value:
			print "EYE Process closing from global or atb"
			bar.save()
			fig.window.stop()


	def on_draw():
		fig.clear(0.0, 0.0, 0.0, 1.0)
		image.draw(x=image.x, y=image.y, z=0.0,
					width=fig.width, height=fig.height)

		if bar.draw_pupil and pupil.ellipse:
			pupil_point.draw()
			pupil_ellipse.draw()


	def on_close():
		bar.save()
		g_pool.quit.value = True
		print "EYE Process closing from window event"

	def on_mouse_press(x, y, button):
		x,y = denormalize(normalize((x,y),fig.width,fig.height),img_arr.shape[1],img_arr.shape[0],flip_y=True)
		if bar.draw_roi.value:
			r.setStart((x,y))

	def on_mouse_drag(x, y, dx, dy, buttons):
		x,y = denormalize(normalize((x,y),fig.width,fig.height),img_arr.shape[1],img_arr.shape[0],flip_y=True)
		x,y = max(0,min(fig.width,x)),max(0,min(fig.height,y))
		if bar.draw_roi.value:
			r.setEnd((x,y))

	def on_mouse_release(x, y, buttons):
		bar.draw_roi.value = 0

	fig.window.push_handlers(on_idle)
	fig.window.push_handlers(atb.glumpy.Handlers(fig.window))
	fig.window.push_handlers(on_draw)
	fig.window.push_handlers(on_mouse_press,on_mouse_drag,on_mouse_release)
	fig.window.push_handlers(on_close)
	fig.window.set_title("Eye")
	fig.window.set_position(650,0)
	glumpy.show()




