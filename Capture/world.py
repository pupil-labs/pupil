import os, sys

import glumpy 
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glumpy.atb as atb
from ctypes import *
import numpy as np

import cv2
import cv2.cv as cv
#from cv2 import VideoWriter
#from cv2.cv import CV_FOURCC as codec

from methods import normalize, denormalize, chessboard, circle_grid, gen_pattern_grid, calibrate_camera
from calibrate import *
from gl_shapes import Point

from multiprocessing import Queue, Value


class Bar(atb.Bar):
	"""docstring for Bar"""
	def __init__(self, name, defs):
		super(Bar, self).__init__(name,**defs) 
		self.fps = 0.0 
		self.exit = c_bool(0)
		self.pattern = c_bool(0)
		self.screen_shot = False
		self.calibration_images = False

		self.calibrate = c_bool(0)
		self.calib_running = c_bool(0)
		self.record_video = c_bool(0)
		self.record_running = c_bool(0)

		self.add_var("FPS", step=0.01, getter=self.get_fps)
		self.add_var("Find Calibration Pattern", self.pattern, key="P", help="Find Calibration Pattern")
		self.add_button("Screen Shot", self.screen_cap, key="SPACE", help="Capture A Frame")
		self.add_var("Calibrate", self.calibrate, key="C", help="Start/Stop Calibration Process")
		self.add_var("Record Video", self.record_video, key="R", help="Start/Stop Recording")
		self.add_var("Exit", self.exit)

	def update_fps(self, dt):
		temp_fps = 1/dt
		self.fps += 0.1*(temp_fps-self.fps)

	def get_fps(self):
		return self.fps

	def screen_cap(self):
		# just flip the switch
		self.screen_shot = not self.screen_shot

class Temp(object):
	"""Temp class to make objects"""
	def __init__(self):
		pass

def world(q, pupil_x, pupil_y, pipe,audio_record):
	"""world
		- Initialize glumpy figure, image, atb controls
		- Execute glumpy main loop
	"""
	# Get image array from queue, initialize glumpy, map img_arr to opengl texture 
	img_shape = q.get() # first item of que is img shape not needed in the world routine as the image is always three channels
	img_arr = q.get()
	fig = glumpy.figure((img_arr.shape[1], img_arr.shape[0]))

	image = glumpy.Image(img_arr)
	image.x, image.y = 0,0

	# global pool
	g_pool = Temp()
	g_pool.pupil_x = pupil_x
	g_pool.pupil_y = pupil_y
	g_pool.audio_record = audio_record

	# pattern object
	pattern = Temp()
	pattern.board = None
	pattern.norm_coords = (0,0)
	pattern.image_coords = (0,0)
	pattern.screen_coords = (0,0)
	pattern.obj_grid = gen_pattern_grid((4,11)) # calib grid
	pattern.obj_points = []
	pattern.img_points = []

	# gaze object
	gaze = Temp()
	gaze.norm_coords = (0,0)
	gaze.map_coords = (0,0)
	gaze.screen_coords = (0,0)
	gaze.coefs = None
	gaze.pt_cloud = None

	# record object
	record = Temp()
	record.writer = None
	record.path_parent = os.path.dirname( os.path.abspath(sys.argv[0]))
	record.path = None
	record.position_list = None
	record.counter = 0

	# initialize gl shape primitives
	pattern_point = Point(color=(0,255,0,0.5)) 
	gaze_point = Point(color=(255,0,0,0.5))

	# Initialize ant tweak bar inherits from atb.Bar (see Bar class)
	atb.init()
	bar = Bar("World", dict(label="Controls",
			help="Scene controls", color=(50,50,50), alpha=50,
			text='light', position=(10, 10), size=(200, 250)) )



	def draw():
		if pattern.board:
			pattern_point.draw()
		gaze_point.draw()


	def on_draw():
		fig.clear(0.0, 0.0, 0.0, 1.0)
		image.draw(x=image.x, y=image.y, z=0.0, 
					width=fig.width, height=fig.height)
		draw()

	def on_idle(dt):
		bar.update_fps(dt)
		img = q.get()

		img_arr[...] = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

		gaze.norm_coords = g_pool.pupil_x.value, g_pool.pupil_y.value 
		gaze.map_coords = map_vector(gaze.norm_coords, gaze.coefs)
		gaze.screen_coords = denormalize(gaze.map_coords, fig.width, fig.height)
		gaze_point.update(gaze.screen_coords)

		if bar.pattern:
			#pattern.board = chessboard(img)
			pattern.board = circle_grid(img)

		if pattern.board:
			# numpy array wants (row,col) for an image this = (height,width)
			# therefore: img.shape[1] = xval, img.shape[0] = yval
			pattern.image_coords = pattern.board[0] # this is the mean of the pattern found
			pattern.norm_coords = normalize(pattern.image_coords, img.shape[1], img.shape[0])
			pattern.screen_coords = denormalize(pattern.norm_coords, fig.width, fig.height)
			pattern.map_coords = pattern.screen_coords
			pattern_point.update(pattern.screen_coords)

		if bar.screen_shot and bar.pattern and pattern.board:
			# calibrate the camera intrinsics if the board is found
			# append list of circle grid center points to pattern.img_points
			# append generic list of circle grid pattern type to  pattern.obj_points
			pattern.img_points.append(pattern.board[1])
			pattern.obj_points.append(pattern.obj_grid)
			print "Number of Images Captured:", len(pattern.img_points)
			#if pattern.img_points.shape[0] > 10:
			if len(pattern.img_points) > 10:
				bar.calibration_images = True

			bar.screen_shot = False

		if (not bar.pattern) and bar.calibration_images:
			camera_matrix = calibrate_camera(	np.asarray(pattern.img_points), 
												np.asarray(pattern.obj_points), 
												(img.shape[1], img.shape[0])	)
			np.save("data/camera_matrix.npy", camera_matrix)

			bar.calibration_images = False

		if bar.calibrate and not bar.calib_running:
			bar.calib_running = 1
			gaze.pt_cloud = [] 
			gaze.coefs = None

		if bar.calibrate and bar.calib_running and pattern.board:
			gaze.pt_cloud.append([gaze.norm_coords[0],gaze.norm_coords[1],
								pattern.norm_coords[0], pattern.norm_coords[1] ])

		if not bar.calibrate and bar.calib_running:
			bar.calib_running = 0
			if gaze.pt_cloud:
				gaze.coefs = calibrate_poly(gaze.pt_cloud)



		if bar.record_video and not bar.record_running:
			record.path = os.path.join(record.path_parent, "data%03d/" %record.counter)
			while True: 
				try:
					os.mkdir(record.path)
					break
				except:
					print "dont want to overwrite data, incrementing counter & trying to make new data folder"
					record.counter +=1
					record.path = os.path.join(record.path_parent, "data%03d/" %record.counter)

			#video
			video_path = os.path.join(record.path,"world.avi")
			#FFV1 -- good speed lossless big file
			#DIVX -- good speed good compression medium file
			record.writer = cv2.VideoWriter(video_path,cv.CV_FOURCC(*'DIVX'),bar.fps, (img.shape[1],img.shape[0]) )
			
			#audio
			audio_path = os.path.join(record.path, "world.wav")
			g_pool.audio_record.value = 1
			pipe.send(audio_path)

			#positions
			record.position_list = []
			bar.record_running = 1


		if bar.record_video and bar.record_running:
			record.writer.write(img)
			record.position_list.append([gaze.map_coords[0], gaze.map_coords[1], dt])

		if not bar.record_video and bar.record_running:
			g_pool.audio_record.value = 0
			positions_path = os.path.join(record.path,"pupil_positions.npy")
			np.save(positions_path, np.asarray(record.position_list))
			record.writer = None
			bar.record_running = 0



		image.update()
		fig.redraw()
		if bar.exit:
			pass
			#fig.window.stop()


	fig.window.push_handlers(on_idle)
	fig.window.push_handlers(atb.glumpy.Handlers(fig.window))
	fig.window.push_handlers(on_draw)	
	fig.window.set_title("World")
	fig.window.set_position(0,0)	
	glumpy.show() 	
