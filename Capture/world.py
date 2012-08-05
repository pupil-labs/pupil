import os, sys

import glumpy 
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glumpy.atb as atb
from ctypes import *
import numpy as np

from glob import glob

import cv2
import cv2.cv as cv

from methods import normalize, denormalize, chessboard, circle_grid, gen_pattern_grid, calibrate_camera
from methods import Temp,capture

from calibrate import *
from gl_shapes import Point

from multiprocessing import Queue, Value


class Bar(atb.Bar):
	"""docstring for Bar"""
	def __init__(self, name,g_pool, defs):
		super(Bar, self).__init__(name,**defs) 
		self.fps = c_float(0.0) 
		self.calibrate = g_pool.calibrate
		self.find_pattern = c_bool(0)
		self.screen_shot = False
		self.calibration_images = False
		self.calibrate_nine = g_pool.cal9
		self.calibrate_nine_step = g_pool.cal9_step
		self.calibrate_nine_stage =  g_pool.cal9_stage
		self.calib_running = g_pool.calibrate
		self.record_video = c_bool(0)
		self.record_running = c_bool(0)
		self.play = g_pool.play
		# play and record are tied together via pointers to the objects
		# self.play = self.record_video

		self.add_var("FPS",self.fps, step=1., readonly=True)
		self.add_var("Find Calibration Pattern", self.find_pattern, key="P", help="Find Calibration Pattern")
		self.add_button("Screen Shot", self.screen_cap, key="SPACE", help="Capture A Frame")
		self.add_var("Calibrate", self.calibrate, key="C", help="Start/Stop Calibration Process")
		self.add_var("Nine_Pt", self.calibrate_nine, key="9", help="Start/Stop 9 Point Calibration Process")
		self.add_var("Record Video", self.record_video, key="R", help="Start/Stop Recording")
		self.add_var("Play Source Video", self.play)
		self.add_var("Exit", g_pool.quit)

	def update_fps(self, dt):
		self.fps.value += .1*(1/dt-self.fps.value)


	def screen_cap(self):
		self.find_pattern.value = True
		self.screen_shot = True


class Temp(object):
	"""Temp class to make objects"""
	def __init__(self):
		pass

def world(src, g_pool):
	"""world
		- Initialize glumpy figure, image, atb controls
		- Execute glumpy main loop
	"""
	cap = capture(src,(640,480))
	s, img_arr = cap.read_RGB()
	fig = glumpy.figure((img_arr.shape[1], img_arr.shape[0]))

	image = glumpy.Image(img_arr)
	image.x, image.y = 0,0


	# pattern object
	pattern = Temp()
	pattern.centers = None
	pattern.norm_coords = (0,0)
	pattern.image_coords = (0,0)
	pattern.screen_coords = (0,0)
	pattern.obj_grid = gen_pattern_grid((4,11)) # calib grid
	pattern.obj_points = []
	pattern.img_points = []
	pattern.map = (0,2,7,16,21,23,39,40,42)
	pattern.board_centers = None


	# gaze object
	gaze = Temp()
	gaze.map_coords = (0,0)
	gaze.screen_coords = (0,0)

	# record object
	record = Temp()
	record.writer = None
	record.path_parent = os.path.dirname( os.path.abspath(sys.argv[0]))
	record.path = None
	record.counter = 0



	# initialize gl shape primitives
	pattern_point = Point(color=(0,255,0,0.5)) 
	gaze_point = Point(color=(255,0,0,0.5))

	# Initialize ant tweak bar inherits from atb.Bar (see Bar class)
	atb.init()
	bar = Bar("World",g_pool, dict(label="Controls",
			help="Scene controls", color=(50,50,50), alpha=50,
			text='light',refresh=.1, position=(10, 10), size=(200, 250)) )





	def on_draw():
		fig.clear(0.0, 0.0, 0.0, 1.0)
		image.draw(x=image.x, y=image.y, z=0.0, 
					width=fig.width, height=fig.height)
		if pattern.centers is not None:
			pattern_point.draw()
		gaze_point.draw()



	def on_idle(dt):
		bar.update_fps(dt)

		# Nine Point calibration state machine timing
		if bar.calibrate_nine.value:
			bar.calibrate.value = True
			bar.find_pattern.value = False
			if bar.calibrate_nine_step.value > 30:
				bar.calibrate_nine_step.value = 0
				bar.calibrate_nine_stage.value += 1
			if bar.calibrate_nine_stage.value > 8:
				bar.calibrate_nine_stage.value = 0 
				bar.calibrate.value = False
				bar.calibrate_nine.value = False
				bar.find_pattern.value = False
			if bar.calibrate_nine_step.value in range(10,25):
				bar.find_pattern.value = True
			
			g_pool.cal9_circle_id.value = pattern.map[bar.calibrate_nine_stage.value]
			bar.calibrate_nine_step.value += 1
		g_pool.player_refresh.set()

		# get an image from the grabber
		s,img = cap.read_RGB()

		# update the image to display
		img_arr[...] = img

		# update gaze points from shared variable pool
		gaze.screen_coords = denormalize((g_pool.pupil_x.value, g_pool.pupil_y.value), fig.width, fig.height)
		gaze_point.update(gaze.screen_coords)

		#pattern detection and its various uses
		pattern.centers = None
		if bar.find_pattern.value:
			pattern.centers = circle_grid(img)
		
		if pattern.centers is not None:
			if bar.calibrate_nine.value:
				pattern.image_coords = pattern.centers[circle_id][0]
			else:
				mean = pattern.centers.sum(0)/pattern.centers.shape[0]
				pattern.image_coords = mean[0]
			pattern.norm_coords = normalize(pattern.image_coords, img.shape[1], img.shape[0])
			pattern.screen_coords = denormalize(pattern.norm_coords, fig.width, fig.height)
			pattern_point.update(pattern.screen_coords)
			g_pool.pattern_x.value, g_pool.pattern_y.value = pattern.norm_coords
		else:
			# If no pattern detected send 0,0 -- check this condition in eye process
			g_pool.pattern_x.value, g_pool.pattern_y.value = 0, 0


		#gather pattern centers and find cam intrisics
		if bar.screen_shot and pattern.centers is not None:
			bar.screen_shot = False
			# calibrate the camera intrinsics if the board is found
			# append list of circle grid center points to pattern.img_points
			# append generic list of circle grid pattern type to  pattern.obj_points
			pattern.img_points.append(pattern.centers)
			pattern.obj_points.append(pattern.obj_grid)
			print "Number of Patterns Captured:", len(pattern.img_points)
			#if pattern.img_points.shape[0] > 10:
			if len(pattern.img_points) > 10:
				camera_matrix, dist_coefs = calibrate_camera(np.asarray(pattern.img_points), 
													np.asarray(pattern.obj_points), 
													(img.shape[1], img.shape[0]))
				np.save("camera_matrix.npy", camera_matrix)
				np.save("dist_coefs.npy", dist_coefs)
				pattern.img_points = []
				bar.find_pattern.value = False
		

		# Setup recording process
		if bar.record_video and not bar.record_running:
			record.path = os.path.join(record.path_parent, "data%03d/" %record.counter)
			while True: 
				try:
					os.mkdir(record.path)
					break
				except:
					print "We dont want to overwrite data, incrementing counter & trying to make new data folder"
					record.counter +=1
					record.path = os.path.join(record.path_parent, "data%03d/" %record.counter)

			#video
			video_path = os.path.join(record.path,"world.avi")
			#FFV1 -- good speed lossless big file
			#DIVX -- good speed good compression medium file
			record.writer = cv2.VideoWriter(video_path,cv.CV_FOURCC(*'DIVX'),bar.fps.value, (img.shape[1],img.shape[0]) )
			
			# audio data to audio process
			audio_path = os.path.join(record.path, "world.wav")
			try:
				g_pool.audio_record.value = 1
				g_pool.audio_tx.send(audio_path)
			except:
				print "no audio initialized"

			# positions data to eye process
			positions_path = os.path.join(record.path,"pupil_positions.npy")
			g_pool.pos_record.value = True
			g_pool.eye_tx.send(positions_path)

			bar.record_running = 1
			g_pool.frame_count_record.value = 0

		# While Recording...
		if bar.record_video and bar.record_running:
			# Save image frames to video writer
			# increment the frame_count_record value 
			# Eye positions can be associated with frames of recording even if different framerates 
			g_pool.frame_count_record.value += 1
			record.writer.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

		# Finish all recordings, clean up. 
		if not bar.record_video and bar.record_running:
			try:
				g_pool.audio_record.value = 0
			except:
				print "no audio recorded"
			
			g_pool.pos_record.value = 0
			record.writer = None
			bar.record_running = 0

		image.update()
		fig.redraw()
		
		if g_pool.quit.value:
			print "WORLD Process closing from global or atb"
			fig.window.stop()

	def on_close():
		g_pool.quit.value = True
		print "WORLD Process closed from window"			

	fig.window.push_handlers(on_idle)
	fig.window.push_handlers(atb.glumpy.Handlers(fig.window))
	fig.window.push_handlers(on_draw)	
	fig.window.push_handlers(on_close)	
	fig.window.set_title("World")
	fig.window.set_position(0,0)	
	glumpy.show() 	
