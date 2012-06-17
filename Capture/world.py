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
		self.calibrate_nine = c_bool(0)
		self.calibrate_nine_step = c_int(0)
		self.calibrate_nine_stage = c_int(0)
		self.calib_running = c_bool(0)
		self.record_video = c_bool(0)
		self.record_running = c_bool(0)
		self.play = c_bool(0)
		# play and record are tied together via pointers to the objects
		self.play = self.record_video

		self.add_var("FPS", step=0.01, getter=self.get_fps)
		self.add_var("Find Calibration Pattern", self.pattern, key="P", help="Find Calibration Pattern")
		self.add_button("Screen Shot", self.screen_cap, key="SPACE", help="Capture A Frame")
		self.add_var("Calibrate", self.calibrate, key="C", help="Start/Stop Calibration Process")
		self.add_var("Nine_Pt", self.calibrate_nine, key="9", help="Start/Stop 9 Point Calibration Process")
		self.add_var("Nine_Pt_Stage", self.calibrate_nine_stage)
		self.add_var("Nine_Pt_Step", self.calibrate_nine_step)
		self.add_var("Record Video", self.record_video, key="R", help="Start/Stop Recording")
		self.add_var("Play Source Video", self.play)
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

def world(q, pupil_x, pupil_y, 
			pattern_x, pattern_y, 
			calibrate, pos_record, 
			audio_pipe, eye_pipe, audio_record,
			player_pipe):
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
	g_pool.pattern_x = pattern_x
	g_pool.pattern_y = pattern_y
	g_pool.audio_record = audio_record
	g_pool.calibrate = calibrate
	g_pool.pos_record = pos_record

	# pattern object
	pattern = Temp()
	pattern.board = None
	pattern.norm_coords = (0,0)
	pattern.image_coords = (0,0)
	pattern.screen_coords = (0,0)
	pattern.obj_grid = gen_pattern_grid((4,11)) # calib grid
	pattern.obj_points = []
	pattern.img_points = []
	pattern.map = (0,2,7,16,21,23,39,40,42)


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

	# player object
	player = Temp()
	player.play_list = glob('src_video/*')
	player.playlist = [os.path.join(record.path_parent, path) for path in player.play_list]
	player.current_video = 0
	player.play_list_len = len(player.play_list)
	player.playing = False

	# initialize gl shape primitives
	pattern_point = Point(color=(0,255,0,0.5)) 
	gaze_point = Point(color=(255,0,0,0.5))

	# Initialize ant tweak bar inherits from atb.Bar (see Bar class)
	atb.init()
	bar = Bar("World", dict(label="Controls",
			help="Scene controls", color=(50,50,50), alpha=50,
			text='light', position=(10, 10), size=(200, 250)) )



	def draw():
		if pattern.board is not None:
			pattern_point.draw()
		gaze_point.draw()


	def on_draw():
		fig.clear(0.0, 0.0, 0.0, 1.0)
		image.draw(x=image.x, y=image.y, z=0.0, 
					width=fig.width, height=fig.height)
		draw()

	def on_idle(dt):
		bar.update_fps(dt)


		# Nine Point calibration state machine timing
		if bar.calibrate_nine.value:
			if bar.calibrate_nine_step.value >= 40:
				bar.calibrate_nine_step.value = 0
				bar.calibrate_nine_stage.value += 1

			if bar.calibrate_nine_stage.value > 8:
				bar.calibrate_nine_stage.value = 0 
				bar.calibrate_nine.value = 0

			if bar.calibrate_nine_step.value in range(5,40):
				bar.calibrate.value = True

			else:
				bar.calibrate.value = False

			player_pipe.send("calibrate")
			circle_id = pattern.map[bar.calibrate_nine_stage.value]

			player_pipe.send((circle_id, bar.calibrate_nine_step.value))
			
			bar.calibrate_nine_step.value += 1

		# Broadcast local calibration state to global pool of variables
		g_pool.calibrate.value = bar.calibrate.value

		# get an image from the grabber
		img = q.get()

		# update the image to display and convert color space
		img_arr[...] = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

		# update gaze points from shared variable pool
		gaze.map_coords = g_pool.pupil_x.value, g_pool.pupil_y.value
		gaze.screen_coords = denormalize(gaze.map_coords, fig.width, fig.height)
		gaze_point.update(gaze.screen_coords)

		if bar.pattern:
			#pattern.board = chessboard(img)
			pattern.board = circle_grid(img)

		if bar.pattern and bar.calibrate_nine.value:
			pattern.board = circle_grid(img, pattern.map[bar.calibrate_nine_stage.value])

		if pattern.board is not None:
			# numpy array wants (row,col) for an image this = (height,width)
			# therefore: img.shape[1] = xval, img.shape[0] = yval
			pattern.image_coords = pattern.board # this is the mean of the pattern found
			pattern.norm_coords = normalize(pattern.image_coords, img.shape[1], img.shape[0])
			pattern.screen_coords = denormalize(pattern.norm_coords, fig.width, fig.height)
			pattern.map_coords = pattern.screen_coords
			pattern_point.update(pattern.screen_coords)

			# broadcast pattern.norm_coords for calibration in eye process
			g_pool.pattern_x.value, g_pool.pattern_y.value = pattern.norm_coords
		else:
			# If no pattern detected send 0,0 -- check this condition in eye process
			g_pool.pattern_x.value, g_pool.pattern_y.value = 0, 0

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
			record.writer = cv2.VideoWriter(video_path,cv.CV_FOURCC(*'DIVX'),bar.fps, (img.shape[1],img.shape[0]) )
			
			# audio data to audio process
			audio_path = os.path.join(record.path, "world.wav")
			g_pool.audio_record.value = 1
			audio_pipe.send(audio_path)

			# positions data to eye process
			positions_path = os.path.join(record.path,"pupil_positions.npy")
			g_pool.pos_record.value = 1
			eye_pipe.send(positions_path)

			bar.record_running = 1

		# While Recording...
		if bar.record_video and bar.record_running:
			# Save image frames to video writer
			record.writer.write(img)

		# Finish all recordings, clean up. 
		if not bar.record_video and bar.record_running:
			g_pool.audio_record.value = 0
			g_pool.pos_record.value = 0
			record.writer = None
			bar.record_running = 0


		if bar.play.value and not player.playing:
			player_pipe.send('load_video')
			player_pipe.send(player.play_list[player.current_video])
			player.playing = True

		if player.playing: 
			player_pipe.send('next_frame')
			status = player_pipe.recv()

			if status:
				pass
			else:
				bar.play.value = 0

		if player.playing and not bar.play.value:
			player.playing = False
			player.current_video += 1
			if player.current_video >= player.play_list_len:
				player.current_video = 0


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
