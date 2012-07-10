import os, sys

import glumpy 
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glumpy.atb as atb
from ctypes import *
import numpy as np

import cv2
import cv2.cv as cv

from methods import *
from calibrate import *
from gl_shapes import Point, Ellipse

from multiprocessing import Queue, Value

class Bar(atb.Bar):
	"""docstring for Bar"""
	def __init__(self, name, defs):
		super(Bar, self).__init__(name,**defs) 
		self.fps = 0.0 
		self.record_video = c_bool(0)
		self.record_running = c_bool(0)

		self.add_var("FPS", step=0.01, getter=self.get_fps)
		self.add_var("Record Video", self.record_video, key="SPACE", help="Start/Stop Recording")


	def update_fps(self, dt):
		temp_fps = 1/dt
		self.fps += 0.1*(temp_fps-self.fps)

	def get_fps(self):
		return self.fps


class Temp(object):
	"""Temp class to make objects"""
	def __init__(self):
		pass


def eye(q):
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

	# record object
	record = Temp()
	record.writer = None
	record.path_parent = os.path.dirname( os.path.abspath(sys.argv[0]))
	record.path = None
	record.counter = 0


	# Initialize ant tweak bar inherits from atb.Bar (see Bar class)
	atb.init()
	bar = Bar("Eye", dict(label="Controls",
			help="Scene controls", color=(50,50,50), alpha=50,
			text='light', position=(10, 10), size=(200, 200)) )

	def on_draw():
		fig.clear(0.0, 0.0, 0.0, 1.0)
		image.draw(x=image.x, y=image.y, z=0.0, 
					width=fig.width, height=fig.height)

	def on_idle(dt):
		bar.update_fps(dt)
		img = q.get()
		img.shape = img_params.shape
		img_cv = img

			
		if len(img.shape) <3:
			img = np.dstack((img,img,img))

		img_arr[...] = img

		# setup recording
		if bar.record_video and not bar.record_running:
			record.path = os.path.join(record.path_parent, "eye%03d/" %record.counter)
			while True: 
				try:
					os.mkdir(record.path)
					break
				except:
					print "We dont want to overwrite data, incrementing counter & trying to make new eye folder"
					record.counter +=1
					record.path = os.path.join(record.path_parent, "eye%03d/" %record.counter)

			#video
			video_path = os.path.join(record.path,"eye.avi")
			#FFV1 -- good speed lossless big file
			#DIVX -- good speed good compression medium file
			record.writer = cv2.VideoWriter(video_path,cv.CV_FOURCC(*'DIVX'),bar.fps, (img.shape[1],img.shape[0]),0)
					
			bar.record_running = 1

		if bar.record_video and bar.record_running:
			record.writer.write(img_cv)

		# Finish all recordings, clean up. 
		if not bar.record_video and bar.record_running:
			record.writer = None
			bar.record_running = 0
			print "Wrote video to: ", record.path

		image.update()
		fig.redraw()


	fig.window.push_handlers(on_idle)
	fig.window.push_handlers(atb.glumpy.Handlers(fig.window))
	fig.window.push_handlers(on_draw)
	fig.window.set_title("Eye")
	fig.window.set_position(0,0)	
	glumpy.show() 	




