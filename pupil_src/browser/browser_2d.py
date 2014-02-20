'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os, sys, argparse

import glumpy
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glumpy.atb as atb
from ctypes import *
import numpy as np
from time import sleep
import cv2

# from gl_shapes import Point
from multiprocessing import Pipe, Value
from methods import denormalize, normalize, flip_horizontal
from gl_shapes import Point

from time import sleep
from video_homography import homography_map, undistort_point


class Bar(atb.Bar):
	"""docstring for Bar"""
	def __init__(self, name, data_path, total_frames, framelist, defs):
		super(Bar, self).__init__(name,**defs)
		self.fps = c_float(0.0)
		self.play = c_bool(0)
		self.get_single = c_bool(0)
		self.frame_num = c_int(0)
		self.display = c_int(0)
		self.exit = c_bool(0)
		self.framelist = framelist
		self.data_path = data_path

		self.record_video = c_bool(0)
		self.record_running = c_bool(0)

		self.add_var("Play", self.play, key="SPACE", help="Play/Pause") #key="SPACE",
		self.add_var("FPS",self.fps, step=0.01)
		# self.add_var("Frame Number", step=1, getter=self.get_frame_num, setter=self.set_frame_num,
		# 			help="Scrub through video frames.",
		# 			min=0, max=10)
		self.add_var("Frame Number", self.frame_num, min=0, max=total_frames-1)
		self.add_var("Display",self.display, step=1,max=3, min=0)
		self.add_button("Step", self.step_forward, key="s", help="Step forward one frame")
		self.add_button("Save Keyframe", self.add_keyframe, key="RETURN", help="Save keyframe to list")
		self.add_var("Record Video", self.record_video, key="R", help="Start/Stop Recording")

		self.add_var("Exit", self.exit)

	def update_fps(self, dt):
		if dt > 0:
			temp_fps = 1/dt
			self.fps.value += 0.1*(temp_fps-self.fps.value)

	def get_fps(self):
		return self.fps

	def screen_cap(self):
		# just flip the switch
		self.screen_shot = not self.screen_shot

	def step_forward(self):
		# just flip the switch
		self.get_single = 1

	def add_keyframe(self):
		if not self.frame_num.value in self.framelist.keyframes:
			print "Added Keyframe: %s to the list" %(self.frame_num.value)
			self.framelist.keyframes.append(self.frame_num.value)
			print "Keyframes:\n%s" %(self.framelist.keyframes)
			print "Number of Keyframes:\n%s" %(len(self.framelist.keyframes))
		else:
			print "Removing Keyframe"
			self.framelist.keyframes.remove(self.frame_num.value)
		if len(self.framelist.keyframes) > 1:
			self.framelist.keyframes = sorted(self.framelist.keyframes)
			self.framelist.otherframes = range(self.framelist.keyframes[0], self.framelist.keyframes[-1])

			self.framelist.otherframes = [i for i in self.framelist.otherframes if i not in self.framelist.keyframes]

			print "Non Keyframes:\n%s" %(self.framelist.otherframes)
			print "Number of Non Keyframes:\n%s" %(len(self.framelist.otherframes))

		np.save(os.path.join(self.data_path, "keyframes.npy"), np.asarray(self.framelist.keyframes))
		np.save(os.path.join(self.data_path, "otherframes.npy"), np.asarray(self.framelist.otherframes))


class Temp(object):
	"""Temp class to make objects"""
	def __init__(self):
		pass

def browser(data_path, video_path, pts_path, cam_intrinsics_path):

	record = Temp()
	record.path = None
	record.writer = None

	c = Temp()

	c.captures = [cv2.VideoCapture(path) for path in video_path]
	total_frames = min([cap.get(7) for cap in c.captures])
	record.fps = min([cap.get(5) for cap in c.captures])

	r, img_arr = c.captures[0].read()
	if len(c.captures)==2:
		r, img_arr2 =c.captures[1].read()


	img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
	fig = glumpy.figure((img_arr.shape[1], img_arr.shape[0]))
	image = glumpy.Image(img_arr)
	image.x, image.y = 0,0

	# gaze object
	gaze = Temp()
	gaze.list = np.load(pts_path)
	# gaze.x_pos = gaze.list[:,0]
	# gaze.y_pos = gaze.list[:,1]
	# gaze.dt = gaze.list[:,2]
	gaze_list = list(gaze.list)

	gaze_point = Point(color=(255,0,0,0.3), scale=40.0)
	positions_by_frame = [[] for frame in range(int(gaze_list[-1][-1]) + 1)]
	while gaze_list:
		s = gaze_list.pop(0)
		frame = int(s[-1])
		positions_by_frame[frame].append({'x': s[0], 'y': s[1], 'dt': s[2]})
	gaze.map = positions_by_frame

	# keyframe list object
	framelist = Temp()
	framelist.keyframes = []
	framelist.otherframes = []

	cam_intrinsics = Temp()
	cam_intrinsics.H_map = []

	g_pool = Temp()


	if cam_intrinsics_path is not None:
		cam_intrinsics.K = np.load(cam_intrinsics_path[0])
		cam_intrinsics.dist_coefs = np.load(cam_intrinsics_path[1])

	atb.init()
	bar = Bar("Browser", data_path, total_frames, framelist, dict(label="Controls",
			help="Scene controls", color=(50,50,50), alpha=50,
			text='light', position=(10, 10), size=(200, 440)))

	def draw():
		gaze_point.draw()


	def on_draw():
		fig.clear(0.0, 0.0, 0.0, 1.0)
		image.draw(x=image.x, y=image.y, z=0.0,
					width=fig.width, height=fig.height)
		draw()

	def on_close():
		pass


		print "Close event !"

	def on_idle(dt):
		bar.update_fps(dt)
		sleep(0.03)

		if bar.play or bar.get_single:
			# load new images
			r, img1 = c.captures[0].read()
			if len(c.captures)==2:
				r, img2 =c.captures[1].read()
				if r and img1.shape != img2.shape:
					img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))

			if not r:
				bar.play.value = 0
				return
			bar.frame_num.value +=1
			#stop playback when at the end of file.

			if bar.frame_num.value == 0:
				bar.play.value = 0

			# Extract corresponing Pupil posistions.
			# Here we are taking only the first values of the frame for positions hence 0 index
			try:
				x_screen, y_screen = denormalize((gaze.map[bar.frame_num.value][0]['x'],
														gaze.map[bar.frame_num.value][0]['y']),
														fig.width, fig.height, flip_y=True)
				img1[int(y_screen), int(x_screen)] = [255,255,255]

				# update gaze.x_screen, gaze.y_screen /OPENGL COORIDANTE SYSTEM
				gaze.x_screen,gaze.y_screen = flip_horizontal((x_screen,y_screen), fig.height)
				gaze_point.update((	gaze.x_screen, gaze.y_screen))
				print x_screen, y_screen
			except:
				pass

			if cam_intrinsics_path is not None and bar.display.value is not 0:
				# undistor world image
				img1 = cv2.undistort(img1, cam_intrinsics.K, cam_intrinsics.dist_coefs)
				# Undistort the gaze point based on the distortion coefs
				x_screen, y_screen = undistort_point((x_screen, y_screen),
									cam_intrinsics.K, cam_intrinsics.dist_coefs)

				if bar.display.value in (2,3):
					# homography mapping
					overlay, H = homography_map(img2, img1) # map img1 onto img2 (the world onto the source video)
					# cam_intrinsics.H_map.append([bar.frame_num.value, H])
					if overlay is not None:

						pt_homog = np.array([x_screen, y_screen, 1])
						pt_homog = np.dot(H, pt_homog)
						pt_homog /= pt_homog[-1] # normalize the gaze.pts
						x_screen, y_screen, z = pt_homog

						img1=overlay #overwrite img with the overlay

				if bar.display.value == 3:
					# cv2.circle(img2, (int(x_screen), int(y_screen)), 10, (0,255,0,100), 1)
					img1=img2 #overwrite img1 with the source video


			# update the img array
			img_arr[...] = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)


			#recorder logic
			if bar.record_video.value and not bar.record_running.value:
				record.path = os.path.join(bar.data_path, "out.avi")
				record.writer = cv2.VideoWriter(record.path,cv2.cv.CV_FOURCC(*'DIVX'),record.fps, (img1.shape[1],img1.shape[0]) )
				bar.record_running.value = 1

			if bar.record_video.value and bar.record_running.value:
				# Save image frames to video writer
				try:
					cv2.circle(img1, (int(x_screen), int(y_screen)), 20, (0,255,0,100), 1)
				except:
					pass
				record.writer.write(img1)

			# Finish all recordings, clean up.
			if not bar.record_video.value and bar.record_running.value:
				record.writer = None
				bar.record_running.value = 0



			#just grab one image.
			bar.get_single = 0


		image.update()
		fig.redraw()
		if bar.exit:
			on_close()
			fig.window.stop()


	fig.window.push_handlers(on_idle)
	fig.window.push_handlers(atb.glumpy.Handlers(fig.window))
	fig.window.push_handlers(on_draw)
	fig.window.push_handlers(on_close)
	fig.window.set_title("Browser")
	fig.window.set_position(0,0)
	glumpy.show()
