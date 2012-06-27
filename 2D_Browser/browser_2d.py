import os, sys, argparse

import glumpy 
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glumpy.atb as atb
from ctypes import *
import numpy as np

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
		self.fps = 0.0 
		self.play = c_bool(0)
		self.get_single = c_bool(0)
		self.frame_num = c_int(0)
		self.display = 0
		self.exit = c_bool(0)
		self.framelist = framelist
		self.data_path = data_path

		self.record_video = c_bool(0)
		self.record_running = c_bool(0)

		self.add_var("Play", self.play, key="SPACE", help="Play/Pause") #key="SPACE",
		self.add_var("FPS", step=0.01, getter=self.get_fps)
		# self.add_var("Frame Number", step=1, getter=self.get_frame_num, setter=self.set_frame_num,
		# 			help="Scrub through video frames.",
		# 			min=0, max=10)
		self.add_var("Frame Number", self.frame_num, min=0, max=total_frames-1)
		self.add_var("Display", step=1, getter=self.get_display, setter=self.set_display,
					max=3, min=0)
		self.add_button("Step", self.step_forward, key="s", help="Step forward one frame")
		self.add_button("Save Keyframe", self.add_keyframe, key="RETURN", help="Save keyframe to list")
		self.add_var("Record Video", self.record_video, key="R", help="Start/Stop Recording")

		self.add_var("Exit", self.exit)

	def update_fps(self, dt):
		if dt > 0:
			temp_fps = 1/dt
			self.fps += 0.1*(temp_fps-self.fps)

	def get_fps(self):
		return self.fps

	def screen_cap(self):
		# just flip the switch
		self.screen_shot = not self.screen_shot

	def get_frame_num(self):
		return self.frame_num
	def set_frame_num(self, val):
		self.frame_num = val

	def get_display(self):
		return self.display
	def set_display(self, val):
		self.display = val

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

def browser(data_path, pipe_video, pts_path, audio_pipe, cam_intrinsics_path, running):

	record = Temp()
	record.path = None
	record.writer = None

	# source video fps
	record.fps = pipe_video.recv()

	# Get image array from queue, initialize glumpy, map img_arr to opengl texture 
	total_frames = pipe_video.recv()

	img_arr = cv2.cvtColor(pipe_video.recv(), cv2.COLOR_BGR2RGB)
	img_arr2 = cv2.cvtColor(pipe_video.recv(), cv2.COLOR_BGR2RGB)
	# img_arr, H = homography_map(img_arr, img_arr2)

	fig = glumpy.figure((img_arr.shape[1], img_arr.shape[0]))
	image = glumpy.Image(img_arr)
	image.x, image.y = 0,0

	# gaze object
	gaze = Temp()
	gaze.list = np.load(pts_path)
	# gaze.x_pos = gaze.list[:,0]
	# gaze.y_pos = gaze.list[:,1]
	# gaze.dt = gaze.list[:,2]
	gaze_point = Point(color=(255,0,0,0.3), scale=40.0)

	gaze_list = list(gaze.list)
	gaze.map = [[{'eye_x': s[0], 'eye_y': s[1], 'dt': s[2]} for s in gaze_list if s[3] == frame] for frame in range(int(gaze_list[-1][-1])+1)]
	gaze.pts = np.array([[i[0]['eye_x'], i[0]['eye_y']] for i in gaze.map if len(i) > 0], dtype=np.float32)


	# keyframe list object
	framelist = Temp()
	framelist.keyframes = []
	framelist.otherframes = []

	cam_intrinsics = Temp()
	cam_intrinsics.H_map = []

	g_pool = Temp()
	g_pool.running = running


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
		g_pool.running.value = 0
		# while pipe_video.poll(0.3):
		while True:
			try:
				pipe_video.recv()
			except:
				print "exception, nothing to recv."
				break
			# dump = pipe_video.recv()
		print "Close event !"

	def on_idle(dt):
		bar.update_fps(dt)

		if bar.play or bar.get_single:

			# audio_pipe.send(bar.play)

			bar.frame_num.value = pipe_video.recv()

			if bar.frame_num.value == 0:
				bar.play.value = 0

			img1 = cv2.cvtColor(pipe_video.recv(), cv2.COLOR_BGR2RGB)
			img2 = cv2.cvtColor(pipe_video.recv(), cv2.COLOR_BGR2RGB)

			overlay_img = img1

			# Here we are taking only the first values of the frame for positions hence 0 index
			gaze.x_screen, gaze.y_screen = denormalize((gaze.map[bar.frame_num.value][0]['eye_x'], 
														gaze.map[bar.frame_num.value][0]['eye_y']), 
														fig.width, fig.height)

			l_x_screen, l_y_screen = denormalize((gaze.map[bar.frame_num.value][0]['eye_x'], 
														gaze.map[bar.frame_num.value][0]['eye_y']), 
														fig.width, fig.height, flip_y=False)

			overlay_img[int(l_y_screen), int(l_x_screen)] = [255,255,255]
			# overlay_img[int(l_y_screen)+1, int(l_x_screen)+1] = [255,255,255]
			# overlay_img[int(l_y_screen), int(l_x_screen)+1] = [255,255,255]
			# overlay_img[int(l_y_screen)+1, int(l_x_screen)] = [255,255,255]


			# show rectified image without homography mapping for debugging
			if cam_intrinsics_path is not None and bar.display == 1:
				img1 = cv2.undistort(img1, cam_intrinsics.K, cam_intrinsics.dist_coefs)

				l_x_screen, l_y_screen = denormalize((gaze.map[bar.frame_num.value][0]['eye_x'], 
														gaze.map[bar.frame_num.value][0]['eye_y']), 
														fig.width, fig.height, flip_y=False)

				# Undistort the gaze point based on the distortion coefs (Is K necessary?) 
				x_screen, y_screen = undistort_point((l_x_screen, l_y_screen), 
									cam_intrinsics.K, cam_intrinsics.dist_coefs)
				
				# turn normalized x,y back into screen coordinates
				# x_screen, y_screen = denormalize((x, y), fig.width, fig.height)


				# update gaze.x_screen, gaze.y_screen
				x_screen,y_screen = flip_horizontal((x_screen,y_screen), fig.height)
				gaze.x_screen = x_screen
				gaze.y_screen = y_screen


			# show homography and rectified image with overlay
			if cam_intrinsics_path is not None and bar.display == 2:
				img1 = cv2.undistort(img1, cam_intrinsics.K, cam_intrinsics.dist_coefs)

				overlay_img, H = homography_map(img2, img1) # flipped img1 & img2 -- now have correct homography for the points
				# cam_intrinsics.H_map.append([bar.frame_num.value, H])

				l_x_screen, l_y_screen = denormalize((gaze.map[bar.frame_num.value][0]['eye_x'], 
														gaze.map[bar.frame_num.value][0]['eye_y']), 
														fig.width, fig.height, flip_y=False)

				# Undistort the gaze point based on the distortion coefs (Is K necessary?) 
				x_screen, y_screen = undistort_point((l_x_screen, l_y_screen), 
									cam_intrinsics.K, cam_intrinsics.dist_coefs)


				gaze.pt_homog = np.array([	x_screen, 
											y_screen, 
											1])

				gaze.pt_homog = np.dot(H, gaze.pt_homog)
				gaze.pt_homog /= gaze.pt_homog[-1] # normalize the gaze.pts

				# print "homog pts: %s,%s" %(gaze.pt_homog[0], gaze.pt_homog[1])
				
				# x coordinate is correct it seems
				# the y coordinate seems to be correct, but flipped
				gaze.x_screen, gaze.y_screen = flip_horizontal((gaze.pt_homog[0], gaze.pt_homog[1]), fig.height)


			if cam_intrinsics_path is not None and bar.display == 3:
				img1 = cv2.undistort(img1, cam_intrinsics.K, cam_intrinsics.dist_coefs)

				overlay_img, H = homography_map(img2, img1) # flipped img1 & img2 -- now have correct homography for the points
				# cam_intrinsics.H_map.append([bar.frame_num.value, H])
				
				l_x_screen, l_y_screen = denormalize((gaze.map[bar.frame_num.value][0]['eye_x'], 
														gaze.map[bar.frame_num.value][0]['eye_y']), 
														fig.width, fig.height, flip_y=False)

				# Undistort the gaze point based on the distortion coefs (Is K necessary?) 
				x_screen, y_screen = undistort_point((l_x_screen, l_y_screen), 
									cam_intrinsics.K, cam_intrinsics.dist_coefs)


				gaze.pt_homog = np.array([	x_screen, 
											y_screen, 
											1])

				gaze.pt_homog = np.dot(H, gaze.pt_homog)
				gaze.pt_homog /= gaze.pt_homog[-1] # normalize the gaze.pts

				# print "homog pts: %s,%s" %(gaze.pt_homog[0], gaze.pt_homog[1])
				cv2.circle(img2, (int(gaze.pt_homog[0]), int(gaze.pt_homog[1])), 50, (255,0,0,100), 2, cv2.CV_AA) 

				# x coordinate is correct it seems
				# the y coordinate seems to be correct, but flipped
				gaze.x_screen, gaze.y_screen = flip_horizontal((gaze.pt_homog[0], gaze.pt_homog[1]), fig.height)


			if bar.display == 0:
				img_arr[...] = img1
				gaze_point.update((	gaze.x_screen, gaze.y_screen))
			if bar.display == 1:
				img_arr[...] = img1
				gaze_point.update((	gaze.x_screen, gaze.y_screen))
			if bar.display == 2:
				img_arr[...] = overlay_img
				gaze_point.update((	gaze.x_screen, gaze.y_screen))
			if bar.display == 3:
				img_arr[...] = img2
				gaze_point.update((	gaze.x_screen, gaze.y_screen))



			if bar.record_video and not bar.record_running:
				record.path = os.path.join(bar.data_path, "out.avi")
				record.writer = cv2.VideoWriter(record.path,cv2.cv.CV_FOURCC(*'DIVX'),record.fps, (img2.shape[1],img2.shape[0]) )
				bar.record_running.value = 1

			if bar.record_video and bar.record_running:
				# Save image frames to video writer
				record.writer.write(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

			# Finish all recordings, clean up. 
			if not bar.record_video and bar.record_running:
				record.writer = None
				bar.record_running = 0

			bar.get_single = 0
	
		# else:
		# 	sleep(0.5)
		# np.save("data/homography_map.npy", cam_intrinsics.H_map)
		

		image.update()
		fig.redraw()
		if bar.exit:
			pass
			#fig.window.stop()
	

	fig.window.push_handlers(on_idle)
	fig.window.push_handlers(atb.glumpy.Handlers(fig.window))
	fig.window.push_handlers(on_draw)	
	fig.window.push_handlers(on_close)	
	fig.window.set_title("Browser")
	fig.window.set_position(0,0)	
	glumpy.show() 	
