import os, sys

import numpy as np 
import cv2

from time import sleep
from multiprocessing import Process, Queue, Pipe, Event
from multiprocessing.sharedctypes import RawValue, Value # RawValue is shared memory without lock, handle with care, this is usefull for ATB it needs cTypes
from eye import eye
from world import world
from player import player
from methods import Temp
from array import array
from struct import unpack, pack
import pyaudio
import wave

from audio import normalize, trim, add_silence

from utilities.usb_camera_interface import cam_interface
from ctypes import *



def main():
	
	audio = False
 	# assing the right id to the cameras
	eye_id = 0
	world_id = 1
	if(0):
		eye_id = "/Users/mkassner/MIT/pupil_google_code/wiki/videos/green_eye_VISandIR_2.mov" # unsing a path to a videofiles allows for developement without a headset.
		world_id = 0




	# create shared globals for pupil coords
	# and pattern coordinates from the world process
	# and global for record and calibrate buttons
	g_pool = Temp()
	g_pool.pupil_x = Value('d', 0.0)
	g_pool.pupil_y = Value('d', 0.0)
	g_pool.pattern_x = Value('d', 0.0) 
	g_pool.pattern_y = Value('d', 0.0) 
	g_pool.frame_count_record = Value('i', 0)
	g_pool.calibrate = RawValue(c_bool, 0)
	g_pool.quit = RawValue(c_bool,0)
	g_pool.pos_record = Value(c_bool, 0)
	g_pool.eye_rx, g_pool.eye_tx = Pipe(False)
	g_pool.player_pipe_new = Event()
	g_pool.player_rx, g_pool.player_tx = Pipe(True)
	g_pool.audio_record = Value(c_bool,False)
	g_pool.audio_rx, g_pool.audio_tx = Pipe(False)


	p_show_eye = Process(target=eye, args=(eye_id, g_pool))
	p_show_world = Process(target=world, args=(world_id,g_pool))
	p_player = Process(target=player, args=(g_pool,))

	# Audio:
	# src=3 for logitech, rate=16000 for logitech 
	# defaults for built in MacBook microphone
	
	if audio: p_audio = Process(target=record_audio, args=(audio_rx,audio_record,3)) 

	p_show_eye.start()
	p_show_world.start()
	p_player.start()

	if audio: p_audio.start()
	
	# when using the logitech h264 compression camera
	# you can't run world camera in its own process
	# it must reside in the main loop
	# grab(world_q, world_id, (640,480))


	p_show_eye.join()
	p_show_world.join()
	if audio: p_audio.join()
	print "main exit"

if __name__ == '__main__':
	main()



########### old Grab Routines ###########

def grab(q,src_id, size=(640,480)):
	"""grab:
		- Initialize a camera feed
		- Stream images to queue 
		- Non-blocking
		- release cam if dropped frames>50
	"""
	cap = cv2.VideoCapture(src_id)
	cap.set(3, size[0])
	cap.set(4, size[1])
	# cap.set(5, 30)
	drop = 50
	while 1:
		status, img = cap.read()
		if status:
			q.put(img.shape)
			break

			
	while True:
		# Hack tell the process to sleep
		sleep(0.01)
		status, img = cap.read()
		#img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		if status:
			try:
				q.put(img, True)
				drop = 50 
			except:
				print "Camera Dropped Frame"
				drop -= 1
				if not drop:
					cap.release()
					return

		else:
			cap.set(1,0) #loops video







