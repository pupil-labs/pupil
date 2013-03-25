import os, sys
import numpy as np
import cv2
from time import sleep
from multiprocessing import Process, Pipe, Event
from multiprocessing.sharedctypes import RawValue, Value
# RawValue is shared memory without lock, handle with care, this is usefull for ATB it needs cTypes
from eye import eye, eye_profiled
from world import world, world_profiled
from player import player
from methods import Temp

# import pyaudio
# import waveatb.
# from audio import normalize, trim, add_silence

from ctypes import c_bool, c_int



def main():
	#assign the right id to the cameras
	eye_src = 0
	world_src = 1

	#video size
	eye_size = (640,360)
	"""
		HD-6000 v4l2-ctl -d /dev/video0 --list-formats-ext
		Size: Discrete 640x480
		Size: Discrete 1280x720
		Size: Discrete 960x544
		Size: Discrete 800x448
		Size: Discrete 640x360
		Size: Discrete 800x600
		Size: Discrete 416x240
		Size: Discrete 352x288
		Size: Discrete 176x144
		Size: Discrete 320x240
		Size: Discrete 160x120
	"""
	world_size = (1280,720)
	"""
		c-525 v4l2-ctl -d /dev/video0 --list-formats-ext
	    Size: Discrete 640x480
        Size: Discrete 160x120
        Size: Discrete 176x144
        Size: Discrete 320x176
        Size: Discrete 320x240
        Size: Discrete 432x240
        Size: Discrete 352x288
        Size: Discrete 544x288
        Size: Discrete 640x360
        Size: Discrete 752x416
        Size: Discrete 800x448
        Size: Discrete 864x480
        Size: Discrete 960x544
        Size: Discrete 1024x576
        Size: Discrete 800x600
        Size: Discrete 1184x656
        Size: Discrete 960x720
        Size: Discrete 1280x720
        Size: Discrete 1392x768
        Size: Discrete 1504x832
        Size: Discrete 1600x896
        Size: Discrete 1280x960
        Size: Discrete 1712x960
        Size: Discrete 1792x1008
        Size: Discrete 1920x1080
    """

	player_size = (800,600)

	#use video for debugging
	use_video = 0

	audio = False

	# use the player: a seperate window for video playback and 9 point calibration animation
	use_player = 1


	if use_video:
		eye_src = "/Users/mkassner/Pupil/pupil_google_code/wiki/videos/green_eye_VISandIR_2.mov" # using a path to a videofiles allows for developement without a headset.
		world_src = 0

	# create shared globals
	g_pool = Temp()
	g_pool.gaze_x = Value('d', 0.0)
	g_pool.gaze_y = Value('d', 0.0)
	g_pool.pattern_x = Value('d', 0.0)
	g_pool.pattern_y = Value('d', 0.0)
	g_pool.frame_count_record = Value('i', 0)
	g_pool.calibrate = RawValue(c_bool, 0)
	g_pool.cal9 = RawValue(c_bool, 0)
	g_pool.cal9_stage = Value('i', 0)
	g_pool.cal9_step = Value('i', 0)
	g_pool.cal9_circle_id = RawValue('i' ,0)
	g_pool.pos_record = Value(c_bool, 0)
	g_pool.eye_rx, g_pool.eye_tx = Pipe(False)

	g_pool.audio_record = Value(c_bool,False)
	g_pool.audio_rx, g_pool.audio_tx = Pipe(False)
	g_pool.player_refresh = Event()
	g_pool.play = RawValue(c_bool,0)
	g_pool.quit = RawValue(c_bool,0)
	# end shared globals

	# set up sub processes
	p_eye = Process(target=eye, args=(eye_src,eye_size, g_pool))
	if use_player: p_player = Process(target=player, args=(g_pool,player_size))
	if audio: p_audio = Process(target=record_audio, args=(g_pool.audio_rx,g.g_pool.audio_record,3))

	# spawn sub processes
	p_eye.start()
	if use_player: p_player.start()
	if audio: p_audio.start()

	# when using some cameras (like our current worldcamera logitch c510)
	# you can't run world camera grabber in its own process
	# it must reside in the main loop when you run on MacOS.
	world(world_src,world_size,g_pool)

	# exit / clean-up
	p_eye.join()
	if use_player: p_player.join()
	if audio: p_audio.join()
	print "main exit"

if __name__ == '__main__':
	main()