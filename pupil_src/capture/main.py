'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

# make shared modules available across pupil_src
import sys, os
loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
sys.path.append(os.path.join(loc[0], 'pupil_src', 'shared_modules'))

from time import sleep
from ctypes import c_bool, c_int
from multiprocessing import Process, Pipe, Event
from multiprocessing.sharedctypes import RawValue, Value, Array

#if you pass any additional argument when calling this script. The profiler will be used.
if len(sys.argv) >=2:
	from eye import eye_profiled as eye
	from world import world_profiled as world
else:
	from eye import eye
	from world import world

from player import player
from methods import Temp


def main():

	# To assign by name: put string(s) in list
	eye_src = ["Microsoft", "6000"]
	world_src = ["Logitech Camera", "C525","C615","C920","C930e"]

	# to assign cameras directly, using integers as demonstrated below
	eye_src = 1
	# world_src = 0

	# to use a pre-recorded video.
	# Use a string to specify the path to your video file as demonstrated below
	# eye_src = "/Users/mkassner/Pupil/pupil_google_code/wiki/videos/eye_simple_filter.avi"
	# world_src = 0

	# Camera video size in pixels (width,height)
	eye_size = (640,360)
	world_size = (1280,720)


	# Use the player - a seperate window for video playback and calibration animation
	use_player = True
	#startup size for the player window: this can be whatever you like
	player_size = (640,360)

	# Create and initialize shared globals
	g_pool = Temp()
	g_pool.gaze = Array('d',(0.0,0.0))
	g_pool.ref = Array('d',(0.0,0.0))
	g_pool.marker = Array('d',(0.0,0.0))
	g_pool.marker_state = Value('d',0.0)
	g_pool.calibrate = Value(c_bool, 0)
	g_pool.pos_record = Value(c_bool, 0)
	g_pool.eye_rx, g_pool.eye_tx = Pipe(False)
	g_pool.player_refresh = Event()
	g_pool.player_input = Value('i',0)
	g_pool.play = RawValue(c_bool,0)
	g_pool.quit = RawValue(c_bool,0)
	# shared constants
	g_pool.eye_src = eye_src
	g_pool.eye_size = eye_size
	g_pool.world_src = world_src
	g_pool.world_size = world_size

	# set up subprocesses
	p_eye = Process(target=eye, args=(g_pool,))
	if use_player: p_player = Process(target=player, args=(g_pool,player_size))

	# spawn subprocesses
	if use_player: p_player.start()
	p_eye.start()

	# On Linux, we need to give the camera driver some time before requesting another camera.
	sleep(1)

	# on MacOS, when using some cameras (like our current logitech worldcamera)
	# you can't run the world camera grabber in its own process
	# it must reside in the main process when you run on MacOS.
	world(g_pool)

	# Exit / clean-up
	p_eye.join()
	if use_player: p_player.join()
	print "main exit"

if __name__ == '__main__':
	main()
