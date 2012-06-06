import os, sys, argparse, time

from ctypes import *

import cv2
import numpy as np

from multiprocessing import Process, Pipe, Value
from browser_2d import browser
from audio import play_audio

class Browser(object):
	"""Command Line manager for 2D Browser"""
	def __init__(self):
		self.parseCommandLineFlags()
		self.exceptions()
		self.load_data()

	def parseCommandLineFlags(self):
		parser = argparse.ArgumentParser(description="2D Browser for PUPIL data.  \
					Recorded eye movements are superimposed onto the recorded video stream for purposes of analysis.")
		parser.add_argument('-d', '--data_path', type=str, 
			help='The data directory from the capture routine (only required input).',
			required=True)
 
		try:
			args = parser.parse_args(namespace=self)						
		except:
			parser.print_help()
			sys.exit()

	def exceptions(self):
		if not os.path.isdir(self.data_path):
			raise Exception, "'%s' is not a directory.  Please specify a data_path directory." %(self.data_path)

	def load_data(self):
		os.chdir(self.data_path)
		self.video_path = os.path.join(self.data_path, "world.avi")
		self.audio_path = os.path.join(self.data_path, "world.wav")
		self.pts_path = os.path.join(self.data_path, "pupil_positions.npy")


def grab(pipe, frame_num, src):
	"""grab:
		- Initialize a camera feed (from file)
		- Return Images on demand
	"""
	#src = os.path.join(path, src)
	cap = cv2.VideoCapture(src)
	# cap.set(3, size[0])
	# cap.set(4, size[1])
	fps = cap.get(5)
	total_frames = cap.get(7)
	pipe.send(total_frames)
	while True:
		start_time = time.time()
		# cap.set(1, frame_num.value)

		status, img = cap.read()
		if status:
			pipe.send(img)
		time_passed = time.time()-start_time
		time.sleep(max(0,1/fps-time_passed))
		
		if frame_num.value < total_frames:
			frame_num.value += 1
		else:
			# loop back to the beginning 
			frame_num.value = 0


def main(data_path, video_path, audio_path, pts_path):	
	rx_video, tx_video = Pipe(False)
	rx_audio, tx_audio = Pipe(False)

	frame_num = Value('i', 0)

	p_browser = Process(target=browser, args=(data_path, rx_video,frame_num, pts_path, tx_audio))
	p_audio = Process(target=play_audio, args=(rx_audio,audio_path))

	p_browser.start()
	p_audio.start()
	
	grab(tx_video, frame_num, video_path)

	p_browser.join()

if __name__ == '__main__':
	#	data_path = "/Volumes/HD_Two/Users/Will/Documents/2012/MIT/Thesis/Thesis_Data/Capture/05052012/mpk_stata_05052012/003"
	b = Browser()
	main(b.data_path, b.video_path, b.audio_path, b.pts_path)


