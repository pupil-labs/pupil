import os, sys

import numpy as np 
import cv2

from time import sleep
from multiprocessing import Process, Queue, Pipe, Value
from eye import eye
from world import world
from player import player

from array import array
from struct import unpack, pack
import pyaudio
import wave

from audio import normalize, trim, add_silence


from utilities.usb_camera_interface import cam_interface
from ctypes import *


def xmos_grab(q,id,size):
	size= size[::-1] # swap sizes as numpy is row first
	drop = 50
	cam = cam_interface()
	buffer = np.zeros(size, dtype=np.uint8) #this should always be a multiple of 4
	cam.aptina_setWindowSize(cam.id0,(size[1],size[0])) #swap sizes back 
	cam.aptina_setWindowPosition(cam.id0,(200,134))
	cam.aptina_LED_control(cam.id0,Disable = 0,Invert =0)
	cam.aptina_AEC_AGC(cam.id0,1,1) # Auto Exposure Control + Auto Gain Control
	cam.aptina_HDR(cam.id0,1)
	q.put(buffer.shape)
	while 1:
		if cam.get_frame(id,buffer): #returns True on sucess
			try:
				q.put(buffer,False)
				drop = 50 
			except:
				drop -= 1
				if not drop:
					cam.release()
					return


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
	drop = 50
	while 1:
		status, img = cap.read()
		if status:
			q.put(img.shape)
			break

			
	while True:
		status, img = cap.read()
		#img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# Hack tell the process to sleep
		sleep(0.01)
		try:
			q.put(img, False)
			drop = 50 
		except:
			# print "Camera Dropped Frame"
			drop -= 1
			if not drop:
				cap.release()
				return

def record_audio(pipe,record, src_id=0, rate=44100, chunk_size=1536, 
				format=pyaudio.paInt32, verbose=False):
	"""record_audio:
		- Initialize a global array to hold audio 
		- if record and not running initiaize PyAudio Stream 
		- if recording and running, add to data list
		- if not recording and not running, clean up data and save to wav
	"""
	g_LRtn = array('h')
	while True:
		#init new recording
		audio_out = pipe.recv() 
		p = pyaudio.PyAudio()
		if verbose: 
			print "Initialized PyAudio stream for recording"
		stream = p.open(input_device_index=src_id, format=format, 
					channels=1, rate=rate, 
					input=True, output=True,
					frames_per_buffer=chunk_size, 
					start = True)
		# stream.start_stream()
		#record loop
		while record.value:
			try:
				data = stream.read(chunk_size)
				L = unpack('<' + ('h'*(len(data)/2)), data) # little endian, signed short
				# print max(L)
				L = array('h', L)
				g_LRtn.extend(L)
				if verbose: 
					print "len(g_LRtn): %s" %(len(g_LRtn))
			except IOError,e:
				if e[1] == pyaudio.paInputOverflowed: 
					print e
					print "Audio Buffer Overflow."
					data = '\x00'*pyaudio.paInt32*chunk_size*1
		#clean up and save file
		if verbose: 
			print "Stopped recording...Saving File"
		sample_width = p.get_sample_size(pyaudio.paInt32)
		stream.stop_stream()
		stream.close()
		p.terminate()

		if verbose: 
			print "len(g_LRtn): ", len(g_LRtn)
		# g_LRtn = normalize(g_LRtn)
		# g_LRtn = trim(g_LRtn)
		# g_LRtn = add_silence(g_LRtn, 0.5)

		data = pack('<' + ('h'*len(g_LRtn)), *g_LRtn)

		wf = wave.open(audio_out, 'wb')
		wf.setnchannels(1)
		wf.setsampwidth(sample_width)
		wf.setframerate(rate)
		wf.writeframes(data)
		wf.close()
		print "Saved audio to: %s" %audio_out
		# clear global array when done recording
		g_LRtn = array('h')

		#repeat and wait at pipe receive


def main():
	eye_id = 0
	world_id = 0

	eye_q = Queue(3)
	world_q = Queue(3)

	audio_rx, audio_tx = Pipe(False)
	audio_record = Value('i',0)

	eye_rx, eye_tx = Pipe(False)

	player_rx, player_tx = Pipe(True)

	# create shared globals for pupil coords
	# and pattern coordinates from the world process
	# and global for record and calibrate buttons
	pupil_x = Value('d', 0.0)
	pupil_y = Value('d', 0.0)
	pattern_x = Value('d', 0.0) 
	pattern_y = Value('d', 0.0) 
	calibrate = Value('i', 0)
	pos_record = Value('i', 0)


	p_grab_eye = Process(target=xmos_grab, args=(eye_q,eye_id,(450,300)))
	# p_grab_world = Process(target=grab, args=(world_q,world_id))

	p_show_eye = Process(target=eye, args=(eye_q, pupil_x, pupil_y, 
											pattern_x, pattern_y, 
											calibrate, pos_record,
											eye_rx))
	p_show_world = Process(target=world, args=(world_q, pupil_x, pupil_y, 
												pattern_x, pattern_y,
												calibrate, pos_record,
												audio_tx, eye_tx, audio_record,
												player_tx))
	p_player = Process(target=player, args=(player_rx,))

	# Audio:
	# src=3 for logitech, rate=16000 for logitech 
	# defaults for built in MacBook microphone
	p_audio = Process(target=record_audio, args=(audio_rx,audio_record,3)) 

	p_show_world.start()
	p_grab_eye.start()
	p_audio.start()
	p_player.start()

	p_show_eye.start()
	# when using the logitech h264 compression camera
	# you can't run world camera in its own process
	# it must reside in the main loop
	grab(world_q, world_id, (640,480))

	p_grab_eye.join()
	# p_grab_world.join()
	p_audio.join()

if __name__ == '__main__':
	main()



