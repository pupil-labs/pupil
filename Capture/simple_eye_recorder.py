import os, sys

import numpy as np 
import cv2

from time import sleep
from multiprocessing import Process, Queue, Pipe, Value
from simple_eye import eye

from utilities.usb_camera_interface import cam_interface
from ctypes import *


def xmos_grab(q,id,size):
	size= size[::-1] # swap sizes as numpy is row first
	drop = 50
	cam = cam_interface()
	buffer = np.zeros(size, dtype=np.uint8) #this should always be a multiple of 4
	cam.aptina_setWindowSize(cam.id0,(size[1],size[0])) #swap sizes back 
	cam.aptina_setWindowPosition(cam.id0,(240,100))
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

def main():
	eye_id = 0

	eye_q = Queue(3)

	# p_grab_eye = Process(target=xmos_grab, args=(eye_q, eye_id, (400,250)))

	p_show_eye = Process(target=eye, args=(eye_q, ))

	# p_grab_eye.start()
	p_show_eye.start()

	xmos_grab(eye_q, eye_id, (400,250))

	# p_grab_eye.join()

if __name__ == '__main__':
	main()