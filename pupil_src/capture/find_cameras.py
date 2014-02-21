'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

def toggle_capture_devices():
	"""
	toggle though all attached camareas to find out
	ids assigned to cameras by your machine - as integers
	"""
	import cv2

	def quick_cap(src):
		vc = cv2.VideoCapture(id)
		vc.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
		vc.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
		return vc

	id = 0
	vc = quick_cap(id)
	rval, frame = vc.read()
	while rval:
		cv2.imshow("camera_id: "+ str(id), frame)
		key = cv2.waitKey(20)
		if key == 27: # exit on ESC
			break
		if key == 32: # space
			id +=1
			vc.release()
			vc = quick_cap(id)
		rval, frame = vc.read()


if __name__ == '__main__':
	toggle_capture_devices()