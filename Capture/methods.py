#!/usr/bin/env python

import numpy as np
import cv2
from ctypes import *



class Temp(object):
	"""Temp class to make objects"""
	def __init__(self):
		pass

class capture():
	"""docstring for capture"""
	def __init__(self, src,size):
		self.src = src
		self.VideoCapture = cv2.VideoCapture(src)
		self.set_size(size)
	
	def set_size(self,(width,height)):
		self.size = (width,height)
		self.np_size = (height,width)
		if isinstance(self.src, int):
			self.VideoCapture.set(3, width)
			self.VideoCapture.set(4, height)

	def new_src(self,src):
		self.VideoCapture = cv2.VideoCapture(src)

	def read(self):
		s, img =self.VideoCapture.read()
		if not s:
			self.rewind()
			s, img = self.VideoCapture.read()
		return s,img

	def read_RGB(self):
		s,img = self.read()
		if s:
			cv2.cvtColor(img, cv2.COLOR_RGB2BGR,img)
		return s,img

	def read_HSV(self):
		s,img = self.read()
		if s:
			cv2.cvtColor(img, cv2.COLOR_RGB2HSV,img)
		return s,img

	def rewind(self):
		self.VideoCapture.set(1,0) #seeek to 0





def grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def extract_darkspot(image, image_lower=0.0, image_upper=255.0):
	"""extract_darkspot:
			head manager function to filter eye image by
			- erasing specular reflections
			- fitting ellipse to filtered image 
		Out: filtered image and center of ellipse
	"""
	binary_img = cv2.inRange(image, np.asarray(image_lower), 
				np.asarray(image_upper))

	# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
	# cv2.dilate(binary_img, kernel, binary_img, iterations=1)
	# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

	# cv2.erode(binary_img, kernel, binary_img, iterations=1)
	#binary_img = 255-binary_img
	return binary_img

def adaptive_threshold(image, image_lower=0.0, image_upper=255.0):
	"""extract_darkspot:
			head manager function to filter eye image by
			- erasing specular reflections
			- fitting ellipse to filtered image 
		Out: filtered image and center of ellipse
	"""
	image_lower = int(image_lower)*4
	image_lower +=1 
	image_lower = max(image_lower,3)
	binary_img = cv2.adaptiveThreshold(image, maxValue= 255, 
											adaptiveMethod= cv2.ADAPTIVE_THRESH_MEAN_C, 
											thresholdType= cv2.THRESH_BINARY_INV,
											blockSize=image_lower,
											C=image_upper-50)

	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	# cv2.erode(binary_img, kernel, binary_img, iterations=1)

	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
	cv2.erode(binary_img, kernel, binary_img, iterations=1)




	# cv2.dilate(binary_img, kernel, binary_img, iterations=1)
	#binary_img = 255-binary_img
	return binary_img


def equalize(image, image_lower=0.0, image_upper=255.0):
	image_lower = int(image_lower*2)/2
	image_lower +=1
	image_lower = max(3,image_lower)
	mean = cv2.medianBlur(image,255)
	image = image - (mean-100) 
	# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
	# cv2.dilate(image, kernel, image, iterations=1)
	return image

def erase_specular(image,lower_threshold=0.0, upper_threshold=150.0):
	"""erase_specular: removes specular reflections
			within given threshold using a binary mask (hi_mask)
	"""
	thresh = cv2.inRange(image, 
				np.asarray(lower_threshold), 
				np.asarray(upper_threshold))
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6,6))
	hilight = cv2.erode(image, kernel, iterations=2)
	hi_mask = cv2.dilate(thresh, kernel, iterations=2)
	# turn the hi_mask into boolean mask
	# currently it has values of 0 or 255 so 255=True
	hi_mask = (hi_mask==255)
	specular = image.copy()
	specular[hi_mask] = hilight[hi_mask]
	return specular



def erase_specular_new(image,lower_threshold=0.0, upper_threshold=150.0):
	"""erase_specular: removes specular reflections
			within given threshold using a binary mask (hi_mask)
	"""
	thresh = cv2.inRange(image, 
				np.asarray(float(lower_threshold)), 
				np.asarray(256.0))

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
	hi_mask = cv2.dilate(thresh, kernel, iterations=2)
	
	specular = cv2.inpaint(image, hi_mask, 2, flags=cv2.INPAINT_TELEA) 
	# return cv2.max(hi_mask,image)
	return specular

def add_horizontal_gradient(image,left=0,right=10):
	offset = np.linspace(left,right,image.shape[1]).astype(image.dtype)
	offset = np.repeat(offset[np.newaxis,:],image.shape[0],0)
	image += offset
	return image

def dif_gaus(image, lower, upper):
	lower, upper = int(lower-1), int(upper-1)
	lower = cv2.GaussianBlur(image,ksize=(lower,lower),sigmaX=0)
	upper = cv2.GaussianBlur(image,ksize=(upper,upper),sigmaX=0)
	# upper +=50
	# lower +=50

	dif = lower-upper
	# dif *= .1
	# dif = cv2.medianBlur(dif,3)
	# dif = 255-dif

	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
	dif = cv2.erode(dif, kernel, iterations=1)
	# dif = cv2.max(image,dif)

	# dif = cv2.dilate(dif, kernel, iterations=1)


	return dif 


def add_vertical_gradient(image,top=10,bottom=0):
	offset = np.linspace(top,bottom,image.shape[0]).astype(image.dtype)
	offset = np.repeat(offset[:,np.newaxis,],image.shape[1],1)
	image += offset
	return image


def fit_ellipse(image, contour_size=80):
	""" fit_ellipse:
			fit an ellipse around the pupil 
			the largest white spot within a binary image
	"""
	c_img = image.copy()
	contours, hierarchy = cv2.findContours(c_img, 
											mode=cv2.RETR_LIST, 
											method=cv2.CHAIN_APPROX_NONE)
	largest_ellipse = {'center': (None,None), 
						'axes': (None, None), 'angle': None, 
						'area': 0.0, 'ratio': None, 
						'major': None, 'minor': None}

	for c in contours:
		if len(c) >= contour_size:
			center, axes, angle = cv2.fitEllipse(c)
			area = axes[0]*axes[1]

			if area > largest_ellipse['area']:
				largest_ellipse['center'] = center
				largest_ellipse['axes'] = axes
				largest_ellipse['angle'] = angle
				largest_ellipse['area'] = area

	if largest_ellipse['angle']:
		largest_ellipse['major'] = max(largest_ellipse['axes'])
		largest_ellipse['minor'] = min(largest_ellipse['axes'])
		largest_ellipse['ratio'] = largest_ellipse['major']/largest_ellipse['minor']  
		if largest_ellipse['ratio'] > 3: # improve me
			print "blink"
			return None
	
		return largest_ellipse
	return None

def chessboard(image, pattern_size=(9,5)):
	status, corners = cv2.findChessboardCorners(image, pattern_size, flags=4)
	if status:
		mean = corners.sum(0)/corners.shape[0]
		# mean is [[x,y]]
		return mean[0], corners
	else:
		return None


def circle_grid(image, circle_id=None, pattern_size=(4,11)):
	"""Circle grid: finds an assymetric circle pattern
	- circle_id: sorted from bottom left to top right (column first)
	- If no circle_id is given, then the mean of circle positions is returned approx. center
	- If no pattern is detected, function returns None
	"""
	status, centers = cv2.findCirclesGridDefault(image, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
	if status:
		if circle_id is None:
			result = centers.sum(0)/centers.shape[0]
			# mean is [[x,y]]
			return result[0], centers
		else:
			return centers[circle_id][0], centers
	else:
		return None, None



def calibrate_camera(img_pts, obj_pts, img_size):
	# generate pattern size
	camera_matrix = np.zeros((3,3))
	dist_coef = np.zeros(4)
	rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, 
													img_size, camera_matrix, dist_coef)
	return camera_matrix, dist_coefs

def gen_pattern_grid(size=(4,11)):
	pattern_grid = []
	for i in xrange(size[1]):
		for j in xrange(size[0]):
			pattern_grid.append([(2*j)+i%2,i,0])
	return np.asarray(pattern_grid, dtype='f4')

def normalize(pos, width, height):
	"""
	normalize return as float
	"""
	x = pos[0]
	y = pos[1]
	x = (x-width/2.)/(width/2.)
	y = (y-height/2.)/(height/2.)
	return x,y

def denormalize(pos, width, height, flip_y=True):
	"""
	denormalize and return as int
	"""
	x = pos[0]
	y = pos[1]
	x = (x*width/2.)+(width/2.)
	if flip_y:
		y = (-y*height/2.)+(height/2.)
	else:
		y = (y*height/2.)+(height/2.)
	return int(x),int(y)

if __name__ == '__main__':
	tst = []	
	for x in range(10):
		tst.append(gen_pattern_grid())
	tst = np.asarray(tst)
	print tst.shape





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
		status, img = cap.read()
		#img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# Hack tell the process to sleep
		sleep(0.01)

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





