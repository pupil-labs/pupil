#!/usr/bin/env python

import numpy as np
import cv2
from ctypes import *

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

	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
	cv2.dilate(binary_img, kernel, binary_img, iterations=1)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

	cv2.erode(binary_img, kernel, binary_img, iterations=1)
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
	"""extract_darkspot:
			head manager function to filter eye image by
			- erasing specular reflections
			- fitting ellipse to filtered image 
		Out: filtered image and center of ellipse
	"""
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
				np.asarray(lower_threshold), 
				np.asarray(upper_threshold))
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
	hilight = np.zeros(thresh.shape).astype(np.uint8)
	hi_mask = cv2.dilate(thresh, kernel, iterations=1)
	# turn the hi_mask into boolean mask
	# currently it has values of 0 or 255 so 255=True
	hi_mask = (hi_mask==255)
	specular = image.copy()
	specular[hi_mask] = hilight[hi_mask]
	return specular


def add_horizontal_gradient(image,left=0,right=15):
	offset = np.linspace(left,right,image.shape[1]).astype(image.dtype)
	offset = np.repeat(offset[np.newaxis,:],image.shape[0],0)
	image += offset
	return image


def add_vertical_gradient(image,top=0,bottom=10):
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


def circle_grid(image, pattern_size=(4,11)):
	status, centers = cv2.findCirclesGridDefault(image, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
	if status:
		mean = centers.sum(0)/centers.shape[0]
		# mean is [[x,y]]
		return mean[0], centers
	else:
		return None

def calibrate_camera(img_pts, obj_pts, img_size):
	# generate pattern size
	camera_matrix = np.zeros((3,3))
	dist_coef = np.zeros(4)
	rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, 
													img_size, camera_matrix, dist_coef)
	return camera_matrix

def gen_pattern_grid(size=(4,11)):
	pattern_grid = []
	for i in xrange(size[1]):
		for j in xrange(size[0]):
			pattern_grid.append([(2*j)+i%2,i,0])
	return np.asarray(pattern_grid, dtype='f4')

def normalize(pos, width, height):
	x = pos[0]
	y = pos[1]
	x = (x-width/2)/(width/2)
	y = (y-height/2)/(height/2)
	return x,y

def denormalize(pos, width, height, flip_y=True):
	x = pos[0]
	y = pos[1]
	x = (x*width/2)+(width/2)
	if flip_y:
		y = (-y*height/2)+(height/2)
	else:
		y = (y*height/2)+(height/2)
	return x,y

tst = []
for i in range(10):
	tst.append(gen_pattern_grid())









