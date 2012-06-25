#!/usr/bin/env python

import numpy as np
import cv2


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
	x, y = pos
	x = (x*width/2)+(width/2)
	if flip_y:
		y = (-y*height/2)+(height/2)
	else:
		y = (y*height/2)+(height/2)
	return x,y

def flip_horizontal(pos, height):
	x, y = pos
	y -= height/2
	y *= -1
	y += height/2
	return x,y







