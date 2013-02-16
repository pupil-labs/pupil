#!/usr/bin/env python

import numpy as np
import cv2
from ctypes import *


def pre_integral_pupil_respone_filter():
###2D filter response as fisrt estimation of pupil position for ROI
	downscale = 4
	best_m = 0
	region_r = min(max(9,l_pool.region_r),61)
	lable = 0
	for s in (region_r-4,region_r,region_r+4):
		#simple best of three optimization
		kernel = make_eye_kernel(s,int(3*s))
		g_img = cv2.filter2D(gray_img[::downscale,::downscale],cv2.CV_32F,kernel,borderType=cv2.BORDER_REFLECT_101)        # ddepth = -1, means destination image has depth same as input image.
		m = np.amax(g_img)
		# print s,m
		x,y = np.where(g_img == m)
		x,y = downscale*y[0],downscale*x[0]
		cv2.putText(gray_img, str(s)+"-"+str(m), (x,y), cv2.FONT_HERSHEY_SIMPLEX, .35,(255,255,255))
		lable+=40
		inner_r = (s*downscale)/2
		outer_r = int(s*downscale*1.)
		if m > best_m:
			best_m = m
			l_pool.region_r = s
			vals = [max(0,v) for v in (x-outer_r,y-outer_r,x+outer_r,y+outer_r)]
			p_r.set(vals)
		# cv2.rectangle(gray_img, (x-inner_r,y-inner_r), (x+inner_r,y+inner_r), (0,0,0))
		# cv2.rectangle(gray_img,  (x-outer_r,y-outer_r), (x+outer_r,y+outer_r), (255,255,255))
	# g_img= cv2.normalize(g_img,alpha = 0,beta = 255,norm_type = cv2.NORM_MINMAX)
	# gray_img = cv2.resize(g_img,gray_img.shape[::-1], interpolation=cv2.INTER_NEAREST)



def make_eye_kernel(inner_size,outer_size):
	offset = (outer_size - inner_size)/2
	inner_count = inner_size**2
	outer_count = outer_size**2-inner_count
	val_inner = -1.0 / inner_count
	val_outer = -val_inner*inner_count/outer_count
	inner = np.ones((inner_size,inner_size),np.float32)*val_inner
	kernel = np.ones((outer_size,outer_size),np.float32)*val_outer
	kernel[offset:offset+inner_size,offset:offset+inner_size]= inner
	return kernel

class Temp(object):
	"""Temp class to make objects"""
	def __init__(self):
		pass

class capture():
	"""docstring for capture"""
	def __init__(self, src, size=None):
		self.src = src
		self.auto_rewind = False
		if isinstance(self.src, int) or isinstance(self.src, str):
			#set up as cv2 capture
			self.VideoCapture = cv2.VideoCapture(src)
			self.set_size(size)
			self.get_frame = self.VideoCapture.read
		elif src == None:
			self.VideoCapture = None
			self.get_frame = None
		else:
			#set up as pipe
			self.VideoCapture = src
			self.size = size
			self.np_size = size[::-1]
			self.VideoCapture.send(self.size) #send desired size to the capture function in the main
			self.get_frame = self.VideoCapture.recv #retrieve first frame

	def set_size(self,size):
		if size is not None:
			if isinstance(self.src, int):
				self.size = size
				width,height = size
				self.VideoCapture.set(3, width)
				self.VideoCapture.set(4, height)
			else:
				self.size = self.VideoCapture.get(3),self.VideoCapture.get(4)
			self.np_size = self.size[::-1]

	def read(self):
		s, img =self.get_frame()
		if  self.auto_rewind and not s:
			self.rewind()
			s, img = self.get_frame()
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
		self.VideoCapture.set(1,0) #seek to 0



def local_grab_threaded(pipe_world,src_id_world,pipe_eye,src_id_eye,g_pool):
	import threading
	from time import sleep, time

	class capture_thread(threading.Thread):
		"""docstring for capture_thread"""
		def __init__(self,pipe,src):
			threading.Thread.__init__(self)
			self.pipe = pipe
			self.src = src
			self.cap_init(src)

		def cap_init(self,src_id):
			self.cap = cv2.VideoCapture(src_id)
			size = self.pipe.recv() #recieve desired size from caputure instance from inside the other process.
			self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, size[0])
			self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, size[1])
		def run(self):
			tick = time()
			while not g_pool.quit.value:
				tick = time()
				self.pipe.send(self.cap.read())
				sleep(max(0,1/31.-(time()-tick)))

	"""grab:
		- Initialize a camera feed
		-this is needed for certain cameras that have to run in the main loop.
		- it pushes image frames to the capture class
		  that is initialize with one pipeend as the source
	"""
	thread1 = capture_thread(pipe_world,src_id_world)
	thread2 = capture_thread(pipe_eye,src_id_eye)
	thread1.start() # This actually causes the thread to run
	thread2.start()
	thread1.join()  # This waits until the thread has completed
	thread2.join()

	print "Local Grab exit"



def local_grab(pipe,src_id,g_pool):
    """grab:
        - Initialize a camera feed
        -this is needed for certain cameras that have to run in the main loop.
        - it pushed image frames to the capture class
            that it initialize with one pipeend as the source
    """

    quit = g_pool.quit
    cap = cv2.VideoCapture(src_id)
    size = pipe.recv() #recieve designed size from caputure instance from inside the other process.
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, size[1])

    while not quit.value:
        try:
            # cap.read() #uncomment to read at .5 fps
            pipe.send(cap.read())
        except:
            pass
    print "Local Grab exit"


def grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def bin_thresholding(image, image_lower=0, image_upper=256):
	"""
	needs docstring
	"""
	binary_img = cv2.inRange(image, np.asarray(image_lower),
				np.asarray(image_upper))

	# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
	# cv2.dilate(binary_img, kernel, binary_img, iterations=1)
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
        dif = cv2.inRange(dif, np.asarray(200),
				np.asarray(256))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        dif = cv2.dilate(dif, kernel, iterations=2)
        dif = cv2.erode(dif, kernel, iterations=1)
        # dif = cv2.max(image,dif)

        # dif = cv2.dilate(dif, kernel, iterations=1)


        return dif

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
				np.asarray(float(lower_threshold)),
				np.asarray(256.0))

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
	hi_mask = cv2.dilate(thresh, kernel, iterations=2)

	specular = cv2.inpaint(image, hi_mask, 2, flags=cv2.INPAINT_TELEA)
	# return cv2.max(hi_mask,image)
	return specular



def chessboard(image, pattern_size=(9,5)):
	status, corners = cv2.findChessboardCorners(image, pattern_size, flags=4)
	if status:
		mean = corners.sum(0)/corners.shape[0]
		# mean is [[x,y]]
		return mean[0], corners
	else:
		return None






def curvature(c):
	try:
		from vector import Vector
	except:
		return
	c = c[:,0]
	curvature = []
	for i in xrange(len(c)-2):
		#find the angle at i+1
		frm = Vector(c[i])
		at = Vector(c[i+1])
		to = Vector(c[i+2])
		a = frm -at
		b = to -at
		angle = a.angle(b)
		curvature.append(angle)
	return curvature



def GetAnglesPolyline(polyline):
    """
    see: http://stackoverflow.com/questions/3486172/angle-between-3-points
    ported to numpy
    returns n-2 signed angles
    """

    points = polyline[:,0]
    a = points[0:-2] # all "a" points
    b = points[1:-1] # b
    c = points[2:]  # c points

    # ab =  b.x - a.x, b.y - a.y
    ab = b-a
    # cb =  b.x - c.x, b.y - c.y
    cb = b-c
    # float dot = (ab.x * cb.x + ab.y * cb.y); # dot product
    # print 'ab:',ab
    # print 'cb:',cb

    # float dot = (ab.x * cb.x + ab.y * cb.y) dot product
    # dot  = np.dot(ab,cb.T) # this is a full matrix mulitplication we only need the diagonal \
    # dot = dot.diagonal() #  because all we look for are the dotproducts of corresponding vectors (ab[n] and cb[n])
    dot = np.sum(ab * cb, axis=1) # or just do the dot product of the correspoing vectors in the first place!

    # float cross = (ab.x * cb.y - ab.y * cb.x) cross product
    cros = np.cross(ab,cb)

    # float alpha = atan2(cross, dot);
    alpha = np.arctan2(cros,dot)
    return alpha * 180. / np.pi #degrees
    # return alpha #radians




# def split_at_angle(contour, curvature, angle):
# 	segments = [[]]
# 	contour = list(contour)
# 	while len(contour)>0:
# 		segments[-1].append(list(contour.pop(0)))
# 		if len(curvature)>0:
# 			if curvature.pop(0)<angle:
# 				segments[-1] = np.array(segments[-1])
# 				segments.append([])
# 	segments[-1] = np.array(segments[-1])
# 	return segments

def split_at_angle(contour, curvature, angle):
	"""
	contour is array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )
	curvature is a n-2 list
	"""
	segments = []
	kink_index = [i for i in range(len(curvature)) if curvature[i] < angle]
	for s,e in zip([0]+kink_index,kink_index+[None]): # list of slice indecies 0,i0,i1,i2,None
		if e is not None:
			segments.append(contour[s:e+1]) #need to include the last index
		else:
			segments.append(contour[s:e])
	return segments


def extract_at_angle(contour, curvature, angle):
	"""
	contour is array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )
	curvature is a n-2 list
	"""
	kinks = []
	kink_index = [i for i in range(len(curvature)) if curvature[i] < angle]
	for s in kink_index: # list of slice indecies 0,i0,i1,i2,None
		kinks.append(contour[s+1]) # because the curvature is n-2 (1st and last are not exsistent)
	return kinks

def split_at_disc(contour, curvature, angle):
	"""
	contour is array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )
	curvature is a n-2 list
	"""
	segments = []
	mean = np.mean(curvature)
	variance = np.abs(mean-curvature)
	kink_index = [i for i in range(len(variance)) if variance[i] > angle]
	for s,e in zip([0]+kink_index,kink_index+[None]): # list of slice indecies 0,i0,i1,i2,None
		segments.append(contour[s:e])
	return segments

def convexity_defect(contour, curvature):
	"""
	contour is array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )
	curvature is a n-2 list
	"""
	kinks = []
	mean = np.mean(curvature)
	if mean>0:
		kink_index = [i for i in range(len(curvature)) if curvature[i] < 0]
	else:
		kink_index = [i for i in range(len(curvature)) if curvature[i] > 0]
	for s in kink_index: # list of slice indecies 0,i0,i1,i2,None
		kinks.append(contour[s+1]) # because the curvature is n-2 (1st and last are not exsistent)
	return kinks


def fit_ellipse(image,edges,bin_dark_img, contour_size=50,ratio=.6,target_size=20.,size_tolerance=20.):
	""" fit_ellipse:
	"""
	c_img = edges.copy()
	contours, hierarchy = cv2.findContours(c_img,
											mode=cv2.RETR_LIST,
											method=cv2.CHAIN_APPROX_NONE,offset=(0,0)) #TC89_KCOS
	# contours is a list containging array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )

	contours = [c for c in contours if c.shape[0]>contour_size]

	cv2.drawContours(image, contours, -1, (255,255,255),thickness=1,lineType=cv2.cv.CV_AA)

	# print contours

	# cv2.drawContours(image, contours, -1, (255,255,255),thickness=1)
	good_contours = contours
	# split_contours = []
	# i = 0
	# for c in contours:
	# 	curvature = np.abs(GetAnglesPolyline(c))
	# 	kink= extract_at_angle(c,curvature,150)
	# 	cv2.drawContours(image, kink, -1, (255,0,0),thickness=2)
	# 	split_contours += split_at_angle(c,curvature,150)


	# # good_contours = split_contours
	# good_contours = []
	# for c in split_contours:
	# 	i +=40
	# 	kink =  convexity_defect(c,GetAnglesPolyline(c))
	# 	cv2.drawContours(image, kink, -1, (255,0,0),thickness=1)
	# 	if c.shape[0]/float(len(kink)+1)>3 and c.shape[0]>=5:
	# 		cv2.drawContours(image, np.array([c]), -1, (i,i,i),thickness=1)
	# 		good_contours.append(c)
	# 	else:
	# 		cv2.drawContours(image, np.array([c]), -1, (255,0,0),thickness=1)

	# for c in split_contours:
	# 	kink =  convexity_defect(c,GetAnglesPolyline(c))
	# 	cv2.drawContours(image, kink, -1, (255,0,0),thickness=1)
	# 	if c.shape[0]/float(len(kink)+1)>3 and c.shape[0]>=5:
	# 		cv2.drawContours(image, c, -1, (0,255,0),thickness=1)
	# 		good_contours.append(c)


	# cv2.drawContours(image, good_contours, -1, (0,255,0),thickness=1)

	# split_contours.sort(key=lambda c: -c.shape[0])
	# for c in split_contours[:]:
	# 	if len(c)>=5:
	# 		cv2.drawContours(image, c[0:1], -1, (0,0,255),thickness=2)
	# 		cv2.drawContours(image, c[-1:], -1, (0,255,0),thickness=2)
	# 		cv2.drawContours(image, c, -1, (0,255,0),thickness=1)

	# 		# cv2.polylines(image,[c[:,0]], isClosed=False, color=(255,0,0),thickness= 1)
	# 		good_contours.append(c)

	# cv2.drawContours(image, good_contours, -1, (255,255,255),thickness=1)
	# good_contours = np.concatenate(good_contours)
	# good_contours = [good_contours]
	largest_ellipse = {'center': (None,None),
						'axes': (None, None), 'angle': None,
						'area': 0.0, 'ratio': None,
						'major': None, 'minor': None}


	shape = edges.shape
	ellipses = (cv2.fitEllipse(c) for c in good_contours)
	ellipses = (e for e in ellipses if (0 <= e[0][1] <= shape[0] and 0<= e[0][0] <= shape[1]))
	ellipses = (e for e in ellipses if bin_dark_img[e[0][1],e[0][0]])
	ellipses = ((size_deviation(e,target_size),e) for e in ellipses if is_round(e,ratio)) # roundness test
	ellipses = [(size_dif,e) for size_dif,e in ellipses if size_dif<size_tolerance ] # size closest to target size
	ellipses.sort(key=lambda e: e[0]) #sort size_deviation
	if ellipses:
		largest = ellipses[0][1]
		largest_ellipse['center'] = largest[0]
		largest_ellipse['angle'] = largest[-1]
		largest_ellipse['axes'] = largest[1]
		largest_ellipse['major'] = max(largest[1])
		largest_ellipse['minor'] = min(largest[1])
		largest_ellipse['ratio'] = largest_ellipse['minor']/largest_ellipse['major']
		return largest_ellipse,ellipses
	return None

def is_round(ellipse,ratio,tolerance=.5):
	center, (axis1,axis2), angle = ellipse

	if axis1 and axis2 and abs( ratio - min(axis2,axis1)/max(axis2,axis1)) <  tolerance:
		return True
	else:
		return False
def size_deviation(ellipse,target_size):
	center, axis, angle = ellipse
	return abs(target_size-max(axis))


def fit_ellipse_old(image,edges,bin_dark_img, contour_size=20,ratio=.6,target_size=20.,size_tolerance=20.):
	""" fit_ellipse:
	"""
	c_img = edges.copy()
	contours, hierarchy = cv2.findContours(c_img,
	                                        mode=cv2.RETR_LIST,
	                                        method=cv2.CHAIN_APPROX_NONE,offset=(0,0))

	contours = [c for c in contours if len(c) >= contour_size]
	for c in contours:
	        if convexity(c,image):
	                cv2.drawContours(image, c, -1, (255,255,255))


	best_ellipse = {'center': (None,None),
	                    'axes': (None, None), 'angle': None,
	                    'area': 0.0, 'ratio': None,
	                    'major': None, 'minor': None}


	shape = edges.shape
	ellipses = (cv2.fitEllipse(c) for c in contours if convexity(c,image))
	ellipses = (e for e in ellipses if (0 <= e[0][1] <= shape[0] and 0<= e[0][0] <= shape[1]))
	ellipses = (e for e in ellipses if bin_dark_img[e[0][1],e[0][0]])
	ellipses = ((size_deviation(e,target_size),e) for e in ellipses if is_round(e,ratio)) # roundness test
	ellipses = [(size_dif,e) for size_dif,e in ellipses if size_dif<size_tolerance ] # size closest to target size
	ellipses.sort(key=lambda e: e[0]) #sort size_deviation
	if ellipses:
		best = ellipses[0][1]
		best_ellipse['center'] = best[0]
		best_ellipse['angle'] = best[-1]
		best_ellipse['axes'] = best[1]
		best_ellipse['major'] = max(best[1])
		best_ellipse['minor'] = min(best[1])
		best_ellipse['ratio'] = best_ellipse['minor']/best_ellipse['major']
		return best_ellipse,ellipses
	return None

def is_round(ellipse,ratio,tolerance=.5):
	center, (axis1,axis2), angle = ellipse

	if axis1 and axis2 and abs( ratio - min(axis2,axis1)/max(axis2,axis1)) <  tolerance:
		return True
	else:
		return False
def size_deviation(ellipse,target_size):
	center, axis, angle = ellipse
	return abs(target_size-max(axis))

def convexity_2(contour,img=None):
	# if img is not None:
		# hull = cv2.convexHull(contour, returnPoints=1)
		# cv2.drawContours(img, hull, -1, (255,0,0))
	hull = cv2.convexHull(contour, returnPoints=0)
	if len(hull)>12:
		res = cv2.convexityDefects(contour, hull) # return: start_index, end_index, farthest_pt_index, fixpt_depth)
		if res is  None:
			return False
		if len(res)>2:
			return True
	return False


def convexity(contour,img=None):
	if img is not None:
		hull = cv2.convexHull(contour, returnPoints=1)
		# cv2.drawContours(img, hull, -1, (255,0,0))
	hull = cv2.convexHull(contour, returnPoints=0)
	if len(hull)>3:
		res = cv2.convexityDefects(contour, hull) # return: start_index, end_index, farthest_pt_index, fixpt_depth)
		if res is  None:
			return False
		if len(res)>3:
			return True
	return False


def circle_grid_old(image, circle_id=None, pattern_size=(4,11)):
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

def circle_grid_old(image, circle_id=None, pattern_size=(4,11)):
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

def circle_grid(image, pattern_size=(4,11)):
	"""Circle grid: finds an assymetric circle pattern
	- circle_id: sorted from bottom left to top right (column first)
	- If no circle_id is given, then the mean of circle positions is returned approx. center
	- If no pattern is detected, function returns None
	"""
	status, centers = cv2.findCirclesGridDefault(image, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
	if status:
		return centers
	else:
		return None



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
	# tst = []
	# for x in range(10):
	# 	tst.append(gen_pattern_grid())
	# tst = np.asarray(tst)
	# print tst.shape


	#test polyline
	#	 *-*   *
	#	 |	\  |
	#	 *	 *-*
	#	 |
	#  *-*
	print "result:", GetAnglesPolyline(np.array([[[0, 0]],[[0, 1]],[[1, 1]],[[2, 1]],[[2, 2]],[[1, 3]],[[1, 4]],[[2,4]]], dtype=np.int32))



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

