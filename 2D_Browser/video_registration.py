import numpy as np
import cv2
import cv2.cv as cv
from functools import partial
from scipy import linalg


"""
Matching algorithms for two objects

"""

FLANN_INDEX_KDTREE = 1	# bug: flann enums are missing

flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
					trees = 4)
					

def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )

def match_bruteforce(desc1, desc2, r_threshold = 0.75):
	res = []
	for i in xrange(len(desc1)):
		dist = anorm( desc2 - desc1[i] )
		n1, n2 = dist.argsort()[:2]
		r = dist[n1] / dist[n2]
		if r < r_threshold:
			res.append((i, n1))
	return np.array(res)


def match_flann(desc1, desc2, r_threshold = 0.6):
	flann = cv2.flann_Index(desc2, flann_params)
	idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
	mask = dist[:,0] / dist[:,1] < r_threshold
	idx1 = np.arange(len(desc1))
	pairs = np.int32( zip(idx1, idx2[:,0]) )
	return pairs[mask]

def draw_match_overlay(img1, img2, H):
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	
	overlay = cv2.warpPerspective(img2, H, (w1,h1))

	# populate the vis array with pixel values from both images
	res = cv2.addWeighted(img1, 0.5, overlay, 0.5, 0.0)
	#vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR) # convert to color image
	return res

def draw_match(img1, img2, p1, p2, status = None, H = None):
	"""
		Function to draw the match between objects
		Parameters:
			img1, img2: source images
			p1, p2: matching points in the images determined by matching function
			status: which points were used for the homography transformations (binary mask)
			H: Homography matrix - which correlates two planes (of the images)
	"""
	
	# first we need to get the size of both images in height, width
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	# then we set the output image as the maximum size of both images combined
	# here the output image is just a black rectanlge with sizes to accomodate the picture
	vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
	# populate the vis array with pixel values from both images
	vis[:h1, :w1] = img1 # rows, columns of imm1 (starting from the left)
	vis[:h2, w1:w1+w2] = img2 # rows, columns (starting from the last col. of img1)
	vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR) # convert to color image

	if H is not None:
		# we want to know where img1 is located relative to img2
		# so we make the corners = img1 boundaries 
		# (if we wanted to find a ROI within img2, then we would populate this with variables)
		corners = np.float32([
							[0, 0], 
							[w1, 0], 
							[w1, h1], 
							[0, h1]])
		# find the relationship between the two images 
		# through the Homography matrix
		# corners are transfmrned via Homography matrix H and then shifted by w1 
		# returned as int32 because opencv doesn't like points as floats for drawing
		corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
		cv2.polylines(vis, [corners], True, (255, 255, 255)) # draw the border of the img1 on vis (img1+img2)

	
	if status is None:
		status = np.ones(len(p1), np.bool_)
	green = (0, 255, 0)
	red = (0, 0, 255)
	for (x1, y1), (x2, y2), inlier in zip(np.int32(p1), np.int32(p2), status):
		col = [red, green][inlier]
		if inlier:
			# draw green lines between matches in the inlier set
			cv2.line(vis, (x1, y1), (x2+w1, y2), col)
			cv2.circle(vis, (x1, y1), 2, col, -1)
			cv2.circle(vis, (x2+w1, y2), 2, col, -1)
		else:
			# draw the red x marks for outliers
			r = 2
			thickness = 3
			cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
			cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
			cv2.line(vis, (x2+w1-r, y2-r), (x2+w1+r, y2+r), col, thickness)
			cv2.line(vis, (x2+w1-r, y2+r), (x2+w1+r, y2-r), col, thickness)
	return vis


def main():	

	fn1 = '/Volumes/HD_Two/Users/Will/Desktop/02.png' # world video
	fn2 = '/Volumes/HD_Two/Users/Will/Desktop/01.png' # src video 


	img1 = cv2.imread(fn1, 0)
	img2 = cv2.imread(fn2, 0)
	
	img1c = cv2.imread(fn1, 1)
	img2c = cv2.imread(fn2, 1)
	
	# img1 = cv2.resize(img1,(0,0),fx=0.25,fy=0.25,interpolation=3)  
	# img2 = cv2.resize(img2,(0,0),fx=0.25,fy=0.25,interpolation=3)  
	# img1c = cv2.resize(img1c,(0,0),fx=0.25,fy=0.25,interpolation=3)  
	# img2c = cv2.resize(img2c,(0,0),fx=0.25,fy=0.25,interpolation=3)  


	surf = cv2.SURF(500)
	kp1, desc1 = surf.detect(img1, None, False)
	kp2, desc2 = surf.detect(img2, None, False)
	desc1.shape = (-1, surf.descriptorSize())
	desc2.shape = (-1, surf.descriptorSize())
	print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

	def match_and_draw(match_fn, r_threshold):
		"""
			Match a set of descriptors using a supplied matching method:
			Parameters:
				- match: a matching function (in this case, bruteforce and flann - see above)
				- r_threshold: radius threshold?
		"""
		# call the matching function passing descriptor vectors
		# m is a list of index values for matched keypoints
		m = match_fn(desc1, desc2, r_threshold) 
		matched_p1 = np.array([kp1[i].pt for i, j in m], np.float32) # get img1 keypoints from match index
		matched_p2 = np.array([kp2[j].pt for i, j in m], np.float32) # get img2 keypoints from match index

		
		H, status = cv2.findHomography(matched_p1, matched_p2, cv2.RANSAC, 5.0) # find homography matrix
		
		# status is a binary mask corresponding to points used from matched points?
		print '%d / %d	inliers/matched' % (np.sum(status), len(status))
		
		res = draw_match_overlay(img2, img1, H)
		vis = draw_match(img1, img2, matched_p1, matched_p2, status, H)
		return vis, res

	print 'bruteforce match:\n',
	vis_brute, r1 = match_and_draw( match_bruteforce, 1.0 ) #.75
	print 'flann match:\n',
	vis_flann, r2 = match_and_draw( match_flann, .6 ) # .6 flann tends to find more distant second
												   # neighbours, so r_threshold is decreased

	cv2.imshow('find_obj SURF', vis_brute)
	cv2.imshow('find_obj SURF flann', vis_flann)
	cv2.imshow('overlay flann', r2)
	cv2.imshow('overlay bf', r1)

	cv.MoveWindow('find_obj SURF', 5, 10)
	cv.MoveWindow('find_obj SURF flann', 830, 10)
	cv.MoveWindow('overlay bf', 5, 365)
	cv.MoveWindow('overlay flann', 830, 365)
	
	
	
	cv2.waitKey()
	
if __name__ == '__main__':
	main()
	#cProfile.run('main()')