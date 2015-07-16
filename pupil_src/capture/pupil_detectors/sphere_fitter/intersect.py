"""
	Andrew Xia playing around with porting c++ code to python
	Moving the geometry/intersect.py to this file (see github for version history)
	line2D and line3D, which originally were in intersect.py, are now in geometry.py
	Created July 2 2015

"""

import numpy as np
import geometry
import logging
logger = logging.getLogger(__name__)

def intersect_2D_lines(line1,line2):
	#not working!
	#finds intersection of 2 lines in 2D. the original intersect() function
	#line1 and line2 should be geometry.line2D() class
	x1 = line1.origin[0]
	y1 = line1.origin[1]
	x2 = line1.origin[0] + line1.direction[0]
	y2 = line1.origin[1] + line1.direction[1]
	x3 = line2.origin[0]
	y3 = line2.origin[1]
	x4 = line2.origin[0] + line2.direction[0]
	y4 = line2.origin[1] + line2.direction[1]

	denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
	if (abs(denom) <= 10e-15 ):
		# rounding errors by python since it isn't perfect.
		# though this is sketchy math :P
		denom = 0

	if (denom == 0): #edge case
		#they have the same slope
		if (line1.direction[0] == 0):
			#vertical line, give it some big value
			slope = None
		else:
			slope = line1.direction[1]/line1.direction[0]
		if (x3 == x1):
			x1 = x2 #switch vars
			x3 = x4
			if (x3 == x4):
				if (y3 == y1):
					raise ValueError("Inputs are same lines, here is one of many points of intersection")
				else:
					raise ValueError("Parallel Lines, no intersect")
		if ((y3-y1)/(x3-x1) == slope):
			#is the same line
			print "Inputs are same lines, here is one of many points of intersection"
			return x1,y1
		else:
			raise ValueError("Parallel Lines, no intersect")
	else:
		#there exists an intersection
		px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*x4 - y3*x4))/denom
		py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*x4 - y3*x4))/denom
		return px,py

def nearest_intersect_3D(lines):
	#finds the learest intersection of many lines (which may not be a real intersection)
	#the original nearest_intersect(const Range& lines) function
	#each element in array lines should be geometry.line3D() class
	A = np.zeros((3,3))
	b = np.zeros(3)
	Ivv = [] #vector of matrices
	for line in lines:
		vi = line.direction.reshape(3,1)
		pi = line.origin
		Ivivi = np.identity(3) - vi.dot(vi.T)
		Ivv.append(Ivivi)
		A += Ivivi
		b += pi.dot(Ivivi)

	# x = A.partialPivLu.solve(b)
	#not sure if partialPivLu actually does anything...
	return np.linalg.solve(A,b)

def nearest_intersect_2D(lines):
	#finds the learest intersection of many lines (which may not be a real intersection)
	#the original nearest_intersect(const Range& lines) function
	#each element in array lines should be geometry.line2D() class
	A = np.zeros((2,2))
	b = np.zeros((2))
	Ivv = [] #vector of matrices
	for line in lines:
		vi = line.direction.reshape(2,1)
		pi = line.origin
		Ivivi = np.identity(2) - vi.dot(vi.T)
		Ivv.append(Ivivi)

		A += Ivivi
		b += pi.dot(Ivivi)
	# x = A.partialPivLu.solve(b) #WHAT?
	#not sure if partialPivLu actually does anything...

	return np.linalg.solve(A,b)

def sphere_intersect(line,sphere):
	#intersection between a line and a sphere, originally called intersect(line,sphere)
	#line should be geometry.line3D() class, sphere is geometry.sphere() class
	v = line.direction
	p = line.origin #put p at origin
	c = sphere.center - p
	r = sphere.radius

	vcvc_cc_rr = v.dot(c)**2 - c.dot(c) + r**2 # from wikipedia :)
	if (vcvc_cc_rr < 0):
		# logger.warning("NO INTERSECTION between line and sphere")
		return None
	s1 = v.dot(c) - np.sqrt(vcvc_cc_rr)
	s2 = v.dot(c) + np.sqrt(vcvc_cc_rr)

	p1 = p + s1*v
	p2 = p + s2*v

	return p1,p2 #a line intersects a sphere at two points

def get_sphere_intersect_params(line,sphere):
	point = sphere_intersect(line,sphere)
	if point == None:
		# logger.warning("NO INTERSECTION between line and sphere")
		return None
	normal = point[0] - sphere.center #take closer point of two
	theta = np.arctan2(normal[1],normal[0])
	psi = np.arctan2(np.sqrt(normal[0]**2 + normal[1]**2),normal[2])
	return geometry.PupilParams(theta, psi, sphere.radius)

def residual_distance_intersect_2D(p, lines):
	#used to calculate residual distance
	x3,y3 = p
	x1 = []
	y1 = []
	dx21 = []
	dy21 = []
	for line in lines:
		x1.append(line.origin[0])
		y1.append(line.origin[1])
		dx21.append(line.direction[0])
		dy21.append(line.direction[1])
	x1 = np.asarray(x1)
	y1 = np.asarray(y1)
	dx21 = np.asarray(dx21)
	dy21 = np.asarray(dy21)

	lensq21 = dx21*dx21 + dy21*dy21

	u = (x3-x1)*dx21 + (y3-y1)*dy21

	u = u / lensq21
	x = x1+ u * dx21
	y = y1+ u * dy21
	dx30 = x3-x
	dy30 = y3-y
	return np.sqrt( dx30**2 + dy30**2 )

def residual_distance_intersect_3D(point,lines):
	x3,y3,z3 = point
	x1 = []
	y1 = []
	z1 = []
	dx1 = []
	dy1 = []
	dz1 = []
	for line in lines:
		x1.append(line.origin[0])
		y1.append(line.origin[1])
		z1.append(line.origin[2])
		dx1.append(line.direction[0])
		dy1.append(line.direction[1])
		dz1.append(line.direction[2])
	x1 = np.asarray(x1)
	y1 = np.asarray(y1)
	z1 = np.asarray(z1)
	dx1 = np.asarray(dx1)
	dy1 = np.asarray(dy1)
	dz1 = np.asarray(dz1)

	lensq21 = dx1**2 + dy1**2 + dz1**2
	u = ((x3-x1)*dx1 + (y3-y1)*dy1 + (z3 - z1)*dz1) / lensq21
	x = x1+ u * dx1
	y = y1+ u * dy1
	z = z1+ u * dz1
	dx3 = x3-x
	dy3 = y3-y
	dz3 = z3-z
	return np.sqrt(dx3**2 + dy3**2 + dz3**2)

################################################
if __name__ == '__main__':
	import geometry
	# #testing stuff
	# huding = geometry.Line2D([5.,7.],[10.,10.])
	# print huding
	# huding2 = geometry.Line2D([3.,5.],[-1.,-1.])
	# print intersect_2D_lines(huding, huding2)

	lines = []
	lines.append(geometry.Line2D([-146.26909863,-99.45170178],[ 0.92645834,0.37639732]))
	lines.append(geometry.Line2D([-133.13767081,-97.685709] ,[ 0.91785761,0.39690981]))
	lines.append(geometry.Line2D([-92.31869627,-8.12721281] ,[ 0.65350508,0.75692213]))
	lines.append(geometry.Line2D([-73.57488028,-65.2793859] ,[ 0.7564165 ,0.65409027]))
	hudong = nearest_intersect_2D(lines)
	print hudong
	print np.mean(residual_distance_intersect_2D(hudong, lines))

	lines = []
	lines.append(geometry.Line3D([3,3,5],[1,0,0]))
	lines.append(geometry.Line3D([3,3,3],[0,1,0]))
	hudong =  nearest_intersect_3D(lines)
	print hudong
	print np.mean(residual_distance_intersect_3D(hudong, lines))

