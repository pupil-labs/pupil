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

	#correct to here

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

	# from wikipedia :)
	vcvc_cc_rr = v.dot(c)**2 - c.dot(c) + r**2
	if (vcvc_cc_rr < 0):
		# logger.warning("NO INTERSECTION between line and sphere")
		return None
	s1 = v.dot(c) - np.sqrt(vcvc_cc_rr)
	s2 = v.dot(c) + np.sqrt(vcvc_cc_rr)

	p1 = p + s1*v
	p2 = p + s2*v

	return p1,p2 #a line intersects a sphere at two points




################################################
if __name__ == '__main__':
	import geometry
	# #testing stuff
	huding = geometry.Line2D([5.,7.],[10.,10.])
	print huding
	huding2 = geometry.Line2D([3.,5.],[-1.,-1.])
	print intersect_2D_lines(huding, huding2)

	#testing nearest_intersect_3D
	# huding = geometry.Line3D([0.835233,3.67143,20], [-0.303085,-0.54173,-0.784008])
	# huding2 = geometry.Line3D([0, 0, 0], [-0.0843631, 0.00459802, 0.996424])
	# print nearest_intersect_3D([huding, huding2])

	# huding = Line3D([ 0.,  0, 0], [ -0.427425,-0.293268,-0.855162])
	# huding2 = Line3D([0, 0, 0], [ -0.150507,0.109365,0.982541])
	# print nearest_intersect_3D([huding, huding2])

	# huding = Line3D([ 0.837533,  3.67339, 20], [ -0.427425,-0.293268,-0.855162])
	# huding2 = Line3D([0, 0, 0], [ -0.150507,0.109365,0.982541])
	# print nearest_intersect_3D([huding, huding2])

	# huding = geometry.Line3D([0.835451, 3.67313, 20],[0.0687859, -0.42695, -0.901655])
	# huding2 = geometry.Line3D([0, 0, 0],[0.0852856, 0.0771611, 0.993364])
	# print [  1.25055031,   1.13423371 , 14.63264343]
	# print nearest_intersect_3D([huding, huding2])

	#testing sphere_intersect
	# l1 = geometry.Line3D((0,0,0),(0.0471124,0.0831234, 0.995425))
	# s1 = geometry.Sphere((0.671767,3.56112,20),6.41521)
	# sphere_intersect(l1,s1)[0]
	# print sphere_intersect(l1,s1)


	print "done"