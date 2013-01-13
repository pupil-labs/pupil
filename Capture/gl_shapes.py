import glumpy
import OpenGL.GL as gl
import numpy as np


class Point():
	"""simple opengl point class"""
	def __init__(self, center=(0.0,0.0), color=(255,0,0,0.5)):
		p = (center[0], center[1], 1.0)
		n = (0,1,0)
		c = color
		self.vertices = np.array([(p,n,c)],
				dtype=[('position','f4',3), ('normal','f4',3), ('color','f4', 4)] )
		self.vbo = glumpy.graphics.VertexBuffer(self.vertices)
		self.found = False

	def update(self, new_center):
		self.vbo.vertices['position'] = (new_center[0], new_center[1], 1.0)
		self.vbo.upload()
		self.found = True


	def draw(self):
		if self.found:
			gl.glDisable(gl.GL_TEXTURE_2D)
			gl.glEnable(gl.GL_POINT_SMOOTH)
			gl.glPointSize(20.0)
			self.vbo.draw(gl.GL_POINTS, 'pnc')
			gl.glEnable(gl.GL_TEXTURE_2D)
			self.found = False
		else:
			pass


class Ellipse():
	"""Simple Ellipse opengl drawing"""
	def __init__(self, center=(0.0,0.0), width=0.0, height=0.0,
				angle=0.0, color=(255,0,0,0.5), step=20):
		self.center = center
		self.width = width
		self.height = height
		self.angle = np.radians(angle)
		self.color = color
		self.step = step

		w2 = 0.5*self.width
		h2 = 0.5*self.height
		c = np.array([self.center[0], self.center[1], 1.0])
		a = np.radians(self.angle)
		r = rotate_z(self.angle)
		p = np.array([[w2*np.cos(np.radians(t)),h2*np.sin(np.radians(t)),c[2], 1.0] \
					for t in np.arange(0,360,self.step)])
		p = np.dot(p,r)[:,:-1]+c

		n= (0.0, 0.0, 1.0)
		c= self.color
		v = [(list(p[i]),n,c) for i in xrange(0,p.shape[0])]

		self.vertices = np.asarray(v, dtype=[('position','f4',3), ('normal','f4',3), ('color','f4',4)] )
		self.vbo = glumpy.graphics.VertexBuffer(self.vertices)

	def update(self, center, ellipse):
		w2 = 0.5*ellipse['axes'][0]
		h2 = 0.5*ellipse['axes'][1]
		#c = np.array([ellipse['center'][0],ellipse['center'][1],1.0])
		c = np.array([center[0],center[1],1.0])
		a = np.radians(ellipse['angle'])
		r = rotate_z(a)
		p = np.array([[w2*np.cos(np.radians(t)),h2*np.sin(np.radians(t)),c[2], 1.0] \
					for t in np.arange(0,360,self.step)])
		p = tuple(np.dot(p,r)[:,:-1]+c)
		self.vbo.vertices['position'] = p
		self.vbo.upload()

	def draw(self):
		gl.glDisable(gl.GL_TEXTURE_2D)
		gl.glEnable (gl.GL_LINE_SMOOTH)
		gl.glLineWidth(2.0)
		self.vbo.draw(gl.GL_LINE_LOOP, 'pnc')
		gl.glEnable(gl.GL_TEXTURE_2D)


def rectangle(c, w, h, a, color=(1.0,0.0,0.0,0.5)):
	w2 = 0.5*w
	h2 = 0.5*h
	a = np.radians(a)
	r = rotate_z(a)
	p=np.array([ [-w2, -h2, c[2]], [-w2, h2, c[2]],
				 [w2, h2, c[2]], [w2, -h2, c[2]] ])
	print p.shape[0]

	p[0] =np.dot(np.hstack((p[0],1.)), r)[:-1]+c
	p[1] =np.dot(np.hstack((p[1],1.)), r)[:-1]+c
	p[2] =np.dot(np.hstack((p[2],1.)), r)[:-1]+c
	p[3] =np.dot(np.hstack((p[3],1.)), r)[:-1]+c
	n= (0.0, 0.0, 1.0)
	c= color
	vertices = np.array([ (p[0], n, c), (p[1], n, c), (p[2], n, c), (p[3], n, c)],
						dtype = [('position','f4',3), ('normal','f4',3), ('color','f4',4)] )
	return glumpy.graphics.VertexBuffer(vertices)


def rotate_z(a):
	rz = np.array([	[np.cos(a),-np.sin(a),	0,	0],
					[np.sin(a),	np.cos(a),	0,	0],
					[0, 		0, 			1, 	0],
					[0, 		0, 			0, 	1] ])
	return rz





