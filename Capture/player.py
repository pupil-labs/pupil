import glumpy 
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glumpy.atb as atb
from ctypes import *
import numpy as np

def make_grid(dim=(11,4)):
	x = range(dim[0])
	y = range(dim[1])

	p = np.array([[[s,i] for s in x] for i in y], dtype=np.float32)
	p[:,1::2,1] += 0.5

	# width of pattern is 1
	# height is scaled accordingly to preserve aspect ratio
	p[:,:,0] /= dim[1]*2
	p[:,:,1] /= dim[1]

	p = np.reshape(p, (-1,2), 'F')

	n = (1., 1., 1.)
	c= (0.0, 0.0, 0.0, 1.0)
	
	vertices = [((x,y,0.0), n, c) for x,y in p]

	vertices  = np.array(vertices, 
				dtype = [('position','f4',3), ('normal','f4',3), ('color','f4',4)] )
	
	grid = glumpy.graphics.VertexBuffer(vertices)
	return grid

def display_circle_grid(circle_id, step):
	gl.glEnable(gl.GL_POINT_SMOOTH)
	gl.glPushMatrix()
	gl.glTranslatef(0.0,fig.height/2,0.)
	gl.glScalef(fig.height,fig.height,0.0)
	gl.glTranslatef((float(fig.width)/float(fig.height))/2.0-10.0/16.0, 
					-.45,
					0.)

	gl.glPointSize((float(fig.height)/10.0)/(step+1))
	gl.glColor4f(1.0,0.0,0.0,1.0)
	gl.glBegin(gl.GL_POINTS)
	gl.glVertex3f(grid.vertices['position'][circle_id][0],grid.vertices['position'][circle_id][1],0.5)
	gl.glEnd()

	gl.glPointSize(float(fig.height)/10.0)
	grid.draw(gl.GL_POINTS, 'pnc')

	gl.glPopMatrix()

class Temp(object):
	"""Temp class to make objects"""
	def __init__(self):
		pass

def player(pipe):
	"""player
		- Shows 9 point calibration pattern
		- Plays a source video synchronized with world process
		- Get src videos from directory (glob)
		- Iterate through videos on each record event
	"""
	capture = Temp()
	capture.remaining_frames = 0

	# Get image array from queue, initialize glumpy, map img_arr to opengl texture 
	img_arr = np.zeros((720,1280,3), dtype=np.uint8)

	fig = glumpy.figure((img_arr.shape[1], img_arr.shape[0]))
	image = glumpy.Image(img_arr)
	image.x, image.y = 0,0

	grid = make_grid()


	def on_draw():
		fig.clear(1.0, 1.0, 1.0, 1.0)
		command = pipe.recv()

		if command == 'calibrate':
			circle_id,step = pipe.recv()
			gl.glEnable(gl.GL_POINT_SMOOTH)
			gl.glPushMatrix()
			gl.glTranslatef(0.0,fig.height/2,0.)
			gl.glScalef(fig.height,fig.height,0.0)
			gl.glTranslatef((float(fig.width)/float(fig.height))/2.0-10.0/16.0, 
							-.45,
							0.)

			gl.glPointSize((float(fig.height)/10.0)*(1-step/40.0))
			gl.glColor4f(1.0,0.0,0.0,1.0)
			gl.glBegin(gl.GL_POINTS)
			gl.glVertex3f(grid.vertices['position'][circle_id][0],grid.vertices['position'][circle_id][1],0.5)
			gl.glEnd()

			gl.glPointSize(float(fig.height)/10.0)
			grid.draw(gl.GL_POINTS, 'pnc')

			gl.glPopMatrix()

		elif command == 'load_video':
			src_id = pipe.recv() # path to video
			cap = cv2.VideoCapture(src_id)
			capture.remaining_frames = cap.get(7)

		elif command == 'next_frame':
			capture.remaining_frames -= 1	
			if capture.remaining_frames:
				status, img = cap.read()
				pipe.send(True)
				img_arr[...] = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

				image.draw(x=image.x, y=image.y, z=0.0, 
							width=fig.width, height=fig.height)
				
				image.update()	
			else:
				pipe.send(False)	


	def on_idle(dt):
		fig.redraw()

	fig.window.push_handlers(on_idle)
	fig.window.push_handlers(on_draw)	
	fig.window.set_title("Player")
	fig.window.set_position(0,0)	
	glumpy.show() 	

