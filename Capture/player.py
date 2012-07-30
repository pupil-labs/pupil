import glumpy 
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glumpy.atb as atb
from ctypes import *
import numpy as np
import cv2
from time import sleep

def make_grid(dim=(11,4)):
	"""
	this function generates the struckture for an asymetrical circle grid
	It returns a Vertext Buffer Object that is used by glumpy to draw it in
	the opengl Window.
	"""
	x,y = range(dim[0]),range(dim[1])
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


class Temp(object):
	"""Temp class to make objects"""
	def __init__(self):
		pass

def player(g_pool):
	"""player
		- Shows 9 point calibration pattern
		- Plays a source video synchronized with world process
		- Get src videos from directory (glob)
		- Iterate through videos on each record event
	"""
	capture = Temp()
	capture.remaining_frames = 0
	capture.cap = None

	# Get image array from queue, initialize glumpy, map img_arr to opengl texture 
	img_arr = np.zeros((720,1280,3), dtype=np.uint8)
	fig = glumpy.figure((img_arr.shape[1], img_arr.shape[0]))
	image = glumpy.Image(img_arr)
	image.x, image.y = 0,0
	grid = make_grid()

	def on_draw():
		fig.clear(1.0, 1.0, 1.0, 1.0)
		
		if g_pool.player_pipe_new.wait(0.3):
			command = g_pool.player_rx.recv()
			g_pool.player_pipe_new.clear()

			if command == 'calibrate':
				circle_id,step = g_pool.player_rx.recv()
				gl.glEnable(gl.GL_POINT_SMOOTH)
				gl.glPushMatrix()
				gl.glTranslatef(0.0,fig.height/2,0.)
				gl.glScalef(fig.height-30,fig.height-30,0.0)
				gl.glTranslatef((float(fig.width)/float(fig.height))/2.0-10.0/16.0, -.45,0.)
				gl.glPointSize((float(fig.height)/20.0)*(1.1-(step+1)/80.0))
				gl.glColor4f(1.0,0.0,0.0,1.0)
				gl.glBegin(gl.GL_POINTS)
				gl.glVertex3f(grid.vertices['position'][circle_id][0],grid.vertices['position'][circle_id][1],0.5)
				gl.glEnd()
				gl.glPointSize(float(fig.height)/20.0)
				grid.draw(gl.GL_POINTS, 'pnc')
				gl.glPopMatrix()

			elif command == 'load_video':
				src_id = g_pool.player_rx.recv() # path to video
				capture.cap = cv2.VideoCapture(src_id)
				# subtract last 10 frames so player process does not get errors for none type in cv2 grab
				# capture.remaining_frames = capture.cap.get(7)-10 

			elif command == 'next_frame':
				capture.remaining_frames -= 1	
				if capture.remaining_frames:
					status, img = capture.cap.read()
					if status:
						img_arr[...] = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
						g_pool.player_rx.send(True)
					else:
						g_pool.player_rx.send(False)

					image.draw(x=image.x, y=image.y, z=0.0, 
								width=fig.width, height=fig.height)
					
					image.update()	
				else:
					g_pool.player_rx.send(False)	
			else:
				#do nothing 
				pass
		
		if g_pool.quit.value:
			print "Player Process closing from global or atb"
			fig.window.stop()

	def on_close():
		g_pool.quit.value = True
		print "Player Process closed from window"			

	def on_idle(dt):
		fig.redraw()

	fig.window.push_handlers(on_idle)
	fig.window.push_handlers(on_draw)	
	fig.window.push_handlers(on_close)	
	fig.window.set_title("Player")
	fig.window.set_position(0,0)	
	glumpy.show() 	

