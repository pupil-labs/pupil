import os, sys
import glumpy
import OpenGL.GL as gl
import glumpy.atb as atb
import numpy as np
import cv2
from methods import capture,Temp
from time import sleep
from glob import glob


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


def player(g_pool,size):
	"""player
		- Shows 9 point calibration pattern
		- Plays a source video synchronized with world process
		- Get src videos from directory (glob)
		- Iterate through videos on each record event
	"""

	img_arr = np.zeros((size[1],size[0],3), dtype=np.uint8)
	fig = glumpy.figure((img_arr.shape[1], img_arr.shape[0]))
	image = glumpy.Image(img_arr)
	image.x, image.y = 0,0
	grid = make_grid()

	# player object
	player = Temp()
	player.play_list = glob('src_video/*')
	path_parent = os.path.dirname( os.path.abspath(sys.argv[0]))
	player.playlist = [os.path.join(path_parent, path) for path in player.play_list]
	player.captures = [capture(src) for src in player.playlist]
	print "Player found %i videos in src_video"%len(player.captures)
	player.captures =  [c for c in player.captures if c is not None]
	print "Player sucessfully loaded %i videos in src_video"%len(player.captures)
	# for c in player.captures: c.auto_rewind = False
	player.current_video = 0

	gl.glEnable( gl.GL_BLEND )
	gl.glEnable(gl.GL_POINT_SMOOTH)
	gl.glColor4f(1.0,0.0,0.0,1.0)


	def on_draw():

		if g_pool.player_refresh.wait(0.1):
			g_pool.player_refresh.clear()
			fig.clear(1.0, 1.0, 1.0, 1.0)

			if g_pool.cal9.value:
				circle_id,step = g_pool.cal9_circle_id.value,g_pool.cal9_step.value
				gl.glPushMatrix()
				gl.glTranslatef(0.0,fig.height/2,0.)
				gl.glScalef(fig.height-30,fig.height-30,0.0)
				gl.glTranslatef((float(fig.width)/float(fig.height))/2.0-10.0/16.0, -.45,0.)
				gl.glPointSize((float(fig.height)/20.0)*(1.01-(step+1)/80.0))
				gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ZERO)
				gl.glBegin(gl.GL_POINTS)
				gl.glVertex3f(grid.vertices['position'][circle_id][0],grid.vertices['position'][circle_id][1],0.5)
				gl.glEnd()
				gl.glPointSize(float(fig.height)/20.0)
				gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
				grid.draw(gl.GL_POINTS, 'pnc')
				gl.glPopMatrix()

			elif g_pool.play.value:
				s, img = player.captures[player.current_video].read_RGB()
				if s:
					img_arr[...] = img
					image.draw(x=image.x, y=image.y, z=0.0, width=fig.width, height=fig.height)
					image.update()
				else:
					player.captures[player.current_video].rewind()
					player.current_video +=1
					if player.current_video >= len(player.captures):
						player.current_video = 0
					g_pool.play.value = False


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
	fig.window.set_position(100,0)
	glumpy.show()

