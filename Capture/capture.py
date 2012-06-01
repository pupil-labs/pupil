import glumpy 
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np 
import glumpy.atb as atb
import cv2
from utilities.usb_camera_interface import cam_interface
from ctypes import *
from time import sleep
from multiprocessing import Process, Queue
from threading import Thread



t = 0.1

stop = False

def generate(q,cam,id,size):
	dropped = 1000
	if id == 0:
		cam = cam_interface()
		buffer = np.zeros(size, dtype=np.uint8) #this should always be a multiple of 4
		cam.aptina_setWindowSize(cam.id0,(size[1],size[0]))
		cam.aptina_setWindowPosition(cam.id0,(1,4))
		cam.aptina_LED_control(cam.id0,Disable = 0,Invert =0)
		cam.aptina_AEC_AGC(cam.id0,1,1)
		cam.aptina_HDR(cam.id0,0)
		q.put(size)
		while 1:
			if cam.get_frame(id,buffer): #returns 1 on sucess
				try:
					# print cam.aptina_get_Gain(cam.id0)
					q.put(buffer,False)
				except:
					dropped -=1
			if not dropped:
				break
		cam.release()

def get_fps():
	return 1/t

class container():
	"""docstring for container"""
	def __init__(self):
		pass
		

def np2img(q0,q1):
	q = q0
	exit = c_bool(0)
	info = container()
	info.shape = q.get()
	img_arr = q.get()
	img_arr.shape=info.shape

	# COLOR_BAYER_BG2RGB = 48L
	# COLOR_BAYER_BG2RGB_VNG = 64L
	# COLOR_BAYER_GB2RGB = 49L
	# COLOR_BAYER_GB2RGB_VNG = 65L
	# COLOR_BAYER_GR2RGB = 47L
	# COLOR_BAYER_GR2RGB_VNG = 63L
	# COLOR_BAYER_RG2RGB = 46L
	# COLOR_BAYER_RG2RGB_VNG = 62L
	code = cv2.COLOR_BAYER_RG2RGB
	img_arr = cv2.cvtColor(img_arr,code)
	fig = glumpy.figure((img_arr.shape[1], img_arr.shape[0]))
	atb.init()

	bar = atb.Bar(name="Controls", label="Controls",
			help="Scene controls", color=(50,50,50), alpha=50,
			text='light', position=(10, 10), size=(200, 440))

	bar.add_var("World/world_fps", step=0.01, getter=get_fps)
	bar.add_var("World/exit", exit)

	image = glumpy.Image(img_arr)
	image.x, image.y = 0,0

	def on_draw():
		fig.clear()
		image.draw(x=image.x, y=image.y, z=0.0, 
					width=fig.width, height=fig.height)


	def on_idle(dt):
		global t
		t = t+0.1*(dt-t)
		tmp = q.get()
		tmp.shape =info.shape

		tmp = cv2.cvtColor(tmp,code)
		# tmp = np.dstack((tmp,tmp,tmp))
		img_arr[...] = tmp

		image.update()
		fig.redraw()
		if exit:
			fig.window.stop()

	fig.window.push_handlers(on_idle)
	fig.window.push_handlers(atb.glumpy.Handlers(fig.window))
	fig.window.push_handlers(on_draw)	
	glumpy.show() 

if __name__ == '__main__':
	cam = None
	q0 = Queue(3)
	q1 = Queue(3)
	# g0 = Process(target=generate, args=(q0,cam,0,(480,752)))
	# g1 = Process(target=generate, args=(q1,cam,1,(480,640)))
	c = Process(target=np2img, args=(q0,q1))
	c.start()
	generate(q0,cam,0,(480,752))
	# g0.start()
	# g1.start()
	c.join()
	# g0.join()
	# g1.join()





