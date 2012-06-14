import numpy as np
from ctypes import *
from time import sleep
import os,sys

class cam_interface(object):
	"""docstring for cam_interface"""
	def __init__(self,id0 = 0x90,id1 = 0x42):
	
		#cytypes binding setup for libusb file
		Path = os.path.dirname(os.path.abspath(sys.argv[0]))
		DLL_location = os.path.join(Path,'utilities/c/lsusb.so')
		# DLL_location = os.path.join(Path,'c/lsusb.so')

		self.lib_usb = CDLL(DLL_location)

		self.libusb_device_handle = c_void_p
		self.lib_usb.open.argtypes =[POINTER(self.libusb_device_handle)]
		self.lib_usb.close.argtypes =[POINTER(self.libusb_device_handle)]
		self.lib_usb.read.argtypes = [POINTER(self.libusb_device_handle), np.ctypeslib.ndpointer(dtype = np.uint8),c_int,c_byte]
		self.lib_usb.write.argtypes = [POINTER(self.libusb_device_handle), np.ctypeslib.ndpointer(dtype = np.uint8),c_int,c_byte]
		self.handle = self.libusb_device_handle()
		self.lib_usb.open(byref(self.handle))
		if self.handle.value is None:
			raise Exception("ERROR Device could not be opended")
			
		#print "LIBUSB DEVICE POINTER as seen in Python", hex(self.handle.value)


		#id s correspond to cameras connected to the xmos capture system
		self.id0, self.id1 = id0, id1  #I2C address used as id
		self.id0_ep, self.id1_ep = 0x82,0x83 #USB EP used to receive image data

		self.aptina_init(id0)

	def release(self):
		self.lib_usb.close(byref(self.handle))

	def cam_cmd(self, action,id,reg=0x00,msg=0x00):
		out_buffer= np.zeros(4, dtype=np.uint8)
		in_buffer = np.zeros(4, dtype=np.uint8)
		out_buffer[:] = ord(action),id,reg,msg
		#in_buffer[:] =out_buffer[:]
		self.lib_usb.write(byref(self.handle),out_buffer, out_buffer.shape[0],0x01)
		self.lib_usb.read(byref(self.handle),in_buffer, in_buffer.shape[0],0x81)
		if in_buffer[0] != 79: # chr(79) = O for ok or output
			print  "Error CAM CMD!"
			print "Return string: ", chr(in_buffer[0]), hex(in_buffer[1]), hex(in_buffer[2]), hex(in_buffer[3])
		if action=='R':
			return in_buffer[3]


	def write8(self,id,reg,msg):
		self.cam_cmd('W',id,reg,msg)

	def read8(self,id,reg):
		return self.cam_cmd('R',id,reg)

	def write16(self,id,reg,msg16):
		APTINA_LOW_REG = 0xF0
		low = 0x000000FF & msg16
		hi = (0x0000FF00 & msg16)>>8
		self.cam_cmd('W',id,reg,hi)
		self.cam_cmd('W',id,APTINA_LOW_REG,low)

	def read16(self,id,reg):
		APTINA_LOW_REG = 0xF0
		hi = self.cam_cmd('R',id,reg)
		low = self.cam_cmd('R',id,APTINA_LOW_REG)
		return (hi<<8)|low


	def next_frame(self,id):
		action="N"
		out_buffer= np.zeros(4, dtype=np.uint8)
		out_buffer[:2] = ord(action), id
		self.lib_usb.write(byref(self.handle),out_buffer, out_buffer.shape[0],0x01)


	def get_frame(self,id,arr):
		"""
		fill a numpy array with data comming from EP 0x82 
		"""

		if id == 0:
			ep = self.id0_ep
			id = self.id0
		elif id == 1:
			ep = self.id1_ep
			id = self.id1
		else:
			raise "ERROR: Cannot get frame, id is not valid"
			return 
		if arr.dtype!=np.uint8:
			raise "ERROR: Cannot get frame, array is not uint8!"
			return 
		
		shape=arr.shape
		arr.shape = -1
		self.next_frame(id) #request frame
		
		if self.lib_usb.read(byref(self.handle),arr, arr.shape[0],ep) ==-1:
			print "USB ERROR getting frame"
			arr.shape = shape
			return False
		arr.shape = shape
		return True



	def aptina_init(self,id):
		"""
		convienece fn to initialize the Aptina MT9V024 CMOS camera
		max framerate,
		max resolution
		raw beyer
		rowwise black level calibration
		LVDS output
		"""
		print "CHIP ID: ", hex(self.read16(id,0x00))

		"""
		LVDS startup sequence:
        1. Power-up sensor
		2. Enable LVDS serial data_h out driver (set R179[4]= 0)
		3. De-assert LVDS power-down (set R177[1] = 0)
		4. Issue a soft RESET R12[0] = 1 followed by R12[0] = 0
		5. Force sync patterns for the deserializer to lock (set R181[0] = 1)
		6. Stop applying sync patterns (set R181[0] = 0)
		"""
		#these registers are 
		self.write16(id,179,0) #enable lvds out
		self.write16(id,177,0) #deassert lvds power down
		self.aptina_reset(id) 

		self.write16(id,181,1) #start lvds sync pattern
		self.write16(id,0x05,300) # Hblank 5E is minimum 61-1023 94 is default
		self.write16(id,0x06,100) # Vblank 2-3228845 45 is default
		self.write16(id,0x70,1) # activate row-wise black level calibration
		self.write16(id,181,0) # stop lvds sync pattern
		self.write16(id,13,0x310) # bit 4 = row flip 
		self.write16(id,194,192) #192 anti eclipse  enable  # 64 disable : this is for looking into the sun
		self.aptina_reset(id)

	def aptina_reset(self,id):
		"""
		issue a soft reset: 
		aborts the current frame and starts a new one
		register settings  stay the same
		Attention: this causes a temporary lvds lock loss
		"""
		self.write16(id,12,1)
		sleep(.2)


	def aptina_HDR(self,id,enable):
		state = enable
		self.write16(id,15,state)
		pass
	def aptina_setWindowSize(self,id,(width,height)):
		"""
		set Window size
		"""
		self.write16(id,0x03,height)
		self.write16(id,0x04,width)


	def aptina_setWindowPosition(self,id,(column_start,row_start)):
		"""
		set Window top left corner
		"""
		column_start = min(max(column_start,1),752)
		row_start = min(max(row_start,4),482)
		self.write16(id,0x01,column_start)
		self.write16(id,0x02,row_start)

	def aptina_AEC_AGC(self,id,AEC,AGC):
		"""
		BIT 0:  Automatic Exposure Control
		BIT 1: Automatic Gain Control
		"""
		AGC = AGC<<1
		state = AEC | AGC
		self.write16(id,0xAF ,state)

	def aptina_get_Gain(self,id):
		return self.read16(id,168)

	def aptina_get_Exposure(self,id):
		return self.read16(id,167)


	def aptina_LED_control(self,id,Disable,Invert):
		"""
		Disable LED_OUT output.
		When this bit is cleared, 
		the output pin LED_OUT is pulsed HIGH when the sensor is undergoing exposure. 
		When this bit is enabled: If enabled (set to 1), and Invert LED_OUT is disabled, 
		the output pin LED_OUT is held in logicLOWstate. IfenabledandInvertLED_OUTis enabled, 
		output pin LED_OUT is held in a logic HIGH state.

		Invert LED_OUT
		Inverts polarity of LED_OUT output. 
		When this bit is set, the output pin LED_OUT is pulsed LOW 
		when the sensor is undergoing exposure.
		"""
		Invert = Invert <<1
		state = Invert | Disable 
		self.write16(id,0x1B ,state)

	def aptina_hblank(self,id,val=None):
		if val is None:
			return self.read16(id,0x05)
		else:
			if val not in range(61,1024): #exluding 1024 
				print 'hblank outside of allowable values 61-1023, using closest possible match'
				val = max(min(val,1023),61)
				print "hblank set to %i" %val
			self.write16(id,0x05,val)


	def aptina_vblank(self,id,val=None):
		if val is None:
			return self.read16(id,0x06)
		else:
			if val not in range(2,3228846): #exluding last val 
				print 'vblank outside of allowable values 2-3228845, using closest possible match'
				val = max(min(val,3228846),2)
				print "hblank set to %i" %val
			self.write16(id,0x06,val)

		
if __name__ == '__main__':
	import cv2
	cam = cam_interface()
	# buffer0 = np.ones((480,752), dtype=np.uint8) #this should always be a multiple of 4
	# buffer1 = np.zeros((480,640), dtype=np.uint8) #this should always be a multiple of 4
	# print "Initialized"
	# # buffer1.shape = (480,640)
	# for x in xrange(1000):
	# 	cam.get_frame(0,buffer0)
	# 	# cam.get_frame(1,buffer0)
	# 	cv2.imshow("cam interface testframe cam1", buffer0)
	cam.release()




