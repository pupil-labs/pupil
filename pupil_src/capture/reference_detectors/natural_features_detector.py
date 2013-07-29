import cv2
import numpy as np
from methods import normalize
from gl_utils import draw_gl_point_norm

import atb
import audio

class Natural_Features_Detector(object):
	"""Calibrate using natural features in a scene.
		Features are selected by a user by clicking on
	"""
	def __init__(self, global_calibrate,shared_pos,screen_marker_pos,screen_marker_state,atb_pos=(0,0)):
		self.first_img = None
		self.point = None
		self.count = 0
		self.detected = False
		self.active = False
		self.global_calibrate = global_calibrate
		self.global_calibrate.value = False
		self.shared_pos = shared_pos
		self.pos = 0,0 # 0,0 is used to indicate no point detected
		self.var1 = c_int(0)
		self.r = 40.0 # radius of circle displayed

		# Creating an ATB Bar is required. Show at least some info about the Ref_Detector
		self._bar = atb.Bar(name = "Reference_Detector", label="Natural Features Detector",
			help="ref detection parameters", color=(50, 50, 50), alpha=100,
			text='light', position=atb_pos,refresh=.3, size=(300, 100))
		self._bar.add_button("Start", self.start)
		self._bar.add_button("Stop", self.stop)

	def start(self):
		audio.say("Starting Calibration")
		self.global_calibrate.value = True
		self.shared_pos[:] = 0,0
		self.active = True

	def stop(self):
		audio.say("Stopping Calibration")
		self.global_calibrate.value = False
		self.shared_pos[:] = 0,0
		self.active = False

	def detect(self,img):
		if self.active:
			if self.first_img is None:
				self.first_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

			if self.count:
				gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
				nextPts, status, err = cv2.calcOpticalFlowPyrLK(self.first_img,gray,self.point,winSize=(100,100))
				if status[0]:
					self.detected = True
					self.point = nextPts
					self.first_img = gray
					nextPts = nextPts[0]
					self.pos = normalize(nextPts,(img.shape[1],img.shape[0]),flip_y=True)
					self.count -=1
				else:
					self.detected = False
					self.pos = 0,0
			else:
				self.detected = False
				self.pos = 0,0

			self.publish()

	def gl_display(self):
		if self.detected:
			draw_gl_point_norm(self.pos,size=self.r,color=(0.,1.,0.,.5))

	def publish(self):
		self.shared_pos[:] = self.pos

	def new_ref(self,pos):
		self.first_img = None
		self.point = np.array([pos,],dtype=np.float32)
		self.count = 30


	def del_bar(self):
		"""Delete the ATB bar manually.
			Python's garbage collector doesn't work on the object otherwise
			Due to the fact that ATB is a c library wrapped in ctypes

		"""
		self._bar.destroy()
		del self._bar

	def __del__(self):
		self.global_calibrate.value = False
		self.shared_pos[:] = 0,0