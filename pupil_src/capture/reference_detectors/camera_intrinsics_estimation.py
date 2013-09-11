import cv2
import numpy as np
from gl_utils import draw_gl_polyline

import atb
import audio

from plugin import Plugin

class Camera_Intrinsics_Estimation(Plugin):
	"""Camera_Intrinsics_Calibration
		not being an actual calibration,
		this method is used to calculate camera intrinsics.

	"""
	def __init__(self, screen_marker_pos,screen_marker_state,atb_pos=(0,0)):
		Plugin.__init__(self)
		self.collect_new = False
		self.calculated = False
		self.obj_grid = _gen_pattern_grid((4, 11))
		self.img_points = []
		self.obj_points = []
		self.count = 10
		self.img_shape = None


		atb_label = "estimate camera instrinsics"
		# Creating an ATB Bar is required. Show at least some info about the Ref_Detector
		self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
			help="ref detection parameters", color=(50, 50, 50), alpha=100,
			text='light', position=atb_pos,refresh=.3, size=(300, 100))
		self._bar.add_button("  Capture Pattern", self.advance, key="SPACE")
		self._bar.add_var("patterns to capture", getter=self.get_count)

	def get_count(self):
		return self.count

	def advance(self):
		if self.count ==10:
			audio.say("Capture 10 calibration patterns.")
		self.collect_new = True

	def new_ref(self,pos):
		pass

	def calculate(self):
		self.calculated = True
		camera_matrix, dist_coefs = _calibrate_camera(np.asarray(self.img_points),
													np.asarray(self.obj_points),
													(self.img_shape[1], self.img_shape[0]))
		np.save("camera_matrix.npy", camera_matrix)
		np.save("dist_coefs.npy", dist_coefs)
		audio.say("Camera calibrated and saved to file")

	def update(self,frame):
		if self.collect_new:
			img = frame.img
			status, grid_points = cv2.findCirclesGridDefault(img, (4,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
			if status:
				self.img_points.append(grid_points)
				self.obj_points.append(self.obj_grid)
				self.collect_new = False
				self.count -=1
				if self.count in range(1,10):
					audio.say("%i" %(self.count))
				self.img_shape = img.shape

		if not self.count and not self.calculated:
			self.calculate()

	def gl_display(self):
		"""
		use gl calls to render
		at least:
			the published position of the reference
		better:
			show the detected postion even if not published
		"""
		for grid_points in self.img_points:
			calib_bounds =  cv2.convexHull(grid_points)[:,0] #we dont need that extra encapsulation that opencv likes so much
			draw_gl_polyline(calib_bounds,(0.,0.,1.,.5), type="Loop")

	def __del__(self):
		pass


# shared helper functions for detectors private to the module
def _calibrate_camera(img_pts, obj_pts, img_size):
	# generate pattern size
	camera_matrix = np.zeros((3,3))
	dist_coef = np.zeros(4)
	rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts,
													img_size, camera_matrix, dist_coef)
	return camera_matrix, dist_coefs

def _gen_pattern_grid(size=(4,11)):
	pattern_grid = []
	for i in xrange(size[1]):
		for j in xrange(size[0]):
			pattern_grid.append([(2*j)+i%2,i,0])
	return np.asarray(pattern_grid, dtype='f4')