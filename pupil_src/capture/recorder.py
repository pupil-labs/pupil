import os, sys
import cv2
import atb

from plugin import Plugin
from time import strftime,localtime

class Recorder(Plugin):
	"""Capture Recorder"""
	def __init__(self, session_str, fps, img_shape, shared_record, shared_frame_count, eye_tx):
		Plugin.__init__(self)
		self.session_str = session_str
		self.base_path = os.path.join(os.path.abspath(__file__).rsplit('pupil_src', 1)[0], "recordings")
		self.shared_record = shared_record
		self.shared_frame_count = shared_frame_count
		self.eye_tx = eye_tx

		# set up base folder called "recordings"
		try:
			os.mkdir(self.base_path)
		except:
			print "recordings folder already exists, using existing."

		session = os.path.join(self.base_path, self.session_str)
		try:
			os.mkdir(session)
		except:
			print "recordings session folder already exists, using existing."

		# set up self incrementing folder within session folder
		counter = 0
		while True:
			self.path = os.path.join(self.base_path, session, "%03d/" % counter)
			try:
				os.mkdir(self.path)
				break
			except:
				print "We dont want to overwrite data, incrementing counter & trying to make new data folder"
				counter += 1

		video_path = os.path.join(self.path, "world.avi")
		self.writer = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), fps, (img_shape[1], img_shape[0]))

		# positions path to eye process
		self.shared_record.value = True
		self.eye_tx.send(self.path)

		atb_pos = (10, 540)
		self._bar = atb.Bar(name = self.__class__.__name__, label='rec: '+session_str,
			help="capture recording control", color=(200, 0, 0), alpha=100,
			text='light', position=atb_pos,refresh=.3, size=(300, 80))
		self._bar.add_button("stop", self.stop_and_destruct, key="s", help="stop recording")


	def update(self, img):
		self.shared_frame_count.value += 1
		self.writer.write(img)

	def stop_and_destruct(self):
		try:
			camera_matrix = np.load("camera_matrix.npy")
			dist_coefs = np.load("dist_coefs.npy")
			cam_path = os.path.join(self.path, "camera_matrix.npy")
			dist_path = os.path.join(self.path, "dist_coefs.npy")
			np.save(cam_path, camera_matrix)
			np.save(dist_path, dist_coefs)
		except:
			print "no camera intrinsics found, will not copy them into recordings folder"

		print "Stopping recording"
		self.shared_record.value = False
		self.alive = False


def get_auto_name():
	return strftime("%Y_%m_%d", localtime())


