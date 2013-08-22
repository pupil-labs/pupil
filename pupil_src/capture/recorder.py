import os, sys
import cv2
import atb
import numpy as np
from plugin import Plugin
from time import strftime,localtime,time,gmtime
from ctypes import create_string_buffer
from git_version import get_tag_commit

class Recorder(Plugin):
	"""Capture Recorder"""
	def __init__(self, session_str, fps, img_shape, shared_record, eye_tx):
		Plugin.__init__(self)
		self.session_str = session_str
		self.base_path = os.path.join(os.path.abspath(__file__).rsplit('pupil_src', 1)[0], "recordings")
		self.shared_record = shared_record
		self.frame_count = 0
		self.timestamps = []
		self.eye_tx = eye_tx

		self.start_time = time()
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

		self.meta_info_path = os.path.join(self.path, "info.csv")

		with open(self.meta_info_path, 'w') as f:
			f.write("Pupil Recording Name:\t"+self.session_str+ "\n")
			f.write("Start Date: \t"+ strftime("%d.%m.%Y", localtime(self.start_time))+ "\n")
			f.write("Start Time: \t"+ strftime("%H:%M:%S", localtime(self.start_time))+ "\n")



		video_path = os.path.join(self.path, "world.avi")
		self.writer = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), fps, (img_shape[1], img_shape[0]))
		self.height = img_shape[0]
		self.width = img_shape[1]
		# positions path to eye process
		self.shared_record.value = True
		self.eye_tx.send(self.path)

		atb_pos = (10, 540)
		self._bar = atb.Bar(name = self.__class__.__name__, label='rec: '+session_str,
			help="capture recording control", color=(200, 0, 0), alpha=100,
			text='light', position=atb_pos,refresh=.3, size=(300, 80))
		self._bar.rec_name = create_string_buffer(512)
		self._bar.add_var("rec time",self._bar.rec_name, getter=lambda: create_string_buffer(self.get_rec_time_str(),512), readonly=True)
		self._bar.add_button("stop", self.stop_and_destruct, key="s", help="stop recording")

	def get_rec_time_str(self):
		rec_time = gmtime(time()-self.start_time)
		return strftime("%H:%M:%S", rec_time)

	def update(self, frame):
		self.frame_count += 1
		self.timestamps.append(frame.timestamp)
		self.writer.write(frame.img)

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

		timestamps_path = os.path.join(self.path, "timestamps.npy")
		np.save(timestamps_path,np.array(self.timestamps))


		with open(self.meta_info_path, 'a') as f:
			f.write("Duration Time: \t"+ self.get_rec_time_str()+ "\n")
			f.write("World Camera Frames: \t"+ str(self.frame_count)+ "\n")
			f.write("World Camera Resolution: \t"+ str(self.width)+"x"+str(self.height)+"\n")
			f.write("Capture Software Version: \t"+ get_tag_commit()+ "\n")
			f.write("user:\t"+os.getlogin()+"\n")
			try:
				sysname, nodename, release, version, machine = os.uname()
			except:
				sysname, nodename, release, version, machine = sys.platform,None,None,None,None
			f.write("Platform:\t"+sysname+"\n")
			f.write("Machine:\t"+nodename+"\n")
			f.write("Release:\t"+release+"\n")
			f.write("Version:\t"+version+"\n")



		print "Stopping recording"
		self.shared_record.value = False
		self.alive = False

	def __del__(self):
		"""incase the plugin get deleted while recording
		"""
		self.stop_and_destruct()

def get_auto_name():
	return strftime("%Y_%m_%d", localtime())
