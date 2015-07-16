"""
	Andrew Xia playing around with porting c++ code to python
	I want to port singgleeyefitter.cpp and singgleeyefitter.h into this python script here
	June 25 2015

	EDIT June 30 2015: This file will contain only the class EyeModelFitter renamed spherefitter, that was
	originally in both singleeyefitter.h and singleeyefitter.cpp.

"""

#importing stuff
import numpy as np
import cv2
import scipy

import geometry
import intersect
import logging
logging.info('Starting logger for...')
logger = logging.getLogger(__name__)



def convert_fov(fov,width):
	fov = fov*scipy.pi/180
	focal_length = (width/2)/np.tan(fov/2)
	return focal_length

def sph2cart(r,theta,psi):
	toreturn = np.matrix([np.sin(theta)*np.cos(psi), np.cos(theta), np.sin(theta)*np.sin(psi)])
	#np is column major, so initializing this should give a 3x1 matrix (or vector)
	return r*toreturn

class Pupil(): #data structure for a pupil
	def __init__(self, ellipse = geometry.Ellipse(), intrinsics = None, radius = 1):
		self.ellipse = ellipse
		self.circle = geometry.Circle3D() #may delete later
		self.projected_circles = self.ellipse.unproject(radius = 1, intrinsics= intrinsics)
		self.params = geometry.PupilParams()
		self.init_valid = False

	def __str__(self):
		return "Pupil Class: " + str(self.ellipse) + str(self.circle) + " " + str(self.params) + " init_valid: " + str(self.init_valid)


#the class
class Sphere_Fitter():

	def __init__(self, intrinsics = None, focal_length = 554.256):
		if intrinsics is None:
			intrinsics = np.identity(3)
			if focal_length != None:
				intrinsics[0,0] = focal_length
				intrinsics[1,1] = focal_length
				logger.warning('no camera intrinsic input, set to focal length')
			else:
				logger.warning('no camera intrinsic input, set to default identity matrix')
		self.intrinsics = intrinsics #camera intrinsics of our webcam.
		self.camera_center = np.array([0,0,0])
		self.eye = geometry.Sphere((0,0,0),0) #model of our eye. geometry.sphere()
		self.projected_eye = geometry.Ellipse() #ellipse that is the projected eye sphere
		self.observations = [] #array containing elements in pupil class, originally "pupils"
		self.scale = 1
		self.model_version = 0

		# self.region_band_width = region_band_width
		# self.region_step_epsilon = region_step_epsilon
		# self.region_scale = 1
		# self.index = []
		self.count = 0 #used for limiting logging purposes

	def add_observation(self,ellipse):
		#ellipse is the ellipse of pupil in camera image
		self.observations.append(Pupil(ellipse = ellipse, intrinsics = self.intrinsics))

	def add_pupil_labs_observation(self,pupil_ellipse):
		converted_ellipse = geometry.Ellipse.from_ellipse_dict(pupil_ellipse)
		self.observations.append(Pupil(ellipse = converted_ellipse, intrinsics = self.intrinsics))


	def reset(self):
		self.observations = []
		eye = geometry.Sphere()
		self.model_version += 1

	def circleFromParams(self, params):
		# params = angles + diameter (theta, psi, radius)
		radial = sph2cart(1, params.theta, params.psi)
		cent = self.eye.center + self.eye.radius*radial
		cent = np.asarray(cent).reshape(3)
		return geometry.Circle3D(cent,radial, params.radius)

	def initialize_model(self):
		if self.eye.center[0] == 0 and self.eye.center[1] == 0 and self.eye.center[2] == 0 and self.eye.radius == 0:
			logger.warning("sphere has not been initialized")
			return

		eye_radius_acc = 0
		eye_radius_count = 0
		for pupil in self.observations:
			if not pupil.circle:
				continue
			if not pupil.init_valid:
				continue

			line1 = geometry.Line3D(origin = self.eye.center, direction =  pupil.circle.normal)
			line2 = geometry.Line3D(self.camera_center, pupil.circle.center/np.linalg.norm(pupil.circle.center))
			pupil_center = intersect.nearest_intersect_3D([line1, line2])
			distance = np.linalg.norm(np.asarray(pupil_center) - np.asarray(self.eye.center)) #normalize this
			eye_radius_acc += distance
			eye_radius_count += 1

		#set eye radius as mean distance from pupil centers to eye center
		self.eye.radius = eye_radius_acc / eye_radius_count
		#second estimate of pupil radius, used to get position of pupil on eye
		for pupil in self.observations:
			self.initialize_single_observation(pupil)

		#scale eye to anthropomorphic average radius of 12mm
		self.scale = 12.0 / self.eye.radius
		self.eye.radius = 12.0
		self.eye.center = np.array(self.eye.center)*self.scale

		for pupil in self.observations:
			pupil.params.radius = pupil.params.radius*self.scale
			pupil.circle = self.circleFromParams(pupil.params)

		#print every 30
		# logger.warning(self.eye)
		# logger.warning(self.eye)

		self.model_version += 1

	def initialize_single_observation(self,pupil):
		# Ignore pupil circle normal, intersect pupil circle
		# center projection line with eyeball sphere
		# try:
		line1 = geometry.Line3D(self.camera_center, pupil.circle.center)
		pupil_centre_sphere_intersect = intersect.sphere_intersect(line1,self.eye)
		if pupil_centre_sphere_intersect == None:
			# logger.warning('no intersection') # the warning is already called in intersect.py
			return
		new_pupil_center = pupil_centre_sphere_intersect[0]
		#given 3D position for pupil (rather than just projection line), recalculate pupil radius at position
		pupil_radius_at_1 = pupil.circle.radius/pupil.circle.center[2] #radius at z=1
		new_pupil_radius = pupil_radius_at_1 * new_pupil_center[2]
		#parametrize new pupil position using spherical coordinates
		center_to_pupil = np.asarray(new_pupil_center) - np.asarray(self.eye.center)
		r = np.linalg.norm(center_to_pupil)

		pupil.params.theta = np.arccos(center_to_pupil[1]/r)
		pupil.params.psi = np.arctan2(center_to_pupil[2],center_to_pupil[0])
		pupil.params.radius = new_pupil_radius

		#update pupil circle to match new parameter
		# pupil.circle = self.circleFromParams(pupil.params)

	def update_model(self):
		self.count += 1
		if self.count == 30 or 1:
			self.unproject_observations()
			self.initialize_model()
			self.count = 0

	def unproject_observations(self, eye_z = 20):
		# ransac default to false so I skip for loop (haven't implemented it yet)
		# this function for every ellipse from the image creates corresponding circles
		# unprojected to the pupil sphere model
		if (len(self.observations) < 2):
			logger.warning("Need at least two observations")
			return
		self.pupil_gazelines_proj = [] #it is a vector<line> !!

		for pupil in self.observations:
			""" get pupil circles
				Do a per-image unprojection of the pupil ellipse into the two fixed
				size circles that would project onto it. The size of the circles
				doesn't matter here, only their center and normal does.
			"""
			unprojection_pair = pupil.projected_circles

			""" get projected circles and gaze vectors
				Project the circle centers and gaze vectors down back onto the image plane.
				We're only using them as line parameterizations, so it doesn't matter which of the two centers/gaze
				vectors we use, as the two gazes are parallel and the centers are co-linear
			"""

			#why do I default use the 0th one, not the 1st one???

			#here maybe write some function that determines which line is better

			c = np.reshape(unprojection_pair[0].center, (3,1)) #it is a 3D circle
			v = np.reshape(unprojection_pair[0].normal, (3,1))
			c_proj = geometry.project_point(c,self.intrinsics)
			v_proj = geometry.project_point(v + c, self.intrinsics) - c_proj
			v_proj = v_proj/np.linalg.norm(v_proj) #normalizing
			line = geometry.Line2D(c_proj, v_proj)
			# print line
			self.pupil_gazelines_proj.append(line)

		""" Get eyeball center
			Find a least-squares 'intersection' (point nearest to all lines) of
			the projected 2D gaze vectors. Then, unproject that circle onto a
			point a fixed distance away.
			For robustness, use RANSAC to eliminate stray gaze lines
			(This has to be done here because it's used by the pupil circle disambiguation)
		"""
		eye_center_proj = []

		# if (use_ransac):
		# 	""" TO BE IMPLEMENTED (or maybe I won't bother since ransac isn't most important part"""
		# 	pass
		# else:
		for pupil in self.observations:
			pupil.init_valid = True
		eye_center_proj = intersect.nearest_intersect_2D(self.pupil_gazelines_proj)
		eye_center_proj = np.reshape(eye_center_proj,(2,))
		# print eye_center_proj
		valid_eye = True

		if (valid_eye):
			self.eye.center = geometry.unproject_point(eye_center_proj,eye_z,self.intrinsics)
			self.eye.radius = 1
			self.projected_eye = self.eye.project(self.intrinsics)

			for i in xrange(len(self.observations)):
				#disambiguate pupil circles using projected eyeball center
				line = self.pupil_gazelines_proj[i]
				c_proj = np.reshape(line.origin, (2,))
				v_proj = np.reshape(line.direction, (2,))

				if (np.dot(c_proj - eye_center_proj, v_proj) >= 0):
					#check if v_proj going away from estimated eye center, take the one going away.
					self.observations[i].circle = self.observations[i].projected_circles[0]
				else:
					self.observations[i].circle = self.observations[i].projected_circles[1]
		else:
			#no inliers, so no eye
			self.eye = Sphere.Sphere()
			# arbitrarily pick first circle
			for i in xrange(len(self.observations)):
				pupil_pair = pupil_unprojection_pairs[i]
				self.observations[i].circle = pupil_pair[0]

		#print every 30
		# self.count += 1
		# if self.count == 30:
		# 	logger.warning(self.eye)
		# 	self.count = 0
		# logger.warning(self.eye)

		self.model_version += 1

if __name__ == '__main__':

	#testing stuff
	huding = Sphere_Fitter(focal_length = 879.193)

	#testing unproject_observation, data from singleeyefitter/img_small
	ellipse1 = geometry.Ellipse((-147.579,100.093), 45.9498, 35.4209, 0.616285*scipy.pi)
	ellipse2 = geometry.Ellipse((-134.405,98.3423), 45.7818, 36.7225, 0.623024*scipy.pi)
	ellipse3 = geometry.Ellipse((-93.3441,9.03186), 47.185, 34.9949, 0.777285*scipy.pi)
	ellipse4 = geometry.Ellipse((-75.7188,67.1436), 69.3691, 60.3261, 0.726734*scipy.pi)
	ellipse5 = geometry.Ellipse((75.7414,69.3795), 63.0367, 58.3862, 0.155879*scipy.pi)
	ellipse6 = geometry.Ellipse((32.4573,44.9959), 56.4572, 51.3966, 0.0248723*scipy.pi)
	ellipse7 = geometry.Ellipse((-92.549,33.4655), 66.5554, 52.3161, 0.750369*scipy.pi)
	ellipse8 = geometry.Ellipse((-31.3134,81.571), 69.4012, 65.275, 0.7931*scipy.pi)
	ellipse9 = geometry.Ellipse((-134.405,98.3423), 45.7818, 36.7225, 0.623024*scipy.pi)
	ellipse10 = geometry.Ellipse((41.84,74.0696), 67.413, 63.518, 0.0462735*scipy.pi)

	huding.add_observation(ellipse1)
	huding.add_observation(ellipse2)
	huding.add_observation(ellipse3)
	huding.add_observation(ellipse4)
	# huding.add_observation(ellipse5)
	# huding.add_observation(ellipse6)
	# huding.add_observation(ellipse7)
	# huding.add_observation(ellipse8)
	# huding.add_observation(ellipse9)
	# huding.add_observation(ellipse10)

	huding.unproject_observations()
	huding.initialize_model()
