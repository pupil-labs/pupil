if __name__ == '__main__':
    import subprocess as sp
    sp.call("python setup.py build_ext --inplace",shell=True)
print "BUILD COMPLETE ______________________"

import eye_model_3d
import timeit

############# TESTING MODEL ##################
if __name__ == '__main__':
	def basic_test():
		model = eye_model_3d.PyEyeModelFitter(focal_length=879.193, x_disp = 320, y_disp = 240)
		# print model
		model.add_observation([422.255,255.123],40.428,30.663,1.116)
		model.add_observation([442.257,365.003],44.205,32.146,1.881)
		model.add_observation([307.473,178.163],41.29,22.765,0.2601)
		model.add_observation([411.339,290.978],51.663,41.082,1.377)
		model.add_observation([198.128,223.905],46.852,34.949,2.659)
		model.add_observation([299.641,177.639],40.133,24.089,0.171)
		model.add_observation([211.669,212.248],46.885,33.538,2.738)
		model.add_observation([196.43,236.69],47.094,38.258,2.632)
		model.add_observation([317.584,189.71],42.599,27.721,0.3)
		model.add_observation([482.762,315.186],38.397,23.238,1.519)
		model.update_model()
		print model.print_eye() #Sphere(center = [ -3.02103998  -4.64862274  49.54492648], radius = 12.0)
		print model.eye
		# print model.num_observations
		for pupil in model.get_all_pupil_observations():
			print pupil[0]
		print " "
		for pupil in model.get_last_pupil_observations(3):
			print pupil[0]

		print model.scale
		
	basic_test()

	# timer functions
	# start_time = timeit.default_timer()
	# timeit.Timer(basic_test).timeit(number=1000)
	# print(timeit.default_timer() - start_time)
