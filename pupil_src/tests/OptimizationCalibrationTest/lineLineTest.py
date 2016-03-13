import os, sys, platform
loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
sys.path.append(os.path.join(loc[0], 'pupil_src', 'shared_modules'))

from calibration_routines.optimization_calibration import  bundle_adjust_calibration

import numpy as np
import cv2
import math
import math_helper
from numpy import array, cross, dot, double, hypot, zeros
from math import acos, atan2, cos, pi, sin, radians
from calibration_routines.visualizer_calibration import *
from calibration_routines.calibrate import find_rigid_transform

def show_result(cam1_center, cam1_points, cam2_points, rotation, translation,avg_distance, name = "Calibration"):

    rotation = np.array(rotation)
    translation = np.array(translation)

    print 'initial rotation: ' , math_helper.about_axis_from_quaternion(initial_rotation)
    print 'initial translation: ' , initial_translation

    print 'found rotation: ' , math_helper.about_axis_from_quaternion(rotation)
    print 'found translation: ' , translation

    print 'true rotation: ' , math_helper.about_axis_from_quaternion(cam2_rotation_quaternion)
    print 'true translation: ' , cam2_center

    print 'avgerage distance of intersections from solver: ' , avg_distance

    #replace with the optimized rotation and translation
    cam2_rotation_matrix = math_helper.quaternion_rotation_matrix(rotation)
    cam2_transformation_matrix  = np.matrix(np.eye(4))
    cam2_transformation_matrix[:3,:3] = cam2_rotation_matrix
    cam2_translation = np.matrix(translation)
    cam2_translation.shape = (3,1)
    cam2_transformation_matrix[:3,3:4] = cam2_translation

    def toWorld(p):
        return np.dot(cam2_rotation_matrix, p)+translation

    eye = { 'center': translation, 'radius': 1.0}

    intersection_points_a = [] #world coords
    intersection_points_b = [] #cam2 coords
    avg_error = 0.0
    for a,b in zip(cam1_points , cam2_points): #world coords , cam2 coords

        line_a = (np.array(cam1_center) , np.array(a))
        line_b = (toWorld(cam1_center) , toWorld(b) ) #convert to world for intersection

        ai, bi, distance =  math_helper.nearest_intersection_points( line_a , line_b ) #world coords
        avg_error +=distance
        intersection_points_a.append(ai)
        intersection_points_b.append(bi)  #cam2 coords , since visualizer expects local coordinates

    avg_error /= len(cam2_points)
    print 'avg distace of intersections re calulated on scaled data: ', avg_error

    #cam2_points_world = [toWorld(p) for p in cam2_points]

    visualizer = Calibration_Visualizer(None,None, intersection_points_a ,cam2_transformation_matrix , intersection_points_b, run_independently = True , name = name)
    visualizer.open_window()
    while visualizer.window:
        visualizer.update_window( None, [] , eye)

    return

if __name__ == '__main__':


    from random import uniform

    cam1_center  = (0,0,0)
    cam1_rotation_quaternion = math_helper.quaternion_about_axis( 0 , (0.0,1.0,0.0) )
    cam1_rotation_matrix = math_helper.quaternion_rotation_matrix(cam1_rotation_quaternion)
    cam1_rotation_angle_axis , _ = cv2.Rodrigues(cam1_rotation_matrix)

    cam2_center  = np.array((5,0,0))
    cam2_rotation_quaternion = math_helper.quaternion_about_axis( -np.pi* 0.1, (0.0,1.0,0.0) )
    cam2_rotation_matrix = math_helper.quaternion_rotation_matrix(cam2_rotation_quaternion)
    cam2_rotation_angle_axis, _  = cv2.Rodrigues(cam2_rotation_matrix)
    random_points = []
    random_points_amount = 30

    x_var = 200
    y_var = 200
    z_var = 20
    z_min = 300
    for i in range(0,random_points_amount):
        random_point = ( uniform(-x_var,x_var) ,  uniform(-y_var,y_var) ,  uniform(z_min,z_min+z_var)  )
        random_points.append(random_point)


    def toEye(p):
        return np.dot(cam2_rotation_matrix.T, p-cam2_center )

    cam1_points = [] #cam1 coords
    cam2_points = [] #cam2 coords
    for p in random_points:
        cam1_points.append(p)
        factor = 0 #randomize point in eye space
        pr = p + np.array( (uniform(-factor,+factor),uniform(-factor,+factor),uniform(-factor,+factor))  )
        p2 = toEye(pr) # to cam2 coordinate system
       # p2 *= 1.2,1.3,1.0
        cam2_points.append(p2)

    sphere_position = (0,0,0)

    cam1_dir = [ p/np.linalg.norm(p) for p in cam1_points]
    cam2_dir = [ p/np.linalg.norm(p) for p in cam2_points]

    initial_R,initial_t = find_rigid_transform(np.array(cam2_points),np.array(cam1_points))
    initial_rotation = math_helper.quaternion_from_rotation_matrix(initial_R)
    initial_translation = np.array(initial_t).reshape(3)
    initial_translation *= np.linalg.norm(cam2_center)/np.linalg.norm(initial_translation)

    o1 = { "directions" : cam1_dir , "translation" : [0,0,0] , "orientation" : cam1_rotation_angle_axis  }
    o2 = { "directions" : cam2_dir , "translation" : cam2_center , "orientation" : cam2_rotation_angle_axis  }
    observations = [o1, o2]

    success, rotations, translations, points = bundle_adjust_calibration( observations , cam1_points, fix_translation = False, use_weight = True  )

    avg_distance = 0.0
    rotation = math_helper.quaternion_from_matrix( cv2.Rodrigues(rotations[1])[0] )
    translation = translations[1]
    #success, rotation, translation, avg_distance = line_line_calibration( cam1_dir, cam2_dir , initial_rotation , initial_translation , fix_translation = False, use_weight = True  )
    # success2, rotation2, translation2, avg_distance2 = line_line_calibration( cam1_dir, cam2_dir , initial_rotation , initial_translation , fix_translation = False , use_weight = True  )
    success2, rotation2, translation2, avg_distance2 = True,initial_rotation,initial_translation,-1
    from multiprocessing import Process
    print "final result -------------------"
    p = Process(target=show_result, args=(cam1_center, points, cam2_points, rotation, translation,avg_distance))
    p.start();

    import time
    time.sleep(1)
    print "inital guess -------------------"
    show_result(cam1_center, cam1_points, cam2_points, rotation2, translation2 ,avg_distance2, "inital guess")

    p.join()

