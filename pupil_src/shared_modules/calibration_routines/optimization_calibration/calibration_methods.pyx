'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from libcpp.vector cimport vector

cimport calibration_methods
from calibration_methods cimport *
import numpy as np

def point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation ,
    fix_translation = False, translation_lower_bound = (15,5,5) ,  translation_upper_bound = (15,5,5) ):


    cdef vector[Vector3] cpp_ref_points
    cdef vector[Vector3] cpp_gaze_directions
    for p in ref_points_3D:
        cpp_ref_points.push_back(Vector3(p[0],p[1],p[2]))

    for p in gaze_directions_3D:
        cpp_gaze_directions.push_back(Vector3(p[0],p[1],p[2]))

    cdef Vector3 cpp_sphere_position
    cpp_sphere_position = Vector3(sphere_position[0],sphere_position[1],sphere_position[2])

    cdef double cpp_orientation[4] #quaternion
    cdef double cpp_translation[3]
    cpp_orientation[:] = initial_orientation
    cpp_translation[:] = initial_translation

    cdef Vector3 cpp_translation_upper_bound = Vector3(translation_upper_bound[0],translation_upper_bound[1],translation_upper_bound[2])
    cdef Vector3 cpp_translation_lower_bound = Vector3(translation_lower_bound[0],translation_lower_bound[1],translation_lower_bound[2])

    ## optimized values are written to cpp_orientation and cpp_translation
    cdef bint success  = pointLineCalibration(cpp_sphere_position, cpp_ref_points, cpp_gaze_directions,
                                             &cpp_orientation[0], &cpp_translation[0], fix_translation,
                                             cpp_translation_lower_bound, cpp_translation_upper_bound )


    return success, cpp_orientation, cpp_translation


# def line_line_calibration( ref_directions_3D, gaze_directions_3D , initial_orientation , initial_translation , fix_translation = False , use_weight = True):


#     cdef vector[Vector3] cpp_ref_directions
#     cdef vector[Vector3] cpp_gaze_directions
#     for p in ref_directions_3D:
#         cpp_ref_directions.push_back(Vector3(p[0],p[1],p[2]))

#     for p in gaze_directions_3D:
#         cpp_gaze_directions.push_back(Vector3(p[0],p[1],p[2]))


#     cdef Quaterniond cpp_orientation
#     cdef Vector3 cpp_translation
#     cpp_orientation = Quaterniond(initial_orientation[0],initial_orientation[1],initial_orientation[2],initial_orientation[3] )
#     cpp_translation = Vector3(initial_translation[0],initial_translation[1],initial_translation[2] )

#     cdef double avgDistance = 0.0

#     ## optimized values are written to cpp_orientation and cpp_translation
#     cdef bint success  = lineLineCalibration(cpp_ref_directions, cpp_gaze_directions,
#                                              cpp_orientation, cpp_translation, avgDistance, fix_translation, use_weight )


#     orientation = ( cpp_orientation.w(),cpp_orientation.x(),cpp_orientation.y(),cpp_orientation.z() )
#     translation = ( cpp_translation[0],cpp_translation[1],cpp_translation[2] )

#     return success, orientation, translation , avgDistance


def bundle_adjust_calibration( observations, fix_translation = False , use_weight = True):

    cdef double avgDistance = 0.0


    cdef vector[Observation] cpp_observations;

    cdef Observation cpp_observation
    cdef vector[double] cpp_camera
    cdef vector[Vector3] cpp_dir
    cdef vector[Vector3] cpp_points

    for o in observations:
        directions = o["directions"]
        translation = o["translation"]
        orientation = o["orientation"]

        cpp_camera.resize(7)
        cpp_camera[0] = orientation[0]
        cpp_camera[1] = orientation[1]
        cpp_camera[2] = orientation[2]
        cpp_camera[3] = orientation[3]
        cpp_camera[4] = translation[0]
        cpp_camera[5] = translation[1]
        cpp_camera[6] = translation[2]

        cpp_dir.clear()
        for p in directions:
            cpp_dir.push_back(Vector3(p[0],p[1],p[2]))

        cpp_observation = Observation()
        cpp_observation.dirs = cpp_dir
        cpp_observation.camera = cpp_camera
        cpp_observations.push_back( cpp_observation )

    cdef vector[vector[double]] camera_results
    cdef vector[double] camera
    ## optimized values are written to cpp_orientation and cpp_translation
    cdef bint success  = bundleAdjustCalibration(cpp_observations, cpp_points, camera_results,  fix_translation, use_weight,  )

    orientations = []
    translations = []

    for i in range(0,camera_results.size() ):
        camera = camera_results.at(i)
        orientations.append( (camera[0],camera[1],camera[2],camera[3] ) )
        translations.append( (camera[4],camera[5],camera[6] ) )


    return success, orientations, translations , avgDistance
