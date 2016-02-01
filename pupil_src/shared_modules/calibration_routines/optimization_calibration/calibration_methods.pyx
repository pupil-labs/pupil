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

def point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D ):



    cdef vector[Vector3] cpp_ref_points
    cdef vector[Vector3] cpp_gaze_directions
    for p in ref_points_3D:
        cpp_ref_points.push_back(Vector3(p[0],p[1],p[2]))

    for p in gaze_directions_3D:
        cpp_gaze_directions.push_back(Vector3(p[0],p[1],p[2]))


    cdef vector[Vector3] cpp_gaze_points
    cdef Vector3 cpp_sphere_position
    cpp_sphere_position = Vector3(sphere_position[0],sphere_position[1],sphere_position[2])
    cdef Matrix4d transformation = pointLineCalibration(cpp_sphere_position, cpp_ref_points , cpp_gaze_directions ,cpp_gaze_points)

    gaze_points = []
    for point in cpp_gaze_points:
        gaze_points.append([point[0],point[1],point[2]])

    py_transformation = np.matrix( np.identity(4 , dtype = np.dtype('d')) )

    for i in range(0,4):
        for j in range(0,4):
            py_transformation[i,j] = transformation(i,j)

    return py_transformation , gaze_points
