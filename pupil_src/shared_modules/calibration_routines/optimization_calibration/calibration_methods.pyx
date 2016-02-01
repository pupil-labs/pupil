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


def point_line_calibration( ref_points_3D, gaze_directions_3D ):



    cdef vector[Vector3] cpp_ref_points
    cdef vector[Vector3] cpp_gaze_directions
    for p in ref_points_3D:
        cpp_ref_points.push_back(Vector3(p[0],p[1],p[2]))

    for p in gaze_directions_3D:
        cpp_gaze_directions.push_back(Vector3(p[0],p[1],p[2]))


    cdef Matrix4d transformation = pointLineCalibration( cpp_ref_points , cpp_gaze_directions)




