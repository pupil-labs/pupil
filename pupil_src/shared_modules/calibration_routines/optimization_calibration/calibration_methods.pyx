'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from libcpp.vector cimport vector

from calibration_methods cimport *
import numpy as np




def bundle_adjust_calibration( initial_observers, initial_points,fix_points = True):


    cdef vector[Observer] cpp_observers;
    cdef Observer cpp_observer
    cdef vector[double] cpp_pose
    cdef vector[Vector3] cpp_observations
    cdef vector[Vector3] cpp_points

    cdef Vector4 rotation_quaternion
    cdef Vector3 rotation_angle_axis
    cdef Vector3 cpp_translation

    for o in initial_observers:
        observations = o["observations"]
        translation = o["translation"]
        rotation = o["rotation"]
        cpp_pose.resize(6)


        rotation_quaternion = Vector4(rotation[0],rotation[1],rotation[2],rotation[3])
        #angle axis rotation: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
        QuaternionToAngleAxis(rotation_quaternion.data(),rotation_angle_axis.data())


        #we need to invert the pose of the observer
        #we will use this rotation translation to tranform the observed points in the cost fn
        cpp_translation = Vector3(-translation[0],-translation[1],-translation[2])

        #invert rotation
        rotation_angle_axis[0] *= -1
        rotation_angle_axis[1] *= -1
        rotation_angle_axis[2] *= -1

        #we have swapped to order rot/trans in the cost fn so we dont need to apply the line below
        #AngleAxisRotatePoint(rotation_angle_axis.data(),cpp_translation.data(),cpp_translation.data())




        #first three is rotation
        cpp_pose[0] = rotation_angle_axis[0]
        cpp_pose[1] = rotation_angle_axis[1]
        cpp_pose[2] = rotation_angle_axis[2]

        #last three is translation
        cpp_pose[3] = cpp_translation[0]
        cpp_pose[4] = cpp_translation[1]
        cpp_pose[5] = cpp_translation[2]

        cpp_observations.clear()
        for p in observations:
            cpp_observations.push_back(Vector3(p[0],p[1],p[2]))

        cpp_observer = Observer()
        cpp_observer.observations = cpp_observations
        cpp_observer.pose = cpp_pose
        cpp_observer.fix_rotation = 1*bool('rotation' in o['fix'])
        cpp_observer.fix_translation = 1*bool('translation' in o['fix'])
        cpp_observers.push_back( cpp_observer )

    for p in initial_points:
        cpp_points.push_back( Vector3(p[0],p[1],p[2]) )


    ## optimized values are written to cpp_orientation and cpp_translation
    cdef double final_cost  = bundleAdjustCalibration(cpp_observers, cpp_points,fix_points)


    observers = []
    for cpp_observer in cpp_observers:
        observer = {}

        #invert translation rotation back to get the pose
        rotation_angle_axis = Vector3(cpp_observer.pose[0],cpp_observer.pose[1],cpp_observer.pose[2])
        cpp_translation = Vector3(-cpp_observer.pose[3],-cpp_observer.pose[4],-cpp_observer.pose[5])

        rotation_angle_axis[0] *=-1
        rotation_angle_axis[1] *=-1
        rotation_angle_axis[2] *=-1

        #we have swapped to order rot/trans in the cost fn so we dont need to apply the line below
        #AngleAxisRotatePoint(rotation_angle_axis.data(),cpp_translation.data(),cpp_translation.data())

        AngleAxisToQuaternion(rotation_angle_axis.data(),rotation_quaternion.data())

        observer['rotation'] = rotation_quaternion[0],rotation_quaternion[1],rotation_quaternion[2],rotation_quaternion[3]
        observer['translation'] = cpp_translation[0],cpp_translation[1],cpp_translation[2]
        observers.append(observer)

    for final,inital in zip(observers,initial_observers):
        final['observations'] = inital['observations']


    points = []
    cdef Vector3 cpp_p
    for cpp_p in cpp_points:
        points.append( (cpp_p[0],cpp_p[1],cpp_p[2]) )

    return final_cost != -1,final_cost, observers, points
