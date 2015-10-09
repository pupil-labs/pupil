from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libc.stdint cimport int32_t
cimport numpy as np

import numpy as np
import math


# cdef extern from 'singleeyefitter/intersect.h' namespace 'singleeyefitter':
#     cdef pair[Matrix31d,Matrix31d] intersect(const ParametrizedLine3d line, const Sphere[double] sphere) except +

cdef extern from '<Eigen/Eigen>' namespace 'Eigen':
    cdef cppclass Matrix21d "Eigen::Matrix<double,2,1>": # eigen defaults to column major layout
        Matrix21d() except +
        double * data()
        double& operator[](size_t)


    cdef cppclass Matrix31d "Eigen::Matrix<double,3,1>": # eigen defaults to column major layout
        Matrix31d() except +
        Matrix31d(double x, double y, double z)
        double * data()
        double& operator[](size_t)

    # cdef cppclass ParametrizedLine3d "Eigen::ParametrizedLine<double, 3>":
    #     ParametrizedLine3d() except +
    #     ParametrizedLine3d(Matrix31d origin, Matrix31d direction)


#typdefs
ctypedef Matrix31d Vector3
ctypedef Matrix21d Vector2

cdef extern from "singleeyefitter/singleeyefitter.h" namespace "singleeyefitter":

    cdef cppclass Ellipse2D[T]:
        Ellipse2D()
        Ellipse2D(T x, T y, T major_radius, T minor_radius, T angle) except +
        Matrix21d center
        T major_radius
        T minor_radius
        T angle

    cdef cppclass Sphere[T]:
        Matrix31d center
        T radius

    cdef cppclass Circle3D[T]:
        Matrix31d center
        Matrix31d normal
        float radius

    cdef cppclass EyeModelFitter:
        EyeModelFitter(double focal_length, double x_disp, double y_disp)
        # EyeModelFitter(double focal_length)
        void reset()
        void initialise_model()
        void unproject_observations(double pupil_radius, double eye_z )
        void add_observation( Ellipse2D[double] ellipse)
        void add_observation( Ellipse2D[double] ellipse, vector[int32_t*] contours , vector[size_t] sizes )

        #######################
        ## Pubil-Laps addons ##
        #######################

        void unproject_contours();
        void unwrap_contours();

        #######################

        cppclass PupilParams:
            float theta
            float psi
            float radius

        cppclass Observation:
            Ellipse2D[double] ellipse
            vector[vector[int32_t]] contours

        cppclass Pupil:
            Pupil() except +
            Observation observation
            vector[vector[Vector3]] unprojected_contours
            vector[vector[Vector2]] unwrapped_contours
            PupilParams params
            Circle3D[double] circle

        #variables
        float model_version
        float focal_length
        vector[Pupil] pupils
        Sphere[double] eye
        #Ellipse2D[double] projected_eye #technically only need center, not whole ellipse. can optimize here
        #float scale




from collections import namedtuple
PyObservation = namedtuple('Observation' , 'ellipse_center, ellipse_major_radius, ellipse_minor_radius, ellipse_angle,params_theta, params_psi, params_radius, circle_center, circle_normal, circle_radius')

cdef class PyEyeModelFitter:
    cdef EyeModelFitter *thisptr
    cdef public int counter
    cdef public int num_observations
    # def __cinit__(self, focal_length):
    #     self.thisptr = new EyeModelFitter(focal_length)
    def __cinit__(self, focal_length, region_band_width = 5 , region_step_epsilon = 0.5):
        self.thisptr = new EyeModelFitter(focal_length, region_band_width, region_step_epsilon)

    def __init__(self,focal_length, region_band_width = 5 , region_step_epsilon = 0.5):
        self.counter = 0
        self.num_observations = 0

    def __dealloc__(self):
        del self.thisptr

    def reset(self):
        self.thisptr.reset()

    def initialise_model(self):
        self.thisptr.initialise_model()

    def unproject_observations(self, pupil_radius = 1, eye_z = 20):
        self.thisptr.unproject_observations(pupil_radius,eye_z)

    def update_model(self, pupil_radius = 1, eye_z = 20):
        # this function runs unproject_observations and initialise_model, and prints
        # the eye model once every 30 iterations.
        if self.counter >= 30:
            self.counter = 0
            print self.print_eye()
        self.counter += 1
        self.thisptr.unproject_observations(pupil_radius,eye_z)
        self.thisptr.initialise_model()

        #now we have an updated eye model
        #use it to unproject contours
        self.thisptr.unproject_contours()
        #calculate uv coords of unprojected contours
        self.thisptr.unwrap_contours()

    # def add_observation(self,center,major_radius,minor_radius,angle):
    #     #standard way of adding an observation
    #     self.thisptr.add_observation(center[0], center[1],major_radius,minor_radius,angle)
    #     self.num_observations += 1

    def add_pupil_labs_observation(self,ellipse_dict, contours, image_size):
        # a special method for taking in arguments from eye.py

        a,b = ellipse_dict['axes']
        if a > b:
            major_radius = a/2.0
            minor_radius = b/2.0
            angle = -ellipse_dict['angle']* math.pi/180
        else:
            major_radius = b/2.0
            minor_radius = a/2.0
            angle = (ellipse_dict['angle']+90)*math.pi/180 # not importing np just for pi constant

        # change coord system to centered origin
        x,y = ellipse_dict['center']
        x -= image_size[0]/2.0
        y = image_size[1]/2.0 - y
        angle = -angle #take y axis flip into account

        # add cpp ellipse
        cdef Ellipse2D[double] ellipse  =  Ellipse2D[double](x,y, major_radius, minor_radius, angle)


        #point coords are in pixels, with origin top left
        # map them so coord origin is centered with y up
        for contour in contours:
            for point in contour:
                point[0] = point[0] - image_size[0]/2.0
                point[1] = image_size[1]/2.0 - point[1]

        cdef vector[int32_t*] contour_ptrs #vector holding pointers to each contour memory
        cdef vector[size_t] contour_sizes   #vector containing the size of each corresponded contour

        # is there a better way of doing this ?
        cdef int32_t[:,:] cc # typed memoryview
        for c in contours:
            cc = c
            contour_ptrs.push_back( &cc[0,0]  )
            contour_sizes.push_back( c.size )

        self.thisptr.add_observation(ellipse, contour_ptrs, contour_sizes  )
        self.num_observations += 1

    def print_ellipse(self,index):
        # self.thisptr.print_ellipse(index)
        cdef Ellipse2D[double] ellipse = self.thisptr.pupils[index].observation.ellipse
        return "Ellipse ( center = [%s, %s], major_radius = %.3f, minor_radius = %.3f, angle = %.3f)"%(ellipse.center[0],ellipse.center[1],ellipse.major_radius,ellipse.minor_radius,ellipse.angle)

    def print_eye(self):
        cdef Sphere[double] eye = self.thisptr.eye
        return "Sphere ( center = [%s, %s, %s], radius = %s)" %(eye.center[0],eye.center[1],eye.center[2],eye.radius)

    # def get_projected_eye_center(self):
    #     cdef Ellipse2D[double] projected_eye = self.thisptr.projected_eye
    #     return (projected_eye.center[0],projected_eye.center[1])

    def get_observation(self,index):
        cdef EyeModelFitter.Pupil p = self.thisptr.pupils[index]
        # returning (Ellipse, Params, Cicle). Ellipse = ([x,y],major,minor,angle). Params = (theta,psi,r)
        # Circle = (center[x,y,z], normal[x,y,z], radius)
        return PyObservation( (p.observation.ellipse.center[0],p.observation.ellipse.center[1]), p.observation.ellipse.major_radius,p.observation.ellipse.minor_radius,p.observation.ellipse.angle,
            p.params.theta,p.params.psi,p.params.radius,
            (p.circle.center[0],p.circle.center[1],p.circle.center[2]),
            (p.circle.normal[0],p.circle.normal[1],p.circle.normal[2]),
            p.circle.radius)

    def get_last_observations(self,count=1):
        cdef EyeModelFitter.Pupil p
        count = min(self.thisptr.pupils.size() , count )
        for i in xrange(self.thisptr.pupils.size()-count,self.thisptr.pupils.size()):
            p = self.thisptr.pupils[i]
            yield  PyObservation( (p.observation.ellipse.center[0],p.observation.ellipse.center[1]), p.observation.ellipse.major_radius,p.observation.ellipse.minor_radius,p.observation.ellipse.angle,
            p.params.theta,p.params.psi,p.params.radius,
            (p.circle.center[0],p.circle.center[1],p.circle.center[2]),
            (p.circle.normal[0],p.circle.normal[1],p.circle.normal[2]),
            p.circle.radius)

    def get_last_contours(self):
        if self.thisptr.pupils.size() == 0:
            return []

        cdef EyeModelFitter.Pupil p = self.thisptr.pupils.back()
        contours = []
        for contour in p.unprojected_contours:
            c = []
            for point in contour:
                c.append([point[0],point[1],point[2]])
            contours.append(c)

        return contours

    def get_last_unwrapped_contours(self):
        if self.thisptr.pupils.size() == 0:
            return []

        cdef EyeModelFitter.Pupil p = self.thisptr.pupils.back()
        contours = []
        for contour in p.unwrapped_contours:
            c = []
            for point in contour:
                c.append([point[0],point[1]])
            contours.append(c)

        return contours

    def get_all_pupil_observations(self):
        cdef EyeModelFitter.Pupil p
        for p in self.thisptr.pupils:
            yield PyObservation( (p.observation.ellipse.center[0],p.observation.ellipse.center[1]), p.observation.ellipse.major_radius,p.observation.ellipse.minor_radius,p.observation.ellipse.angle,
            p.params.theta,p.params.psi,p.params.radius,
            (p.circle.center[0],p.circle.center[1],p.circle.center[2]),
            (p.circle.normal[0],p.circle.normal[1],p.circle.normal[2]),
            p.circle.radius)

    property model_version:
        def __get__(self):
            return self.thisptr.model_version

    # property scale:
    #     def __get__(self):
    #         return self.thisptr.scale

    property eye:
        def __get__(self):
            cdef Sphere[double] eye = self.thisptr.eye
            temp = ((eye.center[0],eye.center[1],eye.center[2]),eye.radius)
            return temp

    property focal_length:
        def __get__(self):
            return self.thisptr.focal_length

    # property projected_eye:
    #     def __get__(self):
    #         cdef Ellipse2D[double] projected_eye = self.thisptr.projected_eye
    #         temp = ((projected_eye.center[0],projected_eye.center[1]),
    #             projected_eye.major_radius,projected_eye.minor_radius,projected_eye.angle)



