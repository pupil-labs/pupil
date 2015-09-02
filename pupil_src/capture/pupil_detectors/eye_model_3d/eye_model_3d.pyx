from libcpp.pair cimport pair
from libcpp.vector cimport vector

cdef extern from "singleeyefitter/singleeyefitter.h" namespace "singleeyefitter":

    cdef cppclass Ellipse2D[T]:
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
        void add_observation(double center_x,double center_y, double major_radius, double minor_radius, double angle)

        cppclass PupilParams:
            float theta
            float psi
            float radius

        cppclass Pupil:
            Pupil() except +
            Ellipse2D[double] ellipse  
            PupilParams params
            Circle3D[double] circle

        #variables
        float model_version
        vector[Pupil] pupils
        Sphere[double] eye
        Ellipse2D[double] projected_eye #technically only need center, not whole ellipse. can optimize here
        float scale

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

    cdef cppclass ParametrizedLine3d "Eigen::ParametrizedLine<double, 3>":
        ParametrizedLine3d() except +
        ParametrizedLine3d(Matrix31d origin, Matrix31d direction)        

cdef class PyEyeModelFitter:
    cdef EyeModelFitter *thisptr
    cdef public int counter
    cdef public int num_observations
    # def __cinit__(self, focal_length):
    #     self.thisptr = new EyeModelFitter(focal_length)
    def __cinit__(self, focal_length, x_disp, y_disp):
        self.thisptr = new EyeModelFitter(focal_length, x_disp, y_disp)

    def __init__(self,focal_length, x_disp, y_disp):
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

    def add_observation(self,center,major_radius,minor_radius,angle):
        #standard way of adding an observation
        self.thisptr.add_observation(center[0], center[1],major_radius,minor_radius,angle)
        self.num_observations += 1

    def add_pupil_labs_observation(self,e_dict):
        # a special method for taking in arguments from eye.py
        a,b = e_dict['axes']
        a,b = e_dict['axes']
        if a > b:
            major_radius = a/2
            minor_radius = b/2
            angle = e_dict['angle']*3.1415926535/180
        else:
            major_radius = b/2
            minor_radius = a/2
            angle = (e_dict['angle']+90)*3.1415926535/180 # not importing np just for pi constant
        # print e_dict['center'][0],e_dict['center'][1],major_radius,minor_radius,angle
        self.thisptr.add_observation(e_dict['center'][0],e_dict['center'][1],major_radius,minor_radius,angle)
        self.num_observations += 1

    def print_ellipse(self,index):
        # self.thisptr.print_ellipse(index)
        cdef Ellipse2D[double] ellipse = self.thisptr.pupils[index].ellipse
        return "Ellipse ( center = [%s, %s], major_radius = %.3f, minor_radius = %.3f, angle = %.3f)"%(ellipse.center[0],ellipse.center[1],ellipse.major_radius,ellipse.minor_radius,ellipse.angle)

    def print_eye(self):
        cdef Sphere[double] eye = self.thisptr.eye
        return "Sphere ( center = [%s, %s, %s], radius = %s)" %(eye.center[0],eye.center[1],eye.center[2],eye.radius)

    def get_projected_eye_center(self):
        cdef Ellipse2D[double] projected_eye = self.thisptr.projected_eye
        return (projected_eye.center[0],projected_eye.center[1])

    def get_pupil_observation(self,index):
        cdef EyeModelFitter.Pupil p = self.thisptr.pupils[index]
        # returning (Ellipse, Params, Cicle). Ellipse = ([x,y],major,minor,angle). Params = (theta,psi,r)
        # Circle = (center[x,y,z], normal[x,y,z], radius)
        return (((p.ellipse.center[0],p.ellipse.center[1]),
            p.ellipse.major_radius,p.ellipse.minor_radius,p.ellipse.angle),
            (p.params.theta,p.params.psi,p.params.radius),
            ((p.circle.center[0],p.circle.center[1],p.circle.center[2]),
            (p.circle.normal[0],p.circle.normal[1],p.circle.normal[2]),
            p.circle.radius))

    def get_last_pupil_observations(self,number):
        cdef EyeModelFitter.Pupil p
        for i in xrange(self.thisptr.pupils.size()-number,self.thisptr.pupils.size()):
            p = self.thisptr.pupils[i]
            yield (((p.ellipse.center[0],p.ellipse.center[1]),
                p.ellipse.major_radius,p.ellipse.minor_radius,p.ellipse.angle),
                (p.params.theta,p.params.psi,p.params.radius),
                ((p.circle.center[0],p.circle.center[1],p.circle.center[2]),
                (p.circle.normal[0],p.circle.normal[1],p.circle.normal[2]),
                p.circle.radius))


    def get_last_pupil_observation(self):
        cdef EyeModelFitter.Pupil p = self.thisptr.pupils.back()
        return (((p.ellipse.center[0],p.ellipse.center[1]),
            p.ellipse.major_radius,p.ellipse.minor_radius,p.ellipse.angle),
            (p.params.theta,p.params.psi,p.params.radius),
            ((p.circle.center[0],p.circle.center[1],p.circle.center[2]),
            (p.circle.normal[0],p.circle.normal[1],p.circle.normal[2]),
            p.circle.radius))        

    def get_all_pupil_observations(self):
        cdef EyeModelFitter.Pupil p
        for p in self.thisptr.pupils:
            yield (((p.ellipse.center[0],p.ellipse.center[1]),
                p.ellipse.major_radius,p.ellipse.minor_radius,p.ellipse.angle),
                (p.params.theta,p.params.psi,p.params.radius),
                ((p.circle.center[0],p.circle.center[1],p.circle.center[2]),
                (p.circle.normal[0],p.circle.normal[1],p.circle.normal[2]),
                p.circle.radius))

    def intersect_contour_with_eye(self,float[:,:] contour):
        #eye is sphere.
        # self.thisptr.intersect_contour_with_eye(contour)
        pass
        # cdef Matrix31d direction
        # cdef Matrix31d origin = Matrix31d(0,0,0)
        # cdef ParametrizedLine3d line
        # cdef pair[Matrix31d,Matrix31d] intersect_pts
        # for point in contour:
        #     direction = Matrix31d(point[0],point[1],point[2])
        #     line = ParametrizedLine3d(origin,direction)
        #     try:
        #         intersect_pts = intersect(line,self.thisptr.eye)
        #     except:
        #         pass
        #     finally:
        #         print intersect_pts.first[0],intersect_pts.first[1],intersect_pts.first[2]

    property model_version:
        def __get__(self):
            return self.thisptr.model_version

    property scale:
        def __get__(self):
            return self.thisptr.scale

    property eye:
        def __get__(self):
            cdef Sphere[double] eye = self.thisptr.eye
            temp = ((eye.center[0],eye.center[1],eye.center[2]),eye.radius)
            return temp

    property projected_eye:
        def __get__(self):
            cdef Ellipse2D[double] projected_eye = self.thisptr.projected_eye
            temp = ((projected_eye.center[0],projected_eye.center[1]),
                projected_eye.major_radius,projected_eye.minor_radius,projected_eye.angle)



