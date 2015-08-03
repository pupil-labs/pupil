from pyglui cimport cygl

cdef extern from "singleeyefitter/singleeyefitter.h" namespace "singleeyefitter":
    cdef cppclass EyeModelFitter:
        EyeModelFitter(double focal_length, double region_band_width, double region_step_epsilon)
        void reset()
        void initialise_model()


cdef class PyEyeModelFitter:
    cdef EyeModelFitter *thisptr
    def __cinit__(self, focal_length):
        self.thisptr = new EyeModelFitter(focal_length,1,1)

    def __init__(self,focal_length):
        pass
    def __dealloc__(self):
        del self.thisptr

    def reset(self):
        self.thisptr.reset()

    def initialise_model(self):
        self.thisptr.initialise_model()


