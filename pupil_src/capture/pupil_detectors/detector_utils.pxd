
# cython: profile=False
from detector cimport *
from methods import  normalize

cdef inline convertToPythonResult( Detector_2D_Result& result, object frame, object roi ):

    cdef float pi = 3.14159265359
    e = ((result.ellipse.center[0],result.ellipse.center[1]), (result.ellipse.minor_radius * 2.0 ,result.ellipse.major_radius * 2.0) , result.ellipse.angle * 180 / pi - 90 )
    py_result = {}
    py_result['confidence'] = result.confidence
    py_result['ellipse'] = e
    py_result['major'] = max(e[1])
    py_result['diameter'] = max(e[1])
    py_result['minor'] = min(e[1])
    py_result['axes'] = e[1]
    py_result['angle'] = e[2]

    norm_center = normalize(e[0],(frame.width, frame.height),flip_y=True)
    py_result['norm_pos'] = norm_center
    py_result['center'] = e[0]
    py_result['timestamp'] = frame.timestamp
    return py_result
