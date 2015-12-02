
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

cdef inline prepareForVisualization3D(  Detector_3D_Result& result ):

    py_visualizationResult = {}

    py_visualizationResult['edges'] = getEdges(result);
    py_visualizationResult['binPositions'] = getBinPositions(result);
    py_visualizationResult['circle'] = getCircle(result);
    py_visualizationResult['contours'] = getContours(result.contours);
    py_visualizationResult['fittedContours'] = getContours(result.fittedCircleContours);
    py_visualizationResult['sphere'] = getSphere(result);
    py_visualizationResult['initialSphere'] = getInitialSphere(result);


    return py_visualizationResult


cdef inline getBinPositions( Detector_3D_Result& result ):
    if result.binPositions.size() == 0:
        return []
    positions = []
    eyePosition = result.sphere.center
    eyeRadius = result.sphere.radius
    #bins are on a unit sphere
    for point in result.binPositions:
        positions.append([point[0]*eyeRadius+eyePosition[0],point[1]*eyeRadius+eyePosition[1],point[2]*eyeRadius+eyePosition[2]])
    return positions

cdef inline getEdges( Detector_3D_Result& result ):
    if result.edges.size() == 0:
        return []
    edges = []
    for point in result.edges:
        edges.append([point[0],point[1],point[2]])
    return edges


cdef inline getCircle(const Detector_3D_Result& result):
    center = result.circle.center
    radius = result.circle.radius
    normal = result.circle.normal
    return [ [center[0],center[1],center[2]], [normal[0],normal[1],normal[2]], radius ]


cdef inline getContours( Contours3D con):

    contours = []
    for contour in con:
        c = []
        for point in contour:
            c.append([point[0],point[1],point[2]])
        contours.append(c)

    return contours


cdef inline getSphere(const Detector_3D_Result& result ):
    sphere = result.sphere
    return [ [sphere.center[0],sphere.center[1],sphere.center[2]],sphere.radius]

cdef inline getInitialSphere(const Detector_3D_Result& result ):
    sphere = result.initialSphere
    return [ [sphere.center[0],sphere.center[1],sphere.center[2]],sphere.radius]
