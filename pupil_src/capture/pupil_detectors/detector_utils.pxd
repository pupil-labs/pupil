
# cython: profile=False
from detector cimport *
from methods import  normalize
from numpy.math cimport PI

cdef extern from 'singleeyefitter/mathHelper.h' namespace 'singleeyefitter::math':

    Matrix21d cart2sph( Matrix31d& m )


cdef inline convertToPythonResult( Detector2DResult& result, object frame, object roi ):


    ellipse = {}
    ellipse['center'] = (result.ellipse.center[0],result.ellipse.center[1])
    ellipse['axes'] =  (result.ellipse.minor_radius * 2.0 ,result.ellipse.major_radius * 2.0)
    ellipse['angle'] = result.ellipse.angle * 180.0 / PI - 90.0

    py_result = {}
    py_result['confidence'] = result.confidence
    py_result['ellipse'] = ellipse
    py_result['diameter'] = max(ellipse['axes'])

    norm_center = normalize( ellipse['center'] , (frame.width, frame.height),flip_y=True)
    py_result['norm_pos'] = norm_center
    py_result['timestamp'] = frame.timestamp
    py_result['method'] = '2D c++'

    return py_result

cdef inline add3DResult( Detector3DResult& result, object py_2D_result, object frame    ):

    circle = {}
    circle['center'] =  (result.circle.center[0],result.circle.center[1], result.circle.center[2])
    circle['normal'] =  (result.circle.normal[0],result.circle.normal[1], result.circle.normal[1])
    circle['radius'] =  result.circle.radius
    py_2D_result['circle3D'] = circle

    py_2D_result['confidence'] = result.confidence

    if result.ellipse.minor_radius != 0.0 and result.ellipse.major_radius != 0.0 :
        ellipse = {}
        ellipse['center'] = (result.ellipse.center[0] + frame.width / 2.0 ,frame.height / 2.0  -  result.ellipse.center[1])
        ellipse['axes'] =  (result.ellipse.minor_radius * 2.0 ,result.ellipse.major_radius * 2.0)
        ellipse['angle'] = - (result.ellipse.angle * 180.0 / PI - 90.0)

        py_2D_result['ellipse'] = ellipse
        norm_center = normalize( ellipse['center'] , (frame.width, frame.height),flip_y=True)
        py_2D_result['norm_pos'] = norm_center

    if not result.sphere.center.isZero() and result.sphere.radius != 0.0:
        sphere = {}
        sphere['center'] =  (result.sphere.center[0],result.sphere.center[1], result.sphere.center[2])
        sphere['radius'] =  result.sphere.radius
        py_2D_result['sphere'] = sphere


    py_2D_result['modelConfidence'] = result.modelConfidence
    py_2D_result['modelID'] = result.modelID

    py_2D_result['diameter_mm'] = result.circle.radius * 2.0

    coords = cart2sph(result.circle.normal)
    py_2D_result['theta'] = coords[0]
    py_2D_result['phi'] = coords[1]
    py_2D_result['method'] = '3D c++'



cdef inline prepareForVisualization3D(  Detector3DResult& result ):

    py_visualizationResult = {}

    py_visualizationResult['edges'] = getEdges(result)
    py_visualizationResult['circle'] = getCircle(result);

    models = []
    for model in result.models:
        props = {}
        props['binPositions'] = getBinPositions(model)
        props['sphere'] = getSphere(model)
        props['initialSphere'] = getInitialSphere(model)
        props['maturity'] = model.maturity
        props['fit'] = model.fit
        props['performance'] = model.performance
        props['modelID'] = model.modelID
        models.append(props)

    py_visualizationResult['models'] = models;

    return py_visualizationResult


cdef inline getBinPositions( ModelDebugProperties& result ):
    if result.binPositions.size() == 0:
        return []
    positions = []
    eyePosition = result.sphere.center
    eyeRadius = result.sphere.radius
    #bins are on a unit sphere
    for point in result.binPositions:
        positions.append([point[0]*eyeRadius+eyePosition[0],point[1]*eyeRadius+eyePosition[1],point[2]*eyeRadius+eyePosition[2]])
    return positions

cdef inline getEdges( Detector3DResult& result ):
    if result.edges.size() == 0:
        return []
    edges = []
    for point in result.edges:
        edges.append([point[0],point[1],point[2]])
    return edges


cdef inline getCircle(const Detector3DResult& result):
    center = result.circle.center
    radius = result.circle.radius
    normal = result.circle.normal
    return [ [center[0],center[1],center[2]], [normal[0],normal[1],normal[2]], radius ]


cdef inline getSphere(const ModelDebugProperties& result ):
    sphere = result.sphere
    return [ [sphere.center[0],sphere.center[1],sphere.center[2]],sphere.radius]

cdef inline getInitialSphere(const ModelDebugProperties& result ):
    sphere = result.initialSphere
    return [ [sphere.center[0],sphere.center[1],sphere.center[2]],sphere.radius]
