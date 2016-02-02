'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2

from methods import undistort_unproject_pts
#logging
import logging
logger = logging.getLogger(__name__)


def calibrate_2d_polynomial(cal_pt_cloud,screen_size=(2,2),threshold = 35, binocular=False):
    """
    we do a simple two pass fitting to a pair of bi-variate polynomials
    return the function to map vector
    """
    # fit once using all avaiable data
    model_n = 7
    if binocular:
        model_n = 13

    cal_pt_cloud = np.array(cal_pt_cloud)

    cx,cy,err_x,err_y = fit_poly_surface(cal_pt_cloud,model_n)
    err_dist,err_mean,err_rms = fit_error_screen(err_x,err_y,screen_size)
    if cal_pt_cloud[err_dist<=threshold].shape[0]: #did not disregard all points..
        # fit again disregarding extreme outliers
        cx,cy,new_err_x,new_err_y = fit_poly_surface(cal_pt_cloud[err_dist<=threshold],model_n)
        map_fn = make_map_function(cx,cy,model_n)
        new_err_dist,new_err_mean,new_err_rms = fit_error_screen(new_err_x,new_err_y,screen_size)

        logger.info('first iteration. root-mean-square residuals: %s, in pixel' %err_rms)
        logger.info('second iteration: ignoring outliers. root-mean-square residuals: %s in pixel',new_err_rms)

        logger.info('used %i data points out of the full dataset %i: subset is %i percent' \
            %(cal_pt_cloud[err_dist<=threshold].shape[0], cal_pt_cloud.shape[0], \
            100*float(cal_pt_cloud[err_dist<=threshold].shape[0])/cal_pt_cloud.shape[0]))

        return map_fn,err_dist<=threshold,(cx,cy,model_n)

    else: # did disregard all points. The data cannot be represented by the model in a meaningful way:
        map_fn = make_map_function(cx,cy,model_n)
        logger.info('First iteration. root-mean-square residuals: %s in pixel, this is bad!'%err_rms)
        logger.warning('The data cannot be represented by the model in a meaningfull way.')
        return map_fn,err_dist<=threshold,(cx,cy,model_n)



def fit_poly_surface(cal_pt_cloud,n=7):
    M = make_model(cal_pt_cloud,n)
    U,w,Vt = np.linalg.svd(M[:,:n],full_matrices=0)
    V = Vt.transpose()
    Ut = U.transpose()
    pseudINV = np.dot(V, np.dot(np.diag(1/w), Ut))
    cx = np.dot(pseudINV, M[:,n])
    cy = np.dot(pseudINV, M[:,n+1])
    # compute model error in world screen units if screen_res specified
    err_x=(np.dot(M[:,:n],cx)-M[:,n])
    err_y=(np.dot(M[:,:n],cy)-M[:,n+1])
    return cx,cy,err_x,err_y

def fit_error_screen(err_x,err_y,(screen_x,screen_y)):
    err_x *= screen_x/2.
    err_y *= screen_y/2.
    err_dist=np.sqrt(err_x*err_x + err_y*err_y)
    err_mean=np.sum(err_dist)/len(err_dist)
    err_rms=np.sqrt(np.sum(err_dist*err_dist)/len(err_dist))
    return err_dist,err_mean,err_rms

def fit_error_angle(err_x,err_y ) :
    err_x *= 2. * np.pi
    err_y *= 2. * np.pi
    err_dist=np.sqrt(err_x*err_x + err_y*err_y)
    err_mean=np.sum(err_dist)/len(err_dist)
    err_rms=np.sqrt(np.sum(err_dist*err_dist)/len(err_dist))
    return err_dist,err_mean,err_rms

def make_model(cal_pt_cloud,n=7):
    n_points = cal_pt_cloud.shape[0]

    if n==3:
        X=cal_pt_cloud[:,0]
        Y=cal_pt_cloud[:,1]
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,2]
        ZY=cal_pt_cloud[:,3]
        M=np.array([X,Y,Ones,ZX,ZY]).transpose()

    elif n==5:
        X0=cal_pt_cloud[:,0]
        Y0=cal_pt_cloud[:,1]
        X1=cal_pt_cloud[:,2]
        Y1=cal_pt_cloud[:,3]
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,4]
        ZY=cal_pt_cloud[:,5]
        M=np.array([X0,Y0,X1,Y1,Ones,ZX,ZY]).transpose()

    elif n==7:
        X=cal_pt_cloud[:,0]
        Y=cal_pt_cloud[:,1]
        XX=X*X
        YY=Y*Y
        XY=X*Y
        XXYY=XX*YY
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,2]
        ZY=cal_pt_cloud[:,3]
        M=np.array([X,Y,XX,YY,XY,XXYY,Ones,ZX,ZY]).transpose()

    elif n==9:
        X=cal_pt_cloud[:,0]
        Y=cal_pt_cloud[:,1]
        XX=X*X
        YY=Y*Y
        XY=X*Y
        XXYY=XX*YY
        XXY=XX*Y
        YYX=YY*X
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,2]
        ZY=cal_pt_cloud[:,3]
        M=np.array([X,Y,XX,YY,XY,XXYY,XXY,YYX,Ones,ZX,ZY]).transpose()

    elif n==13:
        X0=cal_pt_cloud[:,0]
        Y0=cal_pt_cloud[:,1]
        X1=cal_pt_cloud[:,2]
        Y1=cal_pt_cloud[:,3]
        XX0=X0*X0
        YY0=Y0*Y0
        XY0=X0*Y0
        XXYY0=XX0*YY0
        XX1=X1*X1
        YY1=Y1*Y1
        XY1=X1*Y1
        XXYY1=XX1*YY1
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,4]
        ZY=cal_pt_cloud[:,5]
        M=np.array([X0,Y0,X1,Y1,XX0,YY0,XY0,XXYY0,XX1,YY1,XY1,XXYY1,Ones,ZX,ZY]).transpose()

    elif n==17:
        X0=cal_pt_cloud[:,0]
        Y0=cal_pt_cloud[:,1]
        X1=cal_pt_cloud[:,2]
        Y1=cal_pt_cloud[:,3]
        XX0=X0*X0
        YY0=Y0*Y0
        XY0=X0*Y0
        XXYY0=XX0*YY0
        XX1=X1*X1
        YY1=Y1*Y1
        XY1=X1*Y1
        XXYY1=XX1*YY1

        X0X1 = X0*X1
        X0Y1 = X0*Y1
        Y0X1 = Y0*X1
        Y0Y1 = Y0*Y1

        Ones=np.ones(n_points)

        ZX=cal_pt_cloud[:,4]
        ZY=cal_pt_cloud[:,5]
        M=np.array([X0,Y0,X1,Y1,XX0,YY0,XY0,XXYY0,XX1,YY1,XY1,XXYY1,X0X1,X0Y1,Y0X1,Y0Y1,Ones,ZX,ZY]).transpose()

    else:
        raise Exception("ERROR: Model n needs to be 3, 5, 7 or 9")
    return M


def make_map_function(cx,cy,n):
    if n==3:
        def fn((X,Y)):
            x2 = cx[0]*X + cx[1]*Y +cx[2]
            y2 = cy[0]*X + cy[1]*Y +cy[2]
            return x2,y2

    elif n==5:
        def fn((X0,Y0),(X1,Y1)):
            #        X0        Y0        X1        Y1        Ones
            x2 = cx[0]*X0 + cx[1]*Y0 + cx[2]*X1 + cx[3]*Y1 + cx[4]
            y2 = cy[0]*X0 + cy[1]*Y0 + cy[2]*X1 + cy[3]*Y1 + cy[4]
            return x2,y2

    elif n==7:
        def fn((X,Y)):
            x2 = cx[0]*X + cx[1]*Y + cx[2]*X*X + cx[3]*Y*Y + cx[4]*X*Y + cx[5]*Y*Y*X*X +cx[6]
            y2 = cy[0]*X + cy[1]*Y + cy[2]*X*X + cy[3]*Y*Y + cy[4]*X*Y + cy[5]*Y*Y*X*X +cy[6]
            return x2,y2

    elif n==9:
        def fn((X,Y)):
            #          X         Y         XX         YY         XY         XXYY         XXY         YYX         Ones
            x2 = cx[0]*X + cx[1]*Y + cx[2]*X*X + cx[3]*Y*Y + cx[4]*X*Y + cx[5]*Y*Y*X*X + cx[6]*Y*X*X + cx[7]*Y*Y*X + cx[8]
            y2 = cy[0]*X + cy[1]*Y + cy[2]*X*X + cy[3]*Y*Y + cy[4]*X*Y + cy[5]*Y*Y*X*X + cy[6]*Y*X*X + cy[7]*Y*Y*X + cy[8]
            return x2,y2

    elif n==13:
        def fn((X0,Y0),(X1,Y1)):
            #        X0        Y0        X1         Y1            XX0        YY0            XY0            XXYY0                XX1            YY1            XY1            XXYY1        Ones
            x2 = cx[0]*X0 + cx[1]*Y0 + cx[2]*X1 + cx[3]*Y1 + cx[4]*X0*X0 + cx[5]*Y0*Y0 + cx[6]*X0*Y0 + cx[7]*X0*X0*Y0*Y0 + cx[8]*X1*X1 + cx[9]*Y1*Y1 + cx[10]*X1*Y1 + cx[11]*X1*X1*Y1*Y1 + cx[12]
            y2 = cy[0]*X0 + cy[1]*Y0 + cy[2]*X1 + cy[3]*Y1 + cy[4]*X0*X0 + cy[5]*Y0*Y0 + cy[6]*X0*Y0 + cy[7]*X0*X0*Y0*Y0 + cy[8]*X1*X1 + cy[9]*Y1*Y1 + cy[10]*X1*Y1 + cy[11]*X1*X1*Y1*Y1 + cy[12]
            return x2,y2

    elif n==17:
        def fn((X0,Y0),(X1,Y1)):
            #        X0        Y0        X1         Y1            XX0        YY0            XY0            XXYY0                XX1            YY1            XY1            XXYY1            X0X1            X0Y1            Y0X1        Y0Y1           Ones
            x2 = cx[0]*X0 + cx[1]*Y0 + cx[2]*X1 + cx[3]*Y1 + cx[4]*X0*X0 + cx[5]*Y0*Y0 + cx[6]*X0*Y0 + cx[7]*X0*X0*Y0*Y0 + cx[8]*X1*X1 + cx[9]*Y1*Y1 + cx[10]*X1*Y1 + cx[11]*X1*X1*Y1*Y1 + cx[12]*X0*X1 + cx[13]*X0*Y1 + cx[14]*Y0*X1 + cx[15]*Y0*Y1 + cx[16]
            y2 = cy[0]*X0 + cy[1]*Y0 + cy[2]*X1 + cy[3]*Y1 + cy[4]*X0*X0 + cy[5]*Y0*Y0 + cy[6]*X0*Y0 + cy[7]*X0*X0*Y0*Y0 + cy[8]*X1*X1 + cy[9]*Y1*Y1 + cy[10]*X1*Y1 + cy[11]*X1*X1*Y1*Y1 + cy[12]*X0*X1 + cy[13]*X0*Y1 + cy[14]*Y0*X1 + cy[15]*Y0*Y1 + cy[16]
            return x2,y2

    else:
        raise Exception("ERROR: Model n needs to be 3, 5, 7 or 9")

    return fn


def closest_matches_binocular(ref_pts, pupil_pts,max_dispersion=1/15.):
    '''
    get pupil positions closest in time to ref points.
    return list of dict with matching ref, pupil0 and pupil1 data triplets.
    '''
    ref = ref_pts

    pupil0 = [p for p in pupil_pts if p['id']==0]
    pupil1 = [p for p in pupil_pts if p['id']==1]

    pupil0_ts = np.array([p['timestamp'] for p in pupil0])
    pupil1_ts = np.array([p['timestamp'] for p in pupil1])


    def find_nearest_idx(array,value):
        idx = np.searchsorted(array, value, side="left")
        try:
            if abs(value - array[idx-1]) < abs(value - array[idx]):
                return idx-1
            else:
                return idx
        except IndexError:
            return idx-1

    matched = []

    if pupil0 and pupil1:
        for r in ref_pts:
            closest_p0_idx = find_nearest_idx(pupil0_ts,r['timestamp'])
            closest_p0 = pupil0[closest_p0_idx]
            closest_p1_idx = find_nearest_idx(pupil1_ts,r['timestamp'])
            closest_p1 = pupil1[closest_p1_idx]

            dispersion = max(closest_p0['timestamp'],closest_p1['timestamp'],r['timestamp']) - min(closest_p0['timestamp'],closest_p1['timestamp'],r['timestamp'])
            if dispersion < max_dispersion:
                matched.append({'ref':r,'pupil0':closest_p0, 'pupil1':closest_p1})
            else:
                print "to far."
    return matched


def closest_matches_monocular(ref_pts, pupil_pts,max_dispersion=1/15.):
    '''

    get pupil positions closest in time to ref points.
    return list of dict with matching ref and pupil datum.

    if your data is binocular use:
    pupil0 = [p for p in pupil_pts if p['id']==0]
    pupil1 = [p for p in pupil_pts if p['id']==1]
    to get the desired eye and pass it as pupil_pts
    '''

    ref = ref_pts
    pupil0 = pupil_pts
    pupil0_ts = np.array([p['timestamp'] for p in pupil0])

    def find_nearest_idx(array,value):
        idx = np.searchsorted(array, value, side="left")
        try:
            if abs(value - array[idx-1]) < abs(value - array[idx]):
                return idx-1
            else:
                return idx
        except IndexError:
            return idx-1

    matched = []
    if pupil0:
        for r in ref_pts:
            closest_p0_idx = find_nearest_idx(pupil0_ts,r['timestamp'])
            closest_p0 = pupil0[closest_p0_idx]
            dispersion = max(closest_p0['timestamp'],r['timestamp']) - min(closest_p0['timestamp'],r['timestamp'])
            if dispersion < max_dispersion:
                matched.append({'ref':r,'pupil':closest_p0})
            else:
                pass
    return matched


def preprocess_2d_data_monocular(matched_data):
    cal_data = []
    for pair in matched_data:
        ref,pupil = pair['ref'],pair['pupil']
        cal_data.append( (pupil["norm_pos"][0], pupil["norm_pos"][1],ref['norm_pos'][0],ref['norm_pos'][1]) )
    return cal_data

def preprocess_2d_data_binocular(matched_data):
    cal_data = []
    for triplet in matched_data:
        ref,p0,p1 = triplet['ref'],triplet['pupil0'],triplet['pupil1']
        data_pt = p0["norm_pos"][0], p0["norm_pos"][1],p1["norm_pos"][0], p1["norm_pos"][1],ref['norm_pos'][0],ref['norm_pos'][1]
        cal_data.append( data_pt )
    return cal_data

def preprocess_3d_data_monocular(matched_data, camera_intrinsics , calibration_distance):
    camera_matrix = camera_intrinsics["camera_matrix"]
    dist_coefs = camera_intrinsics["dist_coefs"]

    cal_data = []
    for pair in matched_data:
        ref,pupil = pair['ref'],pair['pupil']
        try:
            # taking the pupil normal as line of sight vector
            # we multiply by a fixed (assumed) distace and
            # add the sphere pos to get the 3d gaze point in eye camera 3d coords
            sphere_pos  = np.array(pupil['sphere']['center'])
            gaze_pt_3d = np.array(pupil['circle3D']['normal']) * calibration_distance + sphere_pos

            # projected point uv to normal ray vector of camera
            ref_vector =  undistort_unproject_pts(ref['screen_pos'] , camera_matrix, dist_coefs).tolist()[0]
            ref_vector = ref_vector / np.linalg.norm(ref_vector)
            # assuming a fixed (assumed) distance we get a 3d point in world camera 3d coords.
            ref_pt_3d = ref_vector*calibration_distance


            point_pair_3d = tuple(gaze_pt_3d) , ref_pt_3d
            cal_data.append(point_pair_3d)
        except KeyError as e:
            # this pupil data point did not have 3d detected data.
            pass

    return cal_data


def preprocess_3d_data_binocular(matched_data, camera_intrinsics , calibration_distance):

    camera_matrix = camera_intrinsics["camera_matrix"]
    dist_coefs = camera_intrinsics["dist_coefs"]

    cal_data = []
    for triplet in matched_data:
        ref,p0,p1 = triplet['ref'],triplet['pupil0'],triplet['pupil1']
        try:
            # taking the pupil normal as line of sight vector
            # we multiply by a fixed (assumed) distance and
            # add the sphere pos to get the 3d gaze point in eye camera 3d coords
            sphere_pos0 = np.array(p0['sphere']['center'])
            gaze_pt0 = np.array(p0['circle3D']['normal']) * calibration_distance + sphere_pos0


            sphere_pos1 = np.array(p1['sphere']['center'])
            gaze_pt1 = np.array(p1['circle3D']['normal']) * calibration_distance + sphere_pos1


            # projected point uv to normal ray vector of camera
            ref_vector =  undistort_unproject_pts(ref['screen_pos'] , camera_matrix, dist_coefs).tolist()[0]
            ref_vector = ref_vector / np.linalg.norm(ref_vector)
            # assuming a fixed (assumed) distance we get a 3d point in world camera 3d coords.
            ref_pt_3d = ref_vector*calibration_distance


            point_triple_3d = tuple(gaze_pt0), tuple(gaze_pt1) , ref_pt_3d
            cal_data.append(point_triple_3d)
        except KeyError as e:
            # this pupil data point did not have 3d detected data.
            pass

    return cal_data

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array

    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    # print t

    return R, t


def calculate_residual_3D_Points( ref_points, gaze_points, eye_to_world_matrix ):

    average_distance = 0.0
    distance_variance = 0.0
    transformed_gaze_points = []

    for p in gaze_points:
        point = np.zeros(4)
        point[:3] = p
        point[3] = 1.0
        point = eye_to_world_matrix.dot(point)
        point = np.squeeze(np.asarray(point))
        transformed_gaze_points.append( point[:3] )

    for(a,b) in zip( ref_points, transformed_gaze_points):
        average_distance += np.linalg.norm(a-b)

    average_distance /= len(ref_points)

    for(a,b) in zip( ref_points, transformed_gaze_points):
        distance_variance += (np.linalg.norm(a-b) - average_distance)**2

    distance_variance /= len(ref_points)

    return average_distance, distance_variance



def get_transformation_from_point_set( cal_pt_cloud, camera_matrix , dist_coefs ):
    '''
    this does not yield good results. Instead we set a fixed distance and use a rigit 3d transform.
    '''


    object_points = np.array(cal_pt_cloud[:,0].tolist(), dtype=np.float32)
    image_points =  np.array(cal_pt_cloud[:,1].tolist(), dtype=np.float32)
    image_points = image_points.reshape(-1,1,2)
    #result =  cv2.estimateAffine3D(src, dst)
    #print object_points
    #print image_points

    result = cv2.solvePnP( object_points , image_points, camera_matrix, dist_coefs, flags=cv2.CV_ITERATIVE)
    return  result[1], result[2]

    # print image_points.size
    # print image_points
    # result = cv2.solvePnPRansac( object_points , image_points, camera_matrix, dist_coefs , iterationsCount = 10000, reprojectionError = 3, minInliersCount = int(image_points.size * 0.7) )
    # print 'got inliers: ' , result[2].size
    # return  result[0], result[1]



#NOTUSED
def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.

    v0 and v1 are shape (ndims, \*) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.

    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.

    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[   0.14549,    0.00062,  675.50008],
           [   0.00048,    0.14094,   53.24971],
           [   0.     ,    0.     ,    1.     ]])
    >>> T = translation_matrix(numpy.random.random(3)-0.5)
    >>> R = random_rotation_matrix(numpy.random.random(3))
    >>> S = scale_matrix(random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (numpy.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = numpy.dot(M, v0)
    >>> v0[:3] += numpy.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True

    More examples in superimposition_matrix()

    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=True)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -numpy.mean(v0, axis=1)
    M0 = numpy.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -numpy.mean(v1, axis=1)
    M1 = numpy.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = numpy.concatenate((v0, v1), axis=0)
        u, s, vh = numpy.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = numpy.dot(C, numpy.linalg.pinv(B))
        t = numpy.concatenate((t, numpy.zeros((ndims, 1))), axis=1)
        M = numpy.vstack((t, ((0.0,)*ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = numpy.linalg.svd(numpy.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = numpy.dot(u, vh)
        if numpy.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= numpy.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = numpy.identity(ndims+1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = numpy.sum(v0 * v1, axis=1)
        xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = numpy.linalg.eigh(N)
        q = V[:, numpy.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(numpy.sum(v1) / numpy.sum(v0))

    # move centroids back
    M = numpy.dot(numpy.linalg.inv(M1), numpy.dot(M, M0))
    M /= M[ndims, ndims]
    return M


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from matplotlib import cm
#     from mpl_toolkits.mplot3d import Axes3D

#     cal_pt_cloud = np.load('cal_pt_cloud.npy')
#     # plot input data
#     # Z = cal_pt_cloud
#     # ax.scatter(Z[:,0],Z[:,1],Z[:,2], c= "r")
#     # ax.scatter(Z[:,0],Z[:,1],Z[:,3], c= "b")

#     # fit once
#     model_n = 7
#     cx,cy,err_x,err_y = fit_poly_surface(cal_pt_cloud,model_n)
#     map_fn = make_map_function(cx,cy,model_n)
#     err_dist,err_mean,err_rms = fit_error_screen(err_x,err_y,(1280,720))
#     print err_rms,"in pixel"
#     threshold =15 # err_rms*2

#     # fit again disregarding crass outlines
#     cx,cy,new_err_x,new_err_y = fit_poly_surface(cal_pt_cloud[err_dist<=threshold],model_n)
#     map_fn = make_map_function(cx,cy,model_n)
#     new_err_dist,new_err_mean,new_err_rms = fit_error_screen(new_err_x,new_err_y,(1280,720))
#     print new_err_rms,"in pixel"

#     print "using %i datapoints out of the full dataset %i: subset is %i percent" \
#         %(cal_pt_cloud[err_dist<=threshold].shape[0], cal_pt_cloud.shape[0], \
#         100*float(cal_pt_cloud[err_dist<=threshold].shape[0])/cal_pt_cloud.shape[0])

#     # plot residuals
#     fig_error = plt.figure()
#     plt.scatter(err_x,err_y,c="y")
#     plt.scatter(new_err_x,new_err_y)
#     plt.title("fitting residuals full data set (y) and better subset (b)")


#     # plot projection of eye and world vs observed data
#     X,Y,ZX,ZY = cal_pt_cloud.transpose().copy()
#     X,Y = map_fn((X,Y))
#     X *= 1280/2.
#     Y *= 720/2.
#     ZX *= 1280/2.
#     ZY *= 720/2.
#     fig_projection = plt.figure()
#     plt.scatter(X,Y)
#     plt.scatter(ZX,ZY,c='y')
#     plt.title("world space projection in pixes, mapped and observed (y)")

#     # plot the fitting functions 3D plot
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     outliers =cal_pt_cloud[err_dist>threshold]
#     inliers = cal_pt_cloud[err_dist<=threshold]
#     ax.scatter(outliers[:,0],outliers[:,1],outliers[:,2], c= "y")
#     ax.scatter(outliers[:,0],outliers[:,1],outliers[:,3], c= "y")
#     ax.scatter(inliers[:,0],inliers[:,1],inliers[:,2], c= "r")
#     ax.scatter(inliers[:,0],inliers[:,1],inliers[:,3], c= "b")
#     Z = cal_pt_cloud
#     X = np.linspace(min(Z[:,0])-.2,max(Z[:,0])+.2,num=30,endpoint=True)
#     Y = np.linspace(min(Z[:,1])-.2,max(Z[:,1]+.2),num=30,endpoint=True)
#     X, Y = np.meshgrid(X,Y)
#     ZX,ZY = map_fn((X,Y))
#     ax.plot_surface(X, Y, ZX, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='r')
#     ax.plot_surface(X, Y, ZY, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='b')
#     plt.xlabel("Pupil x in Eye-Space")
#     plt.ylabel("Pupil y Eye-Space")
#     plt.title("Z: Gaze x (blue) Gaze y (red) World-Space, yellow=outliers")

#     # X,Y,_,_ = cal_pt_cloud.transpose()

#     # pts= map_fn((X,Y))
#     # import cv2
#     # pts = np.array(pts,dtype=np.float32).transpose()
#     # print cv2.convexHull(pts)[:,0]
#     plt.show()
