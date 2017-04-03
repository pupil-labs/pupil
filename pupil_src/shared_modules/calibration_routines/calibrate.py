'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2

from methods import undistort_unproject_pts
#logging
import logging
logger = logging.getLogger(__name__)


def calibrate_2d_polynomial(cal_pt_cloud,screen_size=(1,1),threshold = 35, binocular=False):
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

        logger.info('first iteration. root-mean-square residuals: {}, in pixel'.format(err_rms))
        logger.info('second iteration: ignoring outliers. root-mean-square residuals: {} in pixel'.format(new_err_rms))

        used_num = cal_pt_cloud[err_dist <= threshold].shape[0]
        complete_num = cal_pt_cloud.shape[0]
        logger.info('used {} data points out of the full dataset {}: subset is {:.2f} percent'.format(
            used_num, complete_num, 100*float(used_num)/complete_num))

        return map_fn, err_dist <= threshold, ([p.tolist() for p in cx], [p.tolist() for p in cy], model_n)

    else:  # did disregard all points. The data cannot be represented by the model in a meaningful way:
        map_fn = make_map_function(cx,cy,model_n)
        logger.error('First iteration. root-mean-square residuals: {} in pixel, this is bad!'.format(err_rms))
        logger.error('The data cannot be represented by the model in a meaningfull way.')
        return map_fn,err_dist<=threshold,([p.tolist() for p in cx], [p.tolist() for p in cy], model_n)



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

def fit_error_screen(err_x,err_y,screen_pos):
    screen_x,screen_y = screen_pos
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
        def fn(pt):
            X,Y = pt
            x2 = cx[0]*X + cx[1]*Y +cx[2]
            y2 = cy[0]*X + cy[1]*Y +cy[2]
            return x2,y2

    elif n==5:
        def fn(pt_0,pt_1):
            #        X0        Y0        X1        Y1        Ones
            X0,Y0 = pt_0
            X1,Y1 = pt_1
            x2 = cx[0]*X0 + cx[1]*Y0 + cx[2]*X1 + cx[3]*Y1 + cx[4]
            y2 = cy[0]*X0 + cy[1]*Y0 + cy[2]*X1 + cy[3]*Y1 + cy[4]
            return x2,y2

    elif n==7:
        def fn(pt):
            X,Y = pt
            x2 = cx[0]*X + cx[1]*Y + cx[2]*X*X + cx[3]*Y*Y + cx[4]*X*Y + cx[5]*Y*Y*X*X +cx[6]
            y2 = cy[0]*X + cy[1]*Y + cy[2]*X*X + cy[3]*Y*Y + cy[4]*X*Y + cy[5]*Y*Y*X*X +cy[6]
            return x2,y2

    elif n==9:
        def fn(pt):
            #          X         Y         XX         YY         XY         XXYY         XXY         YYX         Ones
            X,Y = pt
            x2 = cx[0]*X + cx[1]*Y + cx[2]*X*X + cx[3]*Y*Y + cx[4]*X*Y + cx[5]*Y*Y*X*X + cx[6]*Y*X*X + cx[7]*Y*Y*X + cx[8]
            y2 = cy[0]*X + cy[1]*Y + cy[2]*X*X + cy[3]*Y*Y + cy[4]*X*Y + cy[5]*Y*Y*X*X + cy[6]*Y*X*X + cy[7]*Y*Y*X + cy[8]
            return x2,y2

    elif n==13:
        def fn(pt_0,pt_1):
            #        X0        Y0        X1         Y1            XX0        YY0            XY0            XXYY0                XX1            YY1            XY1            XXYY1        Ones
            X0,Y0 = pt_0
            X1,Y1 = pt_1
            x2 = cx[0]*X0 + cx[1]*Y0 + cx[2]*X1 + cx[3]*Y1 + cx[4]*X0*X0 + cx[5]*Y0*Y0 + cx[6]*X0*Y0 + cx[7]*X0*X0*Y0*Y0 + cx[8]*X1*X1 + cx[9]*Y1*Y1 + cx[10]*X1*Y1 + cx[11]*X1*X1*Y1*Y1 + cx[12]
            y2 = cy[0]*X0 + cy[1]*Y0 + cy[2]*X1 + cy[3]*Y1 + cy[4]*X0*X0 + cy[5]*Y0*Y0 + cy[6]*X0*Y0 + cy[7]*X0*X0*Y0*Y0 + cy[8]*X1*X1 + cy[9]*Y1*Y1 + cy[10]*X1*Y1 + cy[11]*X1*X1*Y1*Y1 + cy[12]
            return x2,y2

    elif n==17:
        def fn(pt_0,pt_1):
            #        X0        Y0        X1         Y1            XX0        YY0            XY0            XXYY0                XX1            YY1            XY1            XXYY1            X0X1            X0Y1            Y0X1        Y0Y1           Ones
            X0,Y0 = pt_0
            X1,Y1 = pt_1
            x2 = cx[0]*X0 + cx[1]*Y0 + cx[2]*X1 + cx[3]*Y1 + cx[4]*X0*X0 + cx[5]*Y0*Y0 + cx[6]*X0*Y0 + cx[7]*X0*X0*Y0*Y0 + cx[8]*X1*X1 + cx[9]*Y1*Y1 + cx[10]*X1*Y1 + cx[11]*X1*X1*Y1*Y1 + cx[12]*X0*X1 + cx[13]*X0*Y1 + cx[14]*Y0*X1 + cx[15]*Y0*Y1 + cx[16]
            y2 = cy[0]*X0 + cy[1]*Y0 + cy[2]*X1 + cy[3]*Y1 + cy[4]*X0*X0 + cy[5]*Y0*Y0 + cy[6]*X0*Y0 + cy[7]*X0*X0*Y0*Y0 + cy[8]*X1*X1 + cy[9]*Y1*Y1 + cy[10]*X1*Y1 + cy[11]*X1*X1*Y1*Y1 + cy[12]*X0*X1 + cy[13]*X0*Y1 + cy[14]*Y0*X1 + cy[15]*Y0*Y1 + cy[16]
            return x2,y2

    else:
        raise Exception("ERROR: unsopported number of coefficiants.")

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
                matched.append({'ref':r,'pupil':closest_p0, 'pupil1':closest_p1})
            else:
                print("to far.")
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
        ref,p0,p1 = triplet['ref'],triplet['pupil'],triplet['pupil1']
        data_pt = p0["norm_pos"][0], p0["norm_pos"][1],p1["norm_pos"][0], p1["norm_pos"][1],ref['norm_pos'][0],ref['norm_pos'][1]
        cal_data.append( data_pt )
    return cal_data

def preprocess_3d_data(matched_data, camera_intrinsics):
    camera_matrix = np.asarray(camera_intrinsics["camera_matrix"])
    dist_coefs = np.asarray(camera_intrinsics["dist_coefs"])

    ref_processed = []
    pupil0_processed = []
    pupil1_processed = []

    is_binocular = len(matched_data[0] ) == 3
    for data_point in matched_data:
        try:
            # taking the pupil normal as line of sight vector
            pupil0 = data_point['pupil']
            gaze_vector0 = np.array(pupil0['circle_3d']['normal'])
            pupil0_processed.append( gaze_vector0 )

            if is_binocular: # we have binocular data
                pupil1 = data_point['pupil1']
                gaze_vector1 = np.array(pupil1['circle_3d']['normal'])
                pupil1_processed.append( gaze_vector1 )

            # projected point uv to normal ray vector of camera
            ref = data_point['ref']
            ref_vector =  undistort_unproject_pts(ref['screen_pos'] , camera_matrix, dist_coefs).tolist()[0]
            ref_vector = ref_vector / np.linalg.norm(ref_vector)
            # assuming a fixed (assumed) distance we get a 3d point in world camera 3d coords.
            ref_processed.append( np.array(ref_vector) )

        except KeyError as e:
            # this pupil data point did not have 3d detected data.
            pass

    return ref_processed,pupil0_processed,pupil1_processed


def find_rigid_transform(A, B):
    A = np.matrix(A)
    B = np.matrix(B)
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
       print("Reflection detected")
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    return np.array(R), np.array(t).reshape(3)

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
