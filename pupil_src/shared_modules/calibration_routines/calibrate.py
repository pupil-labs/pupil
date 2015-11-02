'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
#logging
import logging
logger = logging.getLogger(__name__)


def get_map_from_cloud(cal_pt_cloud,screen_size=(2,2),threshold = 35,return_inlier_map=False,return_params=False, binocular=False):
    """
    we do a simple two pass fitting to a pair of bi-variate polynomials
    return the function to map vector
    """
    # fit once using all avaiable data
    model_n = 7
    if binocular:
        model_n = 13
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

        if return_inlier_map and return_params:
            return map_fn,err_dist<=threshold,(cx,cy,model_n)
        if return_inlier_map and not return_params:
            return map_fn,err_dist<=threshold
        if return_params and not return_inlier_map:
            return map_fn,(cx,cy,model_n)
        return map_fn
    else: # did disregard all points. The data cannot be represented by the model in a meaningful way:
        map_fn = make_map_function(cx,cy,model_n)
        logger.info('First iteration. root-mean-square residuals: %s in pixel, this is bad!'%err_rms)
        logger.warning('The data cannot be represented by the model in a meaningfull way.')

        if return_inlier_map and return_params:
            return map_fn,err_dist<=threshold,(cx,cy,model_n)
        if return_inlier_map and not return_params:
            return map_fn,err_dist<=threshold
        if return_params and not return_inlier_map:
            return map_fn,(cx,cy,model_n)
        return map_fn



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


def preprocess_data(pupil_pts,ref_pts,id_filter=(0,)):
    '''small utility function to deal with timestamped but uncorrelated data
    input must be lists that contain dicts with at least "timestamp" and "norm_pos" and "id:
    filter id must be (0,) or (1,) or (0,1).
    '''
    assert id_filter in ( (0,),(1,),(0,1) )

    if len(ref_pts)<=2:
        return []

    pupil_pts = [p for p in pupil_pts if p['id'] in id_filter]

    # if filter is set to handle binocular data, e.g. (0,1)
    if id_filter == (0,1):
        return preprocess_data_binocular(pupil_pts, ref_pts)
    else:
        return preprocess_data_monocular(pupil_pts,ref_pts)

def preprocess_data_monocular(pupil_pts,ref_pts):
    cal_data = []
    cur_ref_pt = ref_pts.pop(0)
    next_ref_pt = ref_pts.pop(0)
    while True:
        matched = []
        while pupil_pts:
            #select all points past the half-way point between current and next ref data sample
            if pupil_pts[0]['timestamp'] <=(cur_ref_pt['timestamp']+next_ref_pt['timestamp'])/2.:
                matched.append(pupil_pts.pop(0))
            else:
                for p_pt in matched:
                    #only use close points
                    if abs(p_pt['timestamp']-cur_ref_pt['timestamp']) <= 1/15.: #assuming 30fps + slack
                        data_pt = p_pt["norm_pos"][0], p_pt["norm_pos"][1],cur_ref_pt['norm_pos'][0],cur_ref_pt['norm_pos'][1]
                        cal_data.append(data_pt)
                break
        if ref_pts:
            cur_ref_pt = next_ref_pt
            next_ref_pt = ref_pts.pop(0)
        else:
            break
    return cal_data

def preprocess_data_binocular(pupil_pts, ref_pts):
    matches = []

    cur_ref_pt = ref_pts.pop(0)
    next_ref_pt = ref_pts.pop(0)
    while True:
        matched = [[], [], cur_ref_pt]
        while pupil_pts:
            #select all points past the half-way point between current and next ref data sample
            if pupil_pts[0]['timestamp'] <=(cur_ref_pt['timestamp']+next_ref_pt['timestamp'])/2.:
                if abs(pupil_pts[0]['timestamp']-cur_ref_pt['timestamp']) <= 1/15.: #assuming 30fps + slack
                    eye_id = pupil_pts[0]['id']
                    matched[eye_id].append(pupil_pts.pop(0))
                else:
                    pupil_pts.pop(0)
            else:
                matches.append(matched)
                break
        if ref_pts:
            cur_ref_pt = next_ref_pt
            next_ref_pt = ref_pts.pop(0)
        else:
            break

    cal_data = []
    for pupil_pts_0, pupil_pts_1, ref_pt in matches:
        # there must be at least one sample for each eye
        if len(pupil_pts_0) <= 0 or len(pupil_pts_1) <= 0:
            continue

        p0 = pupil_pts_0.pop(0)
        p1 = pupil_pts_1.pop(0)
        while True:
            data_pt = p0["norm_pos"][0], p0["norm_pos"][1],p1["norm_pos"][0], p1["norm_pos"][1],ref_pt['norm_pos'][0],ref_pt['norm_pos'][1]
            cal_data.append(data_pt)

            # keep sample with higher timestamp and increase the one with lower timestamp
            if p0['timestamp'] <= p1['timestamp'] and pupil_pts_0:
                p0 = pupil_pts_0.pop(0)
                continue
            elif p1['timestamp'] <= p0['timestamp'] and pupil_pts_1:
                p1 = pupil_pts_1.pop(0)
                continue
            elif pupil_pts_0 and not pupil_pts_1:
                p0 = pupil_pts_0.pop(0)
            elif pupil_pts_1 and not pupil_pts_0:
                p1 = pupil_pts_1.pop(0)
            else:
                break

    return cal_data


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
