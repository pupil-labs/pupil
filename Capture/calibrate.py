'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import numpy as np

def get_map_from_cloud(cal_pt_cloud,screen_size=(2,2),verbose=False):
    # fit once unsing all avaialbe data
    model_n = 7
    cx,cy,err_x,err_y = fit_poly_surface(cal_pt_cloud,model_n)
    map_fn = make_map_function(cx,cy,model_n)
    err_dist,err_mean,err_rms = fit_error_screen(err_x,err_y,screen_size)
    threshold = 55 #pixel in world space
    #fit again disregarding extreme outliers
    cx,cy,new_err_x,new_err_y = fit_poly_surface(cal_pt_cloud[err_dist<=threshold],model_n)
    map_fn = make_map_function(cx,cy,model_n)
    new_err_dist,new_err_mean,new_err_rms = fit_error_screen(new_err_x,new_err_y,screen_size)


    if verbose:
        print 'first iteration. root-mean-square residuals:', err_rms,"in pixel"
        print 'second iteration: ignoring outliers. root-mean-square residuals:', new_err_rms, "in pixel"

        print "used %i datapoints out of the full dataset %i: subset is %i percent" \
            %(cal_pt_cloud[err_dist<=threshold].shape[0], cal_pt_cloud.shape[0], \
            100*float(cal_pt_cloud[err_dist<=threshold].shape[0])/cal_pt_cloud.shape[0])
    return map_fn

def fit_poly_surface(cal_pt_cloud,n=7):
    M = make_model(cal_pt_cloud,n)
    U,w,Vt = np.linalg.svd(M[:,:n],full_matrices=0)
    V = Vt.transpose()
    Ut = U.transpose()
    pseudINV = np.dot(V, np.dot(np.diag(1/w), Ut))
    cx = np.dot(pseudINV, M[:,n])
    cy = np.dot(pseudINV, M[:,n+1])
    #compute model error in world screen units if screen_res specified
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
    else:
        raise Exception("ERROR: Model n needs to be 3, 7 or 9")
    return M


def make_map_function(cx,cy,n):
    if n==3:
        def fn((X,Y)):
            x2 = cx[0]*X + cx[1]*Y +cx[2]
            y2 = cy[0]*X + cy[1]*Y +cy[2]
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
    else:
        raise Exception("ERROR: Model n needs to be 3, 7 or 9")

    return fn

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    cal_pt_cloud = np.load('cal_pt_cloud.npy')
    # #plot input data
    # Z = cal_pt_cloud
    # ax.scatter(Z[:,0],Z[:,1],Z[:,2], c= "r")
    # ax.scatter(Z[:,0],Z[:,1],Z[:,3], c= "b")

    ##fit once
    model_n = 7
    cx,cy,err_x,err_y = fit_poly_surface(cal_pt_cloud,model_n)
    map_fn = make_map_function(cx,cy,model_n)
    err_dist,err_mean,err_rms = fit_error_screen(err_x,err_y,(1280,720))
    print err_rms,"in pixel"
    threshold =65 # err_rms*2

    #fit again disregarding krass outlines
    cx,cy,new_err_x,new_err_y = fit_poly_surface(cal_pt_cloud[err_dist<=threshold],model_n)
    map_fn = make_map_function(cx,cy,model_n)
    new_err_dist,new_err_mean,new_err_rms = fit_error_screen(new_err_x,new_err_y,(1280,720))
    print new_err_rms,"in pixel"

    print "using %i datapoints out of the full dataset %i: subset is %i percent" \
        %(cal_pt_cloud[err_dist<=threshold].shape[0], cal_pt_cloud.shape[0], \
        100*float(cal_pt_cloud[err_dist<=threshold].shape[0])/cal_pt_cloud.shape[0])

    ###plot residuals
    fig_error = plt.figure()
    plt.scatter(err_x,err_y,c="y")
    plt.scatter(new_err_x,new_err_y)
    plt.title("fitting residuals full data set (y) and better subset (b)")


    ###plot projection of eye and world vs observed data
    X,Y,ZX,ZY = cal_pt_cloud.transpose().copy()
    X,Y = map_fn((X,Y))
    X *= 1280/2.
    Y *= 720/2.
    ZX *= 1280/2.
    ZY *= 720/2.
    fig_projection = plt.figure()
    plt.scatter(X,Y)
    plt.scatter(ZX,ZY,c='y')
    plt.title("world space projection in pixes, mapped and observed (y)")

    ###plot the fitting functions 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    outliers =cal_pt_cloud[err_dist>threshold]
    inliers = cal_pt_cloud[err_dist<=threshold]
    ax.scatter(outliers[:,0],outliers[:,1],outliers[:,2], c= "y")
    ax.scatter(outliers[:,0],outliers[:,1],outliers[:,3], c= "y")
    ax.scatter(inliers[:,0],inliers[:,1],inliers[:,2], c= "r")
    ax.scatter(inliers[:,0],inliers[:,1],inliers[:,3], c= "b")
    Z = cal_pt_cloud
    X = np.linspace(min(Z[:,0])-.2,max(Z[:,0])+.2,num=30,endpoint=True)
    Y = np.linspace(min(Z[:,1])-.2,max(Z[:,1]+.2),num=30,endpoint=True)
    X, Y = np.meshgrid(X,Y)
    ZX,ZY = map_fn((X,Y))
    ax.plot_surface(X, Y, ZX, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='r')
    ax.plot_surface(X, Y, ZY, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='b')
    plt.xlabel("Pupil x in Eye-Space")
    plt.ylabel("Pupil y Eye-Space")
    plt.title("Z: Gaze x (blue) Gaze y (red) World-Space, yellow=outliers")

    plt.show()
