import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np


def Fit_Polynomial_Surf(X,Y,Z):
    """
    Takes three lists of points and
    performs Singular Value Decomposition
    to find a linerar least squares fit surface
    """
    Z_mean = sum(Z)/float(Z.shape[0])

    One = np.ones(Z.shape)
    Zero = np.zeros(Z.shape)
    XX = X*X
    YY = Y*Y
    XY = X*Y
    XXYY = X*Y*X*Y
    V = np.vstack((One,X,Y,XX,YY,XY,XXYY))
    V = V.transpose()
    U,w,Vt = np.linalg.svd(V,full_matrices=0);
    V = Vt.transpose();
    Ut = U.transpose();
    pseudINV = np.dot(V, np.dot(np.diag(1/w), Ut));
    coefs = np.dot(pseudINV, Z);
    c, x,y,xx,yy,xy,xxyy = coefs
    """
    print "coeffs"
    print "x",x
    print "y",y
    print "xy",xy
    print "xx",xx
    print "yy",yy
    print "xxyy",xxyy
    print "c",c
    """
    return x,y,xx,yy,xy,xxyy,c


def fitPlane(X,Y,Z):
    """
    Takes three lists of points and
    performs Singular Value Decomposition
    to find a linerar least squares fit plane
    """
    #make a 2d array with row-wise vectors
    X,Y,Z = np.array(X), np.array(Y), np.array(Z),
    Data = np.hstack((X[:,np.newaxis],Y[:,np.newaxis],Z[:,np.newaxis]))

    # set centroid to origin
    X_mean = sum(Data[:,0])/Data.shape[0]
    Y_mean = sum(Data[:,1])/Data.shape[0]
    Z_mean = sum(Data[:,2])/Data.shape[0]
    Data = np.array([[x-X_mean ,y-Y_mean ,Data-Z_mean]for x,y,Data in zip(Data[:,0],Data[:,1],Data[:,2]) ])

    #SVD
    u,s,vh = np.linalg.svd(Data)
    v = vh.conj().transpose()
    #nomalize to set first coeffiant to be 1
    vnorm = v/v[0,-1]
    # coefficiant: Data = a*X + b*Y +c
    a = -1/vnorm[2,-1]
    b = -vnorm[1,-1]/vnorm[2,-1]
    c = -X_mean*a-Y_mean*b +Z_mean
    return a,b,c



fig = plt.figure()
ax = fig.gca(projection='3d')


cal_pt_cloud = np.load('cal_pt_cloud.npy')
cal_pt_cloud = np.load('cal_pt_cloud_529.npy')
# cal_pt_cloud = np.load('cal_pt_cloud_good.npy')

Z = cal_pt_cloud
#plot input data
#ax.scatter(Z[:,0],Z[:,1],Z[:,2], c= "r")
#ax.scatter(Z[:,0],Z[:,1],[z+0 for z in Z[:,3]], c= "b")

s =2

if s == 1:
    #create fn plane from coeffients
    Z = cal_pt_cloud
    x,y,c = fitPlane(Z[:,0],Z[:,1],Z[:,2],)
    X = np.linspace(min(Z[:,0]),max(Z[:,0]),num=30,endpoint=True)
    Y = np.linspace(min(Z[:,1]),max(Z[:,1]),num=30,endpoint=True)
    X, Y = np.meshgrid(X,Y)
    Z =x*X + y*Y +c
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.1, antialiased=True,alpha=0.4,color='r')

    #calculate residuals
    Z = cal_pt_cloud
    X = Z[:,0]
    Y = Z[:,1]
    Zobserved = Z[:,2]
    Zmodel = x*X + y*Y +c
    Distance = Zobserved-Zmodel
    X_Distance = Distance*320
    Distance = np.abs(Distance)
    #average Residual
    X_Error = np.sum(Distance)/Distance.shape[0]
    # convert from normalized to screen units
    X_Error*=320
    print 'Average Residual in X in Pixels of World Camera',X_Error

    Z = cal_pt_cloud
    ax.scatter(Z[:,0],Z[:,1],Z[:,2], c= "r")

if s == 1:
    #create fn plane from coeffients
    Z = cal_pt_cloud
    x,y,c = fitPlane(Z[:,0],Z[:,1],Z[:,3],)
    X = np.linspace(min(Z[:,0]),max(Z[:,0]),num=30,endpoint=True)
    Y = np.linspace(min(Z[:,1]),max(Z[:,1]),num=30,endpoint=True)
    X, Y = np.meshgrid(X,Y)
    Z =x*X + y*Y +c
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='b')

    #calculate residuals
    Z = cal_pt_cloud
    X = Z[:,0]
    Y = Z[:,1]
    Zobserved = Z[:,3]
    Zmodel = x*X + y*Y +c
    Distance = Zobserved-Zmodel
    Y_Distance = Distance*240

    Distance = np.abs(Distance)
    #average Residual
    Y_Error = np.sum(Distance)/Distance.shape[0]
    # convert from normalized to screen units
    Y_Error*=240
    print 'Average Residual in X in Pixels of World Camera',Y_Error


    Z = cal_pt_cloud
    ax.scatter(Z[:,0],Z[:,1],Z[:,3], c= "b")


if s==2:
    #create fn plane from coeffients
    Z = cal_pt_cloud
    x,y,xx,yy,xy,xxyy,c = Fit_Polynomial_Surf(Z[:,0],Z[:,1],Z[:,2])
    X = np.linspace(min(Z[:,0]),max(Z[:,0]),num=30,endpoint=True)
    Y = np.linspace(min(Z[:,1]),max(Z[:,1]),num=30,endpoint=True)
    X, Y = np.meshgrid(X,Y)
    Z =x*X + y*Y + xx*X*X + yy*Y*Y + xy*X*Y + xxyy*Y*Y*X*X +c
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='r')

    # print "X Coeffs: x,y,xx,yy,xy,xxyy,c",x,y,xx,yy,xy,xxyy,c

    #calculate residuals
    Z = cal_pt_cloud
    X = Z[:,0]
    Y = Z[:,1]
    Zobserved = Z[:,2]
    Zmodel = x*X + y*Y + xx*X*X + yy*Y*Y + xy*X*Y + xxyy*Y*Y*X*X +c
    Distance = Zobserved-Zmodel
    # Distance *=Distance
    X_Distance = Distance*320
    Distance = np.abs(Distance)
    #average Residual
    X_Error = np.sum(Distance)/Distance.shape[0]
    # convert from normalized to screen units
    X_Error*=320
    print 'Average Residual in X in Pixels of World Camera',X_Error


    Z = cal_pt_cloud
    thresh = 9/320.
    X_Outliers = Distance>=thresh
    X_Inliers = np.logical_not(X_Outliers)
    ax.scatter(Z[Distance<thresh,0],Z[Distance<thresh,1],Z[Distance<thresh,2], c="r",)
    ax.scatter(Z[Distance>=thresh,0],Z[Distance>=thresh,1],Z[Distance>=thresh,2], c="y")

if s ==2:
    #create fn plane from coeffients
    Z = cal_pt_cloud
    x,y,xx,yy,xy,xxyy,c = Fit_Polynomial_Surf(Z[:,0],Z[:,1],Z[:,3])
    X = np.linspace(min(Z[:,0]),max(Z[:,0]),num=30,endpoint=True)
    Y = np.linspace(min(Z[:,1]),max(Z[:,1]),num=30,endpoint=True)
    X, Y = np.meshgrid(X,Y)
    Z =x*X + y*Y + xx*X*X + yy*Y*Y + xy*X*Y + xxyy*Y*Y*X*X +c
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='b')

    # print "Y Coeffs: x,y,xx,yy,xy,xxyy,c",x,y,xx,yy,xy,xxyy,c

    #calculate residuals
    Z = cal_pt_cloud
    X = Z[:,0]
    Y = Z[:,1]
    Zobserved = Z[:,3]
    Zmodel = x*X + y*Y + xx*X*X + yy*Y*Y + xy*X*Y + xxyy*Y*Y*X*X +c
    Distance = Zobserved-Zmodel
    Y_Distance = Distance*240
    Distance = np.abs(Distance)
    Y_Error = np.sum(Distance)/Distance.shape[0]
    Y_Error*=240
    print 'Average Residual in Y in Pixels of World Camera',Y_Error
    Z = cal_pt_cloud
    thresh = 9/240.
    Y_Outliers = Distance>=thresh
    Y_Inliers = np.logical_not(Y_Outliers)
    ax.scatter(Z[Distance<thresh,0],Z[Distance<thresh,1],Z[Distance<thresh,3], c="b",)
    ax.scatter(Z[Distance>=thresh,0],Z[Distance>=thresh,1],Z[Distance>=thresh,3], c="y")


plt.xlabel("Pupil x in Eye-Space")
plt.ylabel("Pupil y Eye-Space")
plt.title("Z: pattern x(red) y(blue),planes are map Fn's. Samples: %i" %Z.shape[0])

fig_error = plt.figure()


plt.scatter(X_Distance,Y_Distance)
plt.title("Residuals")
plt.xlabel("Pupil X Residuals in World-Space, avg. Residual %f pixels" %X_Error)
plt.ylabel("Pupil Y Residuals World-Space, avg. Residual %f pixels" %Y_Error)
plt.show()
