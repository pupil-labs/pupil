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


cal_pt_cloud = np.load('/Volumes/BIN/Moritz/MIT/thesis/Software/Sandbox/cal_pt_cloud.npy')

Z = cal_pt_cloud 
#plot input data
#ax.scatter(Z[:,0],Z[:,1],Z[:,2], c= "r")
#ax.scatter(Z[:,0],Z[:,1],[z+0 for z in Z[:,3]], c= "b")

s =4

if s == 1:
    #create fn plane from coeffients
    Z = cal_pt_cloud 
    x,y,c = fitPlane(Z[:,0],Z[:,1],Z[:,2],)
    X = np.array([i/15.0 for i in range(-4,8)])
    Y = np.array([i/15.0 for i in range(-10,0)])
    X, Y = np.meshgrid(X,Y)
    Z =x*X + y*Y +c
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.1, antialiased=True,alpha=0.4,color='r')
    Z = cal_pt_cloud 
    ax.scatter(Z[:,0],Z[:,1],Z[:,2], c= "r")

if s == 2:
    #create fn plane from coeffients
    Z = cal_pt_cloud 
    x,y,c = fitPlane(Z[:,0],Z[:,1],Z[:,3],)
    X = np.array([i/15.0 for i in range(-4,8)])
    Y = np.array([i/15.0 for i in range(-10,0)])
    X, Y = np.meshgrid(X,Y)
    Z =x*X + y*Y +c
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='b')
    Z = cal_pt_cloud 
    ax.scatter(Z[:,0],Z[:,1],Z[:,3], c= "b")

if s==4:
    #create fn plane from coeffients
    Z = cal_pt_cloud 
    x,y,xx,yy,xy,xxyy,c = Fit_Polynomial_Surf(Z[:,0],Z[:,1],Z[:,2],)
    X = np.array([i/15.0 for i in range(-10,10)])
    Y = np.array([i/15.0 for i in range(-10,5)])
    X, Y = np.meshgrid(X,Y)
    Z =x*X + y*Y + xx*X*X + yy*Y*Y + xy*X*Y + xxyy*Y*Y*X*X +c
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='r')
    Z = cal_pt_cloud 
    ax.scatter(Z[:,0],Z[:,1],Z[:,2], c= "r")

if s ==4:
    #create fn plane from coeffients
    Z = cal_pt_cloud 
    x,y,xx,yy,xy,xxyy,c = Fit_Polynomial_Surf(Z[:,0],Z[:,1],Z[:,3],)
    X = np.array([i/15.0 for i in range(-10,10)])
    Y = np.array([i/15.0 for i in range(-10,5)])
    X, Y = np.meshgrid(X,Y)
    Z =x*X + y*Y + xx*X*X + yy*Y*Y + xy*X*Y + xxyy*Y*Y*X*X +c
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='b')
    Z = cal_pt_cloud 
    ax.scatter(Z[:,0],Z[:,1],Z[:,3], c= "b")

ax.set_zlim3d(-1.5, 1.5)
#plt.axis([-1,.5, -1,.5,])
plt.show()
