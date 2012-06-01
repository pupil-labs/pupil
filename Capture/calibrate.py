'''
Title: Pupil: Eye Tracking Software
calibrate.py
Authors: Moritz Kassner & William Patera
Date: July 12, 2011
Notes: Prototype tested with 2 XBox Live Webcams on Macbook Pro OSX 10.6.7 and Ubuntu 10.10 using OpenCV 2.2
'''

import numpy as np

def fit_polynomial_surf(X,Y,Z):
    """
    Takes three lists of points and 
    performs Singular Value Decomposition 
    to find a linerar least squares fit surface
    """

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

def calibrate_poly(points):
    """
    calibrate takes data in the form of [[x1,y1,x2,y2],[x1,y1,x2,y2],...]
    and finds the transformation that leads from x1,y1 to x2,y2 

    x2 =x*X + y*Y + xx*X*X + yy*Y*Y + xy*X*Y + xxyy*Y*Y*X*X +c
    y2 =x*X + y*Y + xx*X*X + yy*Y*Y + xy*X*Y + xxyy*Y*Y*X*X +c

    by fitting the functions to the calibration data

    """
    points = np.array(points)
    #save points to file
    np.save('data/cal_pt_cloud.npy', points)
    
    x_cofs =fit_polynomial_surf(points[:,0],points[:,1],points[:,2])
    y_cofs =fit_polynomial_surf(points[:,0],points[:,1],points[:,3])


    coefficients = x_cofs+y_cofs  #this is list cocatiniation not addition!

    return coefficients



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


def calibrate_linear(points):
    """
    calibrate takes data in the form of [[x1,y1,x2,y2],[x1,y1,x2,y2],...]
    and finds the linear transformation that leads from x1,y1 to x2,y2 

    x2 = ax*x1+bx*y1+cx
    y2 = ay*x1+by+y1+cy

    by fitting the functions to the calibration data

    """
    points = np.array(points)
    #save points to file
    np.save('cal_pt_cloud.npy', points)
    
    ax,bx,cx = fitPlane(points[:,0],points[:,1],points[:,2])
    ay,by,cy = fitPlane(points[:,0],points[:,1],points[:,3])

    coefficients = (ax,bx,cx,ay,by,cy)
    
    return coefficients

def map_vector(vector,coefficients):
    """
    map eye vector to world vector based on coefficents
    x2 = ax*x1+bx*y1+cx
    y2 = ay*x1+by+y1+cy
    """
    if coefficients is None:
        return vector
    
    if len(coefficients )== 14:
        return map_vector_poly(vector,coefficients)
    elif len(coefficients )== 6:
        return map_vector_linear(vector,coefficients)
    
    else: 
        return 0,0
    

def map_vector_linear(vector,coefficients):
    """
    map eye vector to world vector based on coefficents
    x2 = ax*x1+bx*y1+cx
    y2 = ay*x1+by+y1+cy
    """
    x= vector[0]
    y= vector[1]
    c = coefficients 

    x2 = c[0]*x + c[1]*y + c[2]
    y2 = c[3]*x + c[4]*y + c[5]

    return x2,y2
    
    
def map_vector_poly(vector,coefficients):
    """
    map eye vector to world vector based on coefficents
    x2 =x*X + y*Y + xx*X*X + yy*Y*Y + xy*X*Y + xxyy*Y*Y*X*X +c
    y2 =x*X + y*Y + xx*X*X + yy*Y*Y + xy*X*Y + xxyy*Y*Y*X*X +c
    """
    
    X= vector[0]
    Y= vector[1]
    c = coefficients 
    
    x2 =c[0]*X + c[1]*Y + c[2]*X*X + c[3]*Y*Y + c[4]*X*Y + c[5]*Y*Y*X*X +c[6]
    y2 =c[7]*X + c[8]*Y + c[9]*X*X + c[10]*Y*Y + c[11]*X*Y + c[12]*Y*Y*X*X +c[13]

    return x2,y2
