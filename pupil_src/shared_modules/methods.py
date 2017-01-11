'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os, sys, platform, getpass
from time import time
import numpy as np
try:
    import numexpr as ne
except:
    ne = None
import cv2
import logging
logger = logging.getLogger(__name__)


def timer(dt):
    '''
    a generator used to time window refreshs
    '''
    t = time()
    while True:
        nt = time()
        if nt-t > dt:
            t = nt
            yield True
        else:
            yield False


def delta_t():
    ''' return time between each call like so:

    tick = delta_t()
    def get_dt():
        return next(tick)
    print get_dt()
    sleep(1)
    print get_dt()
    '''
    ts = time()
    while True:
        t = time()
        dt, ts = t-ts, t
        yield dt


class Roi(object):
    """this is a simple 2D Region of Interest class
    it is applied on numpy arrays for convenient slicing
    like this:

    roi_array_slice = full_array[r.view]
    # do something with roi_array_slice

    this creates a view, no data copying done
    """
    def __init__(self, array_shape):
        self.array_shape = array_shape
        self.lX = 0
        self.lY = 0
        self.uX = array_shape[1]
        self.uY = array_shape[0]
        self.nX = 0
        self.nY = 0

    @property
    def view(self):
        return slice(self.lY,self.uY,),slice(self.lX,self.uX)

    @view.setter
    def view(self, value):
        raise Exception('The view field is read-only. Use the set methods instead')

    def add_vector(self,vector):
        """
        adds the roi offset to a len2 vector
        """
        x,y = vector
        return (self.lX+x,self.lY+y)

    def sub_vector(self,vector):
        """
        subs the roi offset to a len2 vector
        """
        x,y = vector
        return (x-self.lX,y-self.lY)

    def set(self,vals):
        if vals is not None and len(vals) is 5:
            self.lX,self.lY,self.uX,self.uY,self.array_shape = vals
        elif vals is not None and len(vals) is 4:
            self.lX,self.lY,self.uX,self.uY= vals

    def get(self):
        return self.lX,self.lY,self.uX,self.uY,self.array_shape



def undistort_unproject_pts(pts_uv, camera_matrix, dist_coefs):
    """
    This function converts a set of 2D image coordinates to vectors in pinhole camera space.
    Hereby the intrinsics of the camera are taken into account.
    UV is converted to normalized image space (think frustum with image plane at z=1) then undistored
    adding a z_coordinate of 1 yield vectors pointing from 0,0,0 to the undistored image pixel.
    @return: ndarray with shape=(n, 3)

    """
    pts_uv = np.array(pts_uv)
    num_pts = pts_uv.size / 2

    pts_uv.shape = (int(num_pts), 1, 2)
    pts_uv = cv2.undistortPoints(pts_uv, camera_matrix, dist_coefs)
    pts_3d = cv2.convertPointsToHomogeneous(np.float32(pts_uv))
    pts_3d.shape = (int(num_pts),3)
    return pts_3d


def project_distort_pts(pts_xyz,camera_matrix, dist_coefs,  rvec = np.array([0,0,0], dtype=np.float32), tvec = np.array([0,0,0], dtype=np.float32) ):

    # projectPoints is the inverse of function implemented above --> should map the intermediate result to the original input
    pts2d, _ = cv2.projectPoints(pts_xyz, rvec , tvec, camera_matrix, dist_coefs)
    return pts2d.reshape(-1,2)

def cart_to_spherical(vector):
    # convert to spherical coordinates
    # source: http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    x,y,z = vector
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos( y /  r ) # for elevation angle defined from Z-axis down
    psi = np.arctan2(z, x)
    return r, theta, psi

def spherical_to_cart( r, theta , phi ):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta) * np.sin(phi)
    return x,y,z

def bin_thresholding(image, image_lower=0, image_upper=256):
    binary_img = cv2.inRange(image, np.asarray(image_lower),
                np.asarray(image_upper))

    return binary_img

def make_eye_kernel(inner_size,outer_size):
    offset = (outer_size - inner_size)/2
    inner_count = inner_size**2
    outer_count = outer_size**2-inner_count
    val_inner = -1.0 / inner_count
    val_outer = -val_inner*inner_count/outer_count
    inner = np.ones((inner_size,inner_size),np.float32)*val_inner
    kernel = np.ones((outer_size,outer_size),np.float32)*val_outer
    kernel[offset:offset+inner_size,offset:offset+inner_size]= inner
    return kernel

def dif_gaus(image, lower, upper):
        lower, upper = int(lower-1), int(upper-1)
        lower = cv2.GaussianBlur(image,ksize=(lower,lower),sigmaX=0)
        upper = cv2.GaussianBlur(image,ksize=(upper,upper),sigmaX=0)
        # upper +=50
        # lower +=50
        dif = lower-upper
        # dif *= .1
        # dif = cv2.medianBlur(dif,3)
        # dif = 255-dif
        dif = cv2.inRange(dif, np.asarray(200),np.asarray(256))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        dif = cv2.dilate(dif, kernel, iterations=2)
        dif = cv2.erode(dif, kernel, iterations=1)
        # dif = cv2.max(image,dif)
        # dif = cv2.dilate(dif, kernel, iterations=1)
        return dif

def equalize(image, image_lower=0.0, image_upper=255.0):
    image_lower = int(image_lower*2)/2
    image_lower +=1
    image_lower = max(3,image_lower)
    mean = cv2.medianBlur(image,255)
    image = image - (mean-100)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # cv2.dilate(image, kernel, image, iterations=1)
    return image


def erase_specular(image,lower_threshold=0.0, upper_threshold=150.0):
    """erase_specular: removes specular reflections
            within given threshold using a binary mask (hi_mask)
    """
    thresh = cv2.inRange(image,
                np.asarray(float(lower_threshold)),
                np.asarray(256.0))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    hi_mask = cv2.dilate(thresh, kernel, iterations=2)

    specular = cv2.inpaint(image, hi_mask, 2, flags=cv2.INPAINT_TELEA)
    # return cv2.max(hi_mask,image)
    return specular


def find_hough_circles(img):
    circles = cv2.HoughCircles(pupil_img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=80)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)



def chessboard(image, pattern_size=(9,5)):
    status, corners = cv2.findChessboardCorners(image, pattern_size, flags=4)
    if status:
        mean = corners.sum(0)/corners.shape[0]
        # mean is [[x,y]]
        return mean[0], corners
    else:
        return None


def curvature(c):
    try:
        from vector import Vector
    except:
        return
    c = c[:,0]
    curvature = []
    for i in range(len(c)-2):
        #find the angle at i+1
        frm = Vector(c[i])
        at = Vector(c[i+1])
        to = Vector(c[i+2])
        a = frm -at
        b = to -at
        angle = a.angle(b)
        curvature.append(angle)
    return curvature



def GetAnglesPolyline(polyline,closed=False):
    """
    see: http://stackoverflow.com/questions/3486172/angle-between-3-points
    ported to numpy
    returns n-2 signed angles
    """

    points = polyline[:,0]

    if closed:
        a = np.roll(points,1,axis=0)
        b = points
        c = np.roll(points,-1,axis=0)
    else:
        a = points[0:-2] # all "a" points
        b = points[1:-1] # b
        c = points[2:]  # c points
    # ab =  b.x - a.x, b.y - a.y
    ab = b-a
    # cb =  b.x - c.x, b.y - c.y
    cb = b-c
    # float dot = (ab.x * cb.x + ab.y * cb.y); # dot product
    # print 'ab:',ab
    # print 'cb:',cb

    # float dot = (ab.x * cb.x + ab.y * cb.y) dot product
    # dot  = np.dot(ab,cb.T) # this is a full matrix mulitplication we only need the diagonal \
    # dot = dot.diagonal() #  because all we look for are the dotproducts of corresponding vectors (ab[n] and cb[n])
    dot = np.sum(ab * cb, axis=1) # or just do the dot product of the correspoing vectors in the first place!

    # float cross = (ab.x * cb.y - ab.y * cb.x) cross product
    cros = np.cross(ab,cb)

    # float alpha = atan2(cross, dot);
    alpha = np.arctan2(cros,dot)
    return alpha*(180./np.pi) #degrees
    # return alpha #radians

# if ne:
#     def GetAnglesPolyline(polyline):
#         """
#         see: http://stackoverflow.com/questions/3486172/angle-between-3-points
#         ported to numpy
#         returns n-2 signed angles
#         same as above but implemented using numexpr
#         SLOWER than just numpy!
#         """

#         points = polyline[:,0]
#         a = points[0:-2] # all "a" points
#         b = points[1:-1] # b
#         c = points[2:]  # c points
#         ax,ay = a[:,0],a[:,1]
#         bx,by = b[:,0],b[:,1]
#         cx,cy = c[:,0],c[:,1]
#         # abx =  '(bx - ax)'
#         # aby =  '(by - ay)'
#         # cbx =  '(bx - cx)'
#         # cby =  '(by - cy)'
#         # # float dot = (ab.x * cb.x + ab.y * cb.y) dot product
#         # dot = '%s * %s + %s * %s' %(abx,cbx,aby,cby)
#         # # float cross = (ab.x * cb.y - ab.y * cb.x) cross product
#         # cross = '(%s * %s - %s * %s)' %(abx,cby,aby,cbx)
#         # # float alpha = atan2(cross, dot);
#         # alpha = "arctan2(%s,%s)" %(cross,dot)
#         # term = '%s*%s'%(alpha,180./np.pi)
#         term = 'arctan2(((bx - ax) * (by - cy) - (by - ay) * (bx - cx)),(bx - ax) * (bx - cx) + (by - ay) * (by - cy))*57.2957795131'
#         return ne.evaluate(term)



def split_at_angle(contour, curvature, angle):
    """
    contour is array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )
    curvature is a n-2 list
    """
    segments = []
    kink_index = [i for i in range(len(curvature)) if curvature[i] < angle]
    for s,e in zip([0]+kink_index,kink_index+[None]): # list of slice indecies 0,i0,i1,i2,None
        if e is not None:
            segments.append(contour[s:e+1]) #need to include the last index
        else:
            segments.append(contour[s:e])
    return segments


def find_kink(curvature, angle):
    """
    contour is array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )
    curvature is a n-2 list
    """
    kinks = []
    kink_index = [i for i in range(len(curvature)) if abs(curvature[i]) < angle]
    return kink_index

def find_change_in_general_direction(curvature):
    """
    return indecies of where the singn of curvature has flipped
    """
    curv_pos = curvature > 0
    split = []
    currently_pos = curv_pos[0]
    for c, is_pos in zip(range(curvature.shape[0]),curv_pos):
        if is_pos !=currently_pos:
            currently_pos = is_pos
            split.append(c)
    return split


def find_kink_and_dir_change(curvature,angle):
    split = []
    if curvature.shape[0] == 0:
        return split
    curv_positive = curvature > 0
    currently_positive = curv_positive[0]
    for idx,c, is_posisitve in zip(range(curvature.shape[0]),curvature,curv_positive):
        if (is_posisitve !=currently_positive) or abs(c) < angle:
            currently_positive = is_posisitve
            split.append(idx)
    return split


def find_slope_disc(curvature,angle = 15):
    # this only makes sense when your polyline is longish
    if len(curvature)<4:
        return []

    i = 2
    split_idx = []
    for anchor1,anchor2,candidate in zip(curvature,curvature[1:],curvature[2:]):
        base_slope = anchor2-anchor1
        new_slope = anchor2 - candidate
        dif = abs(base_slope-new_slope)
        if dif>=angle:
            split_idx.add(i)
        i +=1

    return split_list

def find_slope_disc_test(curvature,angle = 15):
    # this only makes sense when your polyline is longish
    if len(curvature)<4:
        return []
    # mean = np.mean(curvature)
    # print '------------------- start'
    i = 2
    split_idx = set()
    for anchor1,anchor2,candidate in zip(curvature,curvature[1:],curvature[2:]):
        base_slope = anchor2-anchor1
        new_slope = anchor2 - candidate
        dif = abs(base_slope-new_slope)
        if dif>=angle:
            split_idx.add(i)
        # print i,dif
        i +=1
    i-= 3
    for anchor1,anchor2,candidate in zip(curvature[::-1],curvature[:-1:][::-1],curvature[:-2:][::-1]):
        avg = (anchor1+anchor2)/2.
        dif = abs(avg-candidate)
        if dif>=angle:
            split_idx.add(i)
        # print i,dif
        i -=1
    split_list = list(split_idx)
    split_list.sort()
    # print split_list
    # print '-------end'
    return split_list


def points_at_corner_index(contour,index):
    """
    contour is array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )
    #index n-2 because the curvature is n-2 (1st and last are not exsistent), this shifts the index (0 splits at first knot!)
    """
    return [contour[i+1] for i in index]


def split_at_corner_index(contour,index):
    """
    contour is array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )
    #index n-2 because the curvature is n-2 (1st and last are not exsistent), this shifts the index (0 splits at first knot!)
    """
    segments = []
    index = [i+1 for i in index]
    for s,e in zip([0]+index,index+[10000000]): # list of slice indecies 0,i0,i1,i2,
        segments.append(contour[s:e+1])# +1 is for not loosing line segments
    return segments


def convexity_defect(contour, curvature):
    """
    contour is array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )
    curvature is a n-2 list
    """
    kinks = []
    mean = np.mean(curvature)
    if mean>0:
        kink_index = [i for i in range(len(curvature)) if curvature[i] < 0]
    else:
        kink_index = [i for i in range(len(curvature)) if curvature[i] > 0]
    for s in kink_index: # list of slice indecies 0,i0,i1,i2,None
        kinks.append(contour[s+1]) # because the curvature is n-2 (1st and last are not exsistent)
    return kinks,kink_index


def is_round(ellipse,ratio,tolerance=.8):
    center, (axis1,axis2), angle = ellipse

    if axis1 and axis2 and abs( ratio - min(axis2,axis1)/max(axis2,axis1)) <  tolerance:
        return True
    else:
        return False

def size_deviation(ellipse,target_size):
    center, axis, angle = ellipse
    return abs(target_size-max(axis))





def circle_grid(image, pattern_size=(4,11)):
    """Circle grid: finds an assymetric circle pattern
    - circle_id: sorted from bottom left to top right (column first)
    - If no circle_id is given, then the mean of circle positions is returned approx. center
    - If no pattern is detected, function returns None
    """
    status, centers = cv2.findCirclesGridDefault(image, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    if status:
        return centers
    else:
        return None

def calibrate_camera(img_pts, obj_pts, img_size):
    # generate pattern size
    camera_matrix = np.zeros((3,3))
    dist_coef = np.zeros(4)
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts,
                                                    img_size, camera_matrix, dist_coef)
    return camera_matrix, dist_coefs

def gen_pattern_grid(size=(4,11)):
    pattern_grid = []
    for i in range(size[1]):
        for j in range(size[0]):
            pattern_grid.append([(2*j)+i%2,i,0])
    return np.asarray(pattern_grid, dtype='f4')



def normalize(pos, size, flip_y=False):
    """
    normalize return as float
    """
    width,height = size
    x = pos[0]
    y = pos[1]
    x /=float(width)
    y /=float(height)
    if flip_y:
        return x,1-y
    return x,y

def denormalize(pos, size, flip_y=False):
    """
    denormalize
    """
    width,height = size
    x = pos[0]
    y = pos[1]
    x *= width
    if flip_y:
        y = 1-y
    y *= height
    return x,y



def dist_pts_ellipse(ellipse, points):
    """
    return unsigned euclidian distances of points to ellipse
    """
    pos, size, angle = ellipse
    ex,ey = pos
    dx,dy = size
    pts = np.float64(points)
    rx,ry = dx/2., dy/2.
    angle = (angle/180.)*np.pi
    # ex,ey =ex+0.000000001,ey-0.000000001 #hack to make 0 divisions possible this is UGLY!!!
    pts = pts - np.array((ex,ey)) # move pts to ellipse appears at origin , with this we copy data -deliberatly!

    M_rot = np.mat([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    pts = np.array(pts*M_rot) #rotate so that ellipse axis align with coordinate system
    # print "rotated",pts

    pts /= np.array((rx,ry)) #normalize such that ellipse radii=1
    # print "normalize",norm_pts
    norm_mag = np.sqrt((pts*pts).sum(axis=1))
    norm_dist = abs(norm_mag-1) #distance of pt to ellipse in scaled space
    # print 'norm_mag',norm_mag
    # print 'norm_dist',norm_dist
    ratio = (norm_dist)/norm_mag #scale factor to make the pts represent their dist to ellipse
    # print 'ratio',ratio
    scaled_error = np.transpose(pts.T*ratio) # per vector scalar multiplication: makeing sure that boradcasting is done right
    # print "scaled error points", scaled_error
    real_error = scaled_error*np.array((rx,ry))
    # print "real point",real_error
    error_mag = np.sqrt((real_error*real_error).sum(axis=1))
    # print 'real_error',error_mag
    # print 'result:',error_mag
    return error_mag


if ne:
    def dist_pts_ellipse(ellipse, points):
        """
        return unsigned euclidian distances of points to ellipse
        same as above but uses numexpr for 2x speedup
        """
        pos, size, angle = ellipse
        ex,ey = pos
        dx,dy = size
        pts = np.float64(points)
        pts.shape=(-1,2)
        rx,ry = dx/2., dy/2.
        angle = (angle/180.)*np.pi
        # ex,ey = ex+0.000000001 , ey-0.000000001 #hack to make 0 divisions possible this is UGLY!!!
        x = pts[:,0]
        y = pts[:,1]
        # px = '((x-ex) * cos(angle) + (y-ey) * sin(angle))/rx'
        # py = '(-(x-ex) * sin(angle) + (y-ey) * cos(angle))/ry'
        # norm_mag = 'sqrt(('+px+')**2+('+py+')**2)'
        # norm_dist = 'abs('+norm_mag+'-1)'
        # ratio = norm_dist + "/" + norm_mag
        # x_err  = ''+px+'*'+ratio+'*rx'
        # y_err =  ''+py+'*'+ratio+'*ry'
        # term = 'sqrt(('+x_err+')**2 + ('+y_err+')**2 )'
        term = 'sqrt((((x-ex) * cos(angle) + (y-ey) * sin(angle))/rx*abs(sqrt((((x-ex) * cos(angle) + (y-ey) * sin(angle))/rx)**2+((-(x-ex) * sin(angle) + (y-ey) * cos(angle))/ry)**2)-1)/sqrt((((x-ex) * cos(angle) + (y-ey) * sin(angle))/rx)**2+((-(x-ex) * sin(angle) + (y-ey) * cos(angle))/ry)**2)*rx)**2 + ((-(x-ex) * sin(angle) + (y-ey) * cos(angle))/ry*abs(sqrt((((x-ex) * cos(angle) + (y-ey) * sin(angle))/rx)**2+((-(x-ex) * sin(angle) + (y-ey) * cos(angle))/ry)**2)-1)/sqrt((((x-ex) * cos(angle) + (y-ey) * sin(angle))/rx)**2+((-(x-ex) * sin(angle) + (y-ey) * cos(angle))/ry)**2)*ry)**2 )'
        error_mag = ne.evaluate(term)
        return error_mag



def metric(l):
    """
    example metric for search
    """
    # print 'evaluating', idecies
    global evals
    evals +=1
    return sum(l) < 3




def pruning_quick_combine(l,fn,seed_idx=None,max_evals=1e20,max_depth=5):
    """
    l is a list of object to quick_combine.
    the evaluation fn should accept idecies to your list and the list
    it should return a binary result on wether this set is good

    this search finds all combinations but assumes:
        that a bad subset can not be bettered by adding more nodes
        that a good set may not always be improved by a 'passing' superset (purging subsets will revoke this)

    if all items and their combinations pass the evaluation fn you get n**2 -1 solutions
    which leads to (2**n - 1) calls of your evaluation fn

    it needs more evaluations than finding strongly connected components in a graph because:
    (1,5) and (1,6) and (5,6) may work but (1,5,6) may not pass evaluation, (n,m) being list idx's

    """
    if seed_idx:
        non_seed_idx = [i for i in range(len(l)) if i not in seed_idx]
    else:
        #start from every item
        seed_idx = range(len(l)) #never happen, because we have an early exit if we have no seeds! patrick
        non_seed_idx = []
    mapping =  seed_idx+non_seed_idx
    unknown = [[node] for node in range(len(seed_idx))]
    results = []
    prune = []
    eval_count = 0
    while unknown and max_evals:
        path = unknown.pop(0)
        max_evals -= 1
        eval_count +=1
        # print '@idx',[mapping[i] for i in path]
        # print '@content',path
        if not len(path) > max_depth:
            # is this combination even viable, or did a subset fail already?
            if not any(m.issubset(set(path)) for m in prune):
                #we have not tested this and a subset of this was sucessfull before
                if fn([l[mapping[i]] for i in path]):
                    # yes this was good, keep as solution
                    results.append([mapping[i] for i in path])
                    # lets explore more by creating paths to each remaining node
                    decedents = [path+[i] for i in range(path[-1]+1,len(mapping)) ]
                    unknown.extend(decedents)
                else:
                    # print "pruning",path
                    prune.append(set(path))
    return results




# def is_subset(needle,haystack):
#     """ Check if needle is ordered subset of haystack in O(n)
#     taken from:
#     http://stackoverflow.com/questions/1318935/python-list-filtering-remove-subsets-from-list-of-lists
#     """

#     if len(haystack) < len(needle): return False

#     index = 0
#     for element in needle:
#         try:
#            index = haystack.index(element, index) + 1
#         except ValueError:
#             return False
#     else:
#        return True

# def filter_subsets(lists):
#     """ Given list of lists, return new list of lists without subsets
#     taken from:
#     http://stackoverflow.com/questions/1318935/python-list-filtering-remove-subsets-from-list-of-lists
#     """

#     for needle in lists:
#         if not any(is_subset(needle, haystack) for haystack in lists
#             if needle is not haystack):
#             yield needle

def filter_subsets(l):
    return [m for i, m in enumerate(l) if not any(set(m).issubset(set(n)) for n in (l[:i] + l[i+1:]))]



def get_system_info():
    try:
        if platform.system() == "Windows":
            username = os.environ["USERNAME"]
            sysname, nodename, release, version, machine, _ = platform.uname()
        else:
            username = getpass.getuser()
            sysname, nodename, release, version, machine = os.uname()
    except Exception as e:
        logger.error(e)
        username = 'unknown'
        sysname, nodename, release, version, machine = sys.platform,'unknown','unknown','unknown','unknown'

    return "User: {}, Platform: {}, Machine: {}, Release: {}, Version: {}".format(username,sysname,nodename,release,version)


if __name__ == '__main__':
    # tst = []
    # for x in range(10):
    #   tst.append(gen_pattern_grid())
    # tst = np.asarray(tst)
    # print tst.shape


    #test polyline
    #    *-*   *
    #    |  \  |
    #    *   *-*
    #    |
    #  *-*
    pl = np.array([[[0, 0]],[[0, 1]],[[1, 1]],[[2, 1]],[[2, 2]],[[1, 3]],[[1, 4]],[[2,4]]], dtype=np.int32)
    curvature = GetAnglesPolyline(pl,closed=0)
    curvature = GetAnglesPolyline(pl,closed=1)
    # print curvature
    # print find_curv_disc(curvature)
    # idx =  find_kink_and_dir_change(curvature,60)
    # print idx
    # print split_at_corner_index(pl,idx)
    # ellipse = ((0,0),(np.sqrt(2),np.sqrt(2)),0)
    # pts = np.array([(0,1),(.5,.5),(0,-1)])
    # # print pts.dtype
    # print dist_pts_ellipse(ellipse,pts)
    # print pts
    # # print test()

    # l = [1,2,1,0,1,0]
    # print len(l)
    # # evals = 0
    # # r = quick_combine(l,metric)
    # # # print r
    # # print filter_subsets(r)
    # # print evals

    # evals = 0
    # r = pruning_quick_combine(l,metric,[2])
    # print r
    # print filter_subsets(r)
    # print evals






