'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2

class Temp(object):
    """Temp class to make objects"""
    def __init__(self):
        pass

class Roi(object):
    """this is a simple 2D Region of Interest class
    it is applied on numpy arrays for convenient slicing
    like this:

    roi_array_slice = full_array[r.lY:r.uY,r.lX:r.uX]
    # do something with roi_array_slice
    full_array[r.lY:r.uY,r.lX:r.uX] = roi_array_slice

    this creates a view, no data copying done
    """
    def __init__(self, array_shape):
        self.array_shape = array_shape
        self.lX = 0
        self.lY = 0
        self.uX = array_shape[1]-0
        self.uY = array_shape[0]-0
        self.nX = 0
        self.nY = 0

    def setStart(self,(x,y)):
        x,y = int(x),int(y)
        x,y = max(0,x),max(0,y)
        self.nX,self.nY = x,y

    def setEnd(self,(x,y)):
            x,y = int(x),int(y)
            x,y = max(0,x),max(0,y)
            # make sure the ROI actually contains enough pixels
            if abs(self.nX - x) > 25 and abs(self.nY - y)>25:
                self.lX = min(x,self.nX)
                self.lY = min(y,self.nY)
                self.uX = max(x,self.nX)
                self.uY = max(y,self.nY)

    def add_vector(self,(x,y)):
        """
        adds the roi offset to a len2 vector
        """
        return (self.lX+x,self.lY+y)

    def set(self,vals):
        if vals is not None and len(vals) is 4:
            self.lX,self.lY,self.uX,self.uY = vals

    def get(self):
        return self.lX,self.lY,self.uX,self.uY

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


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
    circles = cv2.HoughCircles(pupil_img,cv2.cv.CV_HOUGH_GRADIENT,1,20,
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
    for i in xrange(len(c)-2):
        #find the angle at i+1
        frm = Vector(c[i])
        at = Vector(c[i+1])
        to = Vector(c[i+2])
        a = frm -at
        b = to -at
        angle = a.angle(b)
        curvature.append(angle)
    return curvature


def GetAnglesPolyline(polyline):
    """
    see: http://stackoverflow.com/questions/3486172/angle-between-3-points
    ported to numpy
    returns n-2 signed angles
    """

    points = polyline[:,0]
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
    return alpha * 180. / np.pi #degrees
    # return alpha #radians


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
    curv_pos = curvature > 0
    currently_pos = curv_pos[0]
    for idx,c, is_pos in zip(range(curvature.shape[0]),curvature,curv_pos):
        if (is_pos !=currently_pos) or abs(c) < angle:
            currently_pos = is_pos
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
        print i,dif
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
    for i in xrange(size[1]):
        for j in xrange(size[0]):
            pattern_grid.append([(2*j)+i%2,i,0])
    return np.asarray(pattern_grid, dtype='f4')



def normalize(pos, (width, height),flip_y=False):
    """
    normalize return as float
    """
    x = pos[0]
    y = pos[1]
    x = (x-width/2.)/(width/2.)
    y = (y-height/2.)/(height/2.)
    if flip_y:
        return x,-y
    return x,y

def denormalize(pos, (width, height), flip_y=False):
    """
    denormalize
    """
    x = pos[0]
    y = pos[1]
    x = (x*width/2.)+(width/2.)
    if flip_y:
        y = -y
    y = (y*height/2.)+(height/2.)
    return x,y



def dist_pts_ellipse(((ex,ey),(dx,dy),angle),pts):
    """
    return unsigned euclidian distances of points to ellipse
    """
    pts = np.float64(pts)
    rx,ry = dx/2., dy/2.
    angle = (angle/180.)*np.pi
    ex,ey =ex+0.000000001,ey-0.000000001 #hack to make 0 divisions possible this is UGLY!!!
    pts -= np.array((ex,ey)) # move pts to ellipse appears at origin
    M_rot = np.mat([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    pts = np.array(pts*M_rot) #rotate so that ellipse axis align with coordinate system
    # print "rotated",pts
    norm_pts = pts/np.array((rx,ry)) #normalize such that ellipse radii=1
    # print "normalize",norm_pts
    norm_mag = np.sqrt((norm_pts*norm_pts).sum(axis=1))
    norm_dist = abs(norm_mag-1) #distance of pt to ellipse in scaled space
    # print 'norm_mag',norm_mag
    # print 'norm_dist',norm_dist
    ratio = (norm_dist)/norm_mag #scale factor to make the pts represent their dist to ellipse
    # print 'ratio',ratio
    scaled_error = np.transpose(norm_pts.T*ratio) # per vector scalar multiplication
    # print "scaled error points", scaled_error
    real_error = scaled_error*np.array((rx,ry))
    # print "real point",real_error
    error_mag = np.sqrt((real_error*real_error).sum(axis=1))
    # print 'real_error',error_mag
    # print 'result:',error_mag
    return error_mag




def metric(idecies,l):
    """
    example metric for search
    """
    # print 'evaluating', idecies
    global evals
    evals +=1
    return sum([l[i] for i in idecies]) < 3


def cached_metric(indecies,l,prune):
    """
    example metric for search
    """
    if any(m.issubset(set(indecies)) for m in prune):
        print "pruning", indecies
        return False
    else:
        global evals
        evals +=1
        print 'evaluating', indecies

        res = sum([l[i] for i in indecies]) < 3
        if not res:
            prune.append(set(indecies))
        return res


def quick_combine(l,fn):
    """
    this search finds all combinations but assumes:
        that a bad subset can not be bettered by adding more nodes
        that a good set may not always be improved by a 'passing' superset (purging subsets will revoke this)

    if all items and their combinations pass the evaluation fn you get n**2 -1 solutions
    which leads to (2**n - 1) calls of your evaluation fn

    it needs more evaluations than finding strongly connected components in a graph because:
    (1,5) and (1,6) and (5,6) may work but (1,5,6) may not pass evaluation, (n,m) being list idx's

    the evaluation fn should accept idecies to your list and the list
    it should return a binary result on wether this set is good
    """

    def down(path,l,fn):
        # print "@",path
        ret = [path]
        for next in range(path[-1]+1,len(l)):
            if fn(path+[next],l):
                ret.extend(down(path+[next],l,fn))
        return ret


    ret = []
    for node in range(0,len(l)):
        if fn([node],l):
            ret.extend(down([node],l,fn))
    return ret


def pruning_quick_combine(l,fn,seed_idx=None):
    """
    l is a list of object to quick_combine.
    the evaluation fn should accept idecies to your list and the list
    it should return a binary result on wether this set is good
    it should accept a list it can use as storage

    this search finds all combinations but assumes:
        that a bad subset can not be bettered by adding more nodes
        that a good set may not always be improved by a 'passing' superset (purging subsets will revoke this)

    if all items and their combinations pass the evaluation fn you get n**2 -1 solutions
    which leads to (2**n - 1) calls of your evaluation fn

    it needs more evaluations than finding strongly connected components in a graph because:
    (1,5) and (1,6) and (5,6) may work but (1,5,6) may not pass evaluation, (n,m) being list idx's

    """
    if seed_idx:
        #Warning right now, seeds need to be before non-seeds
        unknown = [[node] for node in seed_idx]
    else:
        #start from every item
        unknown = [[node] for node in range(len(l))]
    results = []
    cache = []
    while unknown:
        path = unknown.pop(0)
        if fn(path,l,cache): # is this a good combination?
            results.append(path)
            decedents = [path+[i] for i in range(path[-1]+1,len(l)) ]
            unknown.extend(decedents)

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
    # pl = np.array([[[0, 0]],[[0, 1]],[[1, 1]],[[2, 1]],[[2, 2]],[[1, 3]],[[1, 4]],[[2,4]]], dtype=np.int32)
    # curvature = GetAnglesPolyline(pl)
    # print curvature
    # print find_curv_disc(curvature)
    # idx =  find_kink_and_dir_change(curvature,60)
    # print idx
    # print split_at_corner_index(pl,idx)
    ellipse = ((1,0),(2,1),0)
    pts = np.array([(2,0),(1.9,0),(-1.9,0)])
    # print dist_pts_ellipse(ellipse,pts)


    l = [1,3,1,2,1,2,0,2,2,4,4]
    # evals = 0
    # r = quick_combine(l,metric)
    # # print r
    # print filter_subsets(r)
    # print evals

    evals = 0
    r = pruning_quick_combine(l,cached_metric,[0])
    # print r
    print filter_subsets(r)
    print evals




