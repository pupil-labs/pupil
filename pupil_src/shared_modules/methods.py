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



    # # var  = np.var(curvature)
    # return mean,dif


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

def fit_ellipse(debug_img,edges,bin_dark_img, contour_size=80,target_ratio=1.0,target_size=20.,size_tolerance=20.):
    """ fit_ellipse: old, please look at pupil_detectors.py
    """
    c_img = edges.copy()
    contours, hierarchy = cv2.findContours(c_img,
                                            mode=cv2.RETR_LIST,
                                            method=cv2.CHAIN_APPROX_NONE,offset=(0,0)) #TC89_KCOS
    # contours is a list containging array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )

    contours = [c for c in contours if c.shape[0]>contour_size]

    cv2.drawContours(debug_img, contours, -1, (255,255,255),thickness=1,lineType=cv2.cv.CV_AA)

    # print contours

    # cv2.drawContours(debug_img, contours, -1, (255,255,255),thickness=1)
    good_contours = contours
    # split_contours = []
    # i = 0
    # for c in contours:
    #   curvature = np.abs(GetAnglesPolyline(c))
    #   kink= extract_at_angle(c,curvature,150)
    #   cv2.drawContours(debug_img, kink, -1, (255,0,0),thickness=2)
    #   split_contours += split_at_angle(c,curvature,150)


    # # good_contours = split_contours
    # good_contours = []
    # for c in split_contours:
    #   i +=40
    #   kink =  convexity_defect(c,GetAnglesPolyline(c))
    #   cv2.drawContours(debug_img, kink, -1, (255,0,0),thickness=1)
    #   if c.shape[0]/float(len(kink)+1)>3 and c.shape[0]>=5:
    #       cv2.drawContours(debug_img, np.array([c]), -1, (i,i,i),thickness=1)
    #       good_contours.append(c)
    #   else:
    #       cv2.drawContours(debug_img, np.array([c]), -1, (255,0,0),thickness=1)

    # for c in split_contours:
    #   kink =  convexity_defect(c,GetAnglesPolyline(c))
    #   cv2.drawContours(debug_img, kink, -1, (255,0,0),thickness=1)
    #   if c.shape[0]/float(len(kink)+1)>3 and c.shape[0]>=5:
    #       cv2.drawContours(debug_img, c, -1, (0,255,0),thickness=1)
    #       good_contours.append(c)


    # cv2.drawContours(debug_img, good_contours, -1, (0,255,0),thickness=1)

    # split_contours.sort(key=lambda c: -c.shape[0])
    # for c in split_contours[:]:
    #   if len(c)>=5:
    #       cv2.drawContours(debug_img, c[0:1], -1, (0,0,255),thickness=2)
    #       cv2.drawContours(debug_img, c[-1:], -1, (0,255,0),thickness=2)
    #       cv2.drawContours(debug_img, c, -1, (0,255,0),thickness=1)

    #       # cv2.polylines(debug_img,[c[:,0]], isClosed=False, color=(255,0,0),thickness= 1)
    #       good_contours.append(c)

    # cv2.drawContours(debug_img, good_contours, -1, (255,255,255),thickness=1)
    # good_contours = np.concatenate(good_contours)
    # good_contours = [good_contours]

    shape = edges.shape
    ellipses = (cv2.fitEllipse(c) for c in good_contours)
    ellipses = (e for e in ellipses if (0 <= e[0][1] <= shape[0] and 0<= e[0][0] <= shape[1]))
    ellipses = (e for e in ellipses if bin_dark_img[e[0][1],e[0][0]])
    ellipses = ((size_deviation(e,target_size),e) for e in ellipses if is_round(e,target_ratio)) # roundness test
    ellipses = [(size_dif,e) for size_dif,e in ellipses if size_dif<size_tolerance ] # size closest to target size
    ellipses.sort(key=lambda e: e[0]) #sort size_deviation
    if ellipses:
        best_ellipse = {'center': (None,None),
                'axes': (None, None),
                'angle': None,
                'area': 0.0,
                'ratio': None,
                'major': None,
                'minor': None}

        largest = ellipses[0][1]
        best_ellipse['center'] = largest[0]
        best_ellipse['angle'] = largest[-1]
        best_ellipse['axes'] = largest[1]
        best_ellipse['major'] = max(largest[1])
        best_ellipse['minor'] = min(largest[1])
        best_ellipse['ratio'] = best_ellipse['minor']/best_ellipse['major']
        return best_ellipse,ellipses
    return None

def is_round(ellipse,ratio,tolerance=.8):
    center, (axis1,axis2), angle = ellipse

    if axis1 and axis2 and abs( ratio - min(axis2,axis1)/max(axis2,axis1)) <  tolerance:
        return True
    else:
        return False

def size_deviation(ellipse,target_size):
    center, axis, angle = ellipse
    return abs(target_size-max(axis))



def convexity_2(contour,img=None):
    if img is not None:
        hull = cv2.convexHull(contour, returnPoints=1)
        cv2.drawContours(img, hull, -1, (255,0,0))
    hull = cv2.convexHull(contour, returnPoints=0)
    if len(hull)>12:
        res = cv2.convexityDefects(contour, hull) # return: start_index, end_index, farthest_pt_index, fixpt_depth)
        if res is  None:
            return False
        if len(res)>2:
            return True
    return False


def convexity(contour,img=None):
    if img is not None:
        hull = cv2.convexHull(contour, returnPoints=1)
        cv2.drawContours(img, hull, -1, (255,0,0))

    hull = cv2.convexHull(contour, returnPoints=0)
    if len(hull)>3:
        res = cv2.convexityDefects(contour, hull) # return: start_index, end_index, farthest_pt_index, fixpt_depth)
        if res is  None:
            return False
        if len(res)>3:
            return True
    return False


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
    curvature = GetAnglesPolyline(pl)
    print curvature
    print find_curv_disc(curvature)
    # idx =  find_kink_and_dir_change(curvature,60)
    # print idx
    # print split_at_corner_index(pl,idx)
