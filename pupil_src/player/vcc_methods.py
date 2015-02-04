import itertools
import numpy as np
import cv2

# Each frame has an index (ID) hierarchy tree defined by the cv2.RETR_TREE from cv2.findContours
# this is just to remember that its a tree
_RETR_TREE = 0

# Constants for the hierarchy[_RETR_TREE][contour][{next,back,child,parent}]
_ID_NEXT = 0
_ID_BACK = 1
_ID_CHILD = 2
_ID_PARENT = 3

# Channel constants
_CH_B = 0  
_CH_G = 1
_CH_R = 2
_CH_0 = 3

def ellipse_to_contour(ellipse, alfa):
    center = (int(round(ellipse[0][0])),int(round(ellipse[0][1]))) 
    axes = (int(round(ellipse[1][0]/alfa)),int(round(ellipse[1][1]/alfa)))
    angle = int(round(ellipse[2]))
    # delta == precision angle
    return cv2.ellipse2Poly(center, axes, angle, arcStart=0, arcEnd=360, delta=1)

def man_dist(e,other):
    return abs(e[0][0]-other[0][0])+abs(e[0][1]-other[0][1])

def get_cluster_hierarchy(ellipses,dist_threshold):
    cluster_hierarchy = []
    for e in ellipses:
        cluster_set = []
        for other in ellipses: 
            # distance to other ellipse is smaller than min dist threshold and minor of both ellipses
            if man_dist(e,other) < dist_threshold and man_dist(e,other) < min(min(*e[1]),min(other[1])):
                cluster_set.append(other)
            else:
                pass

        # sort by screen y
        cluster_set.sort(key = lambda e: (e[1][0] * e[1][1]))

        # avoid repetition
        if not cluster_set in cluster_hierarchy: 
            cluster_hierarchy.append(cluster_set)
        else:
            pass

    return cluster_hierarchy

def ellipses_from_findContours(img, cv2_thresh_mode, delta_area_threshold, visual_debug):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = cv2.adaptiveThreshold(gray_img, 255,
                                    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    thresholdType = cv2_thresh_mode,
                                    blockSize = 5,
                                    C = -3)

    contours, hierarchy = cv2.findContours(edges,
                                    mode = cv2.RETR_TREE,
                                    method = cv2.CHAIN_APPROX_NONE,
                                    offset = (0,0)) #TC89_KCOS

    # remove extra encapsulation
    hierarchy = hierarchy[0]
    # turn outmost list into array
    contours =  np.array(contours)
    # keep only contours                        with parents     and      children
    contained_contours = contours[np.logical_and(hierarchy[:, 3] >= 0, hierarchy[:,2] >= 0)]
    # turn on to debug contours
    if visual_debug:
        cv2.drawContours(img, contained_contours,-1, (0,0,255))

    # need at least 5 points to fit ellipse
    contained_contours =  [c for c in contained_contours if len(c) >= 5]

    ellipses = [cv2.fitEllipse(c) for c in contained_contours]
    candidate_ellipses = []
    # filter for ellipses that have similar area as the source contour
    for e,c in zip(ellipses, contained_contours):
        a,b = e[1][0] / 2., e[1][1] / 2.
        if abs(cv2.contourArea(c) - np.pi * a * b) < delta_area_threshold:
            candidate_ellipses.append(e)
    return candidate_ellipses

def get_cluster(ellipses,dist_threshold,min_ring_count):
    for e in ellipses:
        remainders = []
        close_ones = []
        for other in ellipses:
            # distance to other ellipse is smaller than min dist threshold and minor of both ellipses
            if man_dist(e,other) < dist_threshold and man_dist(e,other) < min(min(*e[1]),min(other[1])):
                close_ones.append(other)
            else:
                remainders.append(other)

        # breaking on first occurence of min_ring_count
        if len(close_ones) >= min_ring_count:
            # sort by major axis to return smallest ellipse first
            close_ones.sort(key = lambda e: max(e[1]))
            return close_ones, remainders
    return [], []

def get_canditate_ellipses(img, img_threshold, cv2_thresh_mode, area_threshold, dist_threshold, min_ring_count, visual_debug):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # get threshold image used to get crisp-clean edges
    # cv2.ADAPTIVE_THRESH_MEAN_C     
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    edges = cv2.adaptiveThreshold(gray_img, 255,
                                    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    thresholdType = cv2_thresh_mode,
                                    blockSize = 5,
                                    C = -3)
    #edges = cv2.threshold(gray_img, img_threshold, 255, cv2_thresh_mode)
    # cv2.flip(edges,1 ,dst = edges,)
    # display the image for debugging purpuses
    # img[:] = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
     # from edges to contours to ellipses CV_RETR_CCsOMP ls fr hole
    contours, hierarchy = cv2.findContours(edges,
                                    mode = cv2.RETR_TREE,
                                    method = cv2.CHAIN_APPROX_NONE,
                                    offset = (0,0)) #TC89_KCOS


    # remove extra encapsulation
    hierarchy = hierarchy[0]
    # turn outmost list into array
    contours =  np.array(contours)
    # keep only contours                        with parents     and      children
    contained_contours = contours[np.logical_and(hierarchy[:, 3] >= 0, hierarchy[:,2] >= 0)]
    # turn on to debug contours
    if visual_debug:
        cv2.drawContours(img, contained_contours,-1, (0,0,255))

    # need at least 5 points to fit ellipse
    contained_contours =  [c for c in contained_contours if len(c) >= 5]

    ellipses = [cv2.fitEllipse(c) for c in contained_contours]
    candidate_ellipses = []
    # filter for ellipses that have similar area as the source contour
    for e,c in zip(ellipses,contained_contours):
        a,b = e[1][0] / 2., e[1][1] / 2.
        if abs(cv2.contourArea(c) - np.pi * a * b) < area_threshold:
            candidate_ellipses.append(e)
    auxiliar = []
    remainders = []
    candidate_ellipses, remainders = get_cluster(candidate_ellipses,
                                                dist_threshold = dist_threshold,
                                                min_ring_count=min_ring_count)

    remainders, auxiliar = get_cluster(remainders,
                                                dist_threshold = dist_threshold,
                                                min_ring_count= min_ring_count -1)

    return candidate_ellipses, remainders

def find_edges(img, threshold, cv2_thresh_mode):
    blur = cv2.GaussianBlur(img,(5,5),0)
    #gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    edges = []
    # channels = (blur[:,:,_CH_B], blur[:,:,_CH_G], blur[:,:,_CH_R])
    # channels =  cv2.split(blur)
    for gray in (blur[:,:,_CH_B], blur[:,:,_CH_G], blur[:,:,_CH_R]):
        if threshold == 0:
            edg = cv2.Canny(gray, 0, 50, apertureSize = 5)
            edg = cv2.dilate(edg, None)
            edges.append(edg)
        else:
            retval, edg = cv2.threshold(gray, threshold, 255, cv2_thresh_mode)
            edges.append(edg)
    return edges

def find_edges2(img, threshold): #from calibration_routines/circle_detector.py
    edges = []
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray,(5,5),20)
    for channel in xrange(0,3,1):
        # get threshold image used to get crisp-clean edges
        edg = cv2.adaptiveThreshold(gray, threshold, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
        edges.append(edg)
    return edges

def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1) * np.dot(d2, d2) ) )

def is_circle(cnt):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    radius = w / 2
    isc = abs(1 - (w / h)) <= 1 and abs(1 - (area / (np.pi * pow(radius, 2)))) <= 20 #20, adjusted by trial and error
    return isc

def idx(depth, prime, hierarchy): #get_contour_id_from_depth
    if not depth == 0:
        next_level = hierarchy[_RETR_TREE][prime][_ID_PARENT]
        return idx(depth -1, next_level, hierarchy)
    else:
        return hierarchy[_RETR_TREE][prime][_ID_PARENT]  

# approximate and draw contour by its index
def draw_approx(img, index, contours, y, detection_color):
    cnt = contours[index]
    epsilon = y * cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    if cv2.isContourConvex(approx):
        cv2.drawContours(img,[approx],0 ,detection_color[:3],2) 
        return approx
    else:
        approx = []
        return approx

# get and draw contours
def draw_contours(img, contours, hierarchy, y, detection_color):    
    form_contours = []
    for i, cnt in enumerate(contours):
        epsilon = y * cv2.arcLength(cnt,True) # 0.007, adjusted by experimentation
        # logger.info('For epsilon= ' + str(epsilon))       
        approx = cv2.approxPolyDP(cnt,epsilon,True)

        if hierarchy[_RETR_TREE][i][_ID_CHILD] == -1: # if the contour has no child
            if cv2.isContourConvex(approx): 
                if len(approx) > 5:
                    if is_circle(approx) and cv2.contourArea(approx) > 1000:
                        cv2.drawContours(img,[approx],0 ,detection_color,1)
                        #form_contours.append(approx)
                        form_contours.append(approx)
                        #approx = draw_approx(img, idx(2, i, hierarchy), contours, y, detection_color)
                        if len(approx) > 0: 
                            form_contours.append(approx)
                        # cv2.putText(img, 'o', (int(x + w), int(y + h)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                    #else:
                        #cv2.drawContours(img,[approx],0 ,(255,255,0),1)
                        # cv2.putText(img, 'D', (int(x + w), int(y + h)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))        
                #elif len(approx) == 4:
                #    approx = approx.reshape(-1, 2)
                #    max_cos = np.max([angle_cos( approx[i], approx[(i + 1) % 4], approx[(i + 2) % 4] ) for i in xrange(4)])
                #    if max_cos < 0.1:                        
                #        cv2.drawContours(img,[approx],0 ,(200, 33, 50),1)
                        # cv2.putText(img, 'u"\u25A0"', (int(x + w), int(y + h)), cv2.FONT_HERSHEY_PLAIN, 1, (15, 255, 128))
            #else:
                #cv2.drawContours(img,[approx],0 ,(0, 255, 0),1) 
                # cv2.putText(img, 'c', (int(x + w), int(y + h)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))  
    return form_contours

def PolygonTestRC(contour, pt, count = 0, counter_code = ''):
    count += 1
    Inside = cv2.pointPolygonTest(contour, pt, False)
    if Inside > -1:
        counter_code = counter_code + '+' + str(count)
    else:
        counter_code = counter_code + '-' + str(count)
    return count, counter_code

def PolygonTestEx(contours, pt, contours_counter = 0, counter_code = ''):
    for x in xrange(1, len(contours) + 1):
        Inside = cv2.pointPolygonTest(contours[x -1], pt, False)
        # Inside = contours[x]
        if Inside > -1:
            counter_code = counter_code + '+' + str(x + contours_counter)
        else:
            counter_code = counter_code + '-' + str(x + contours_counter)
    contours_counter = contours_counter + len(contours)
    return contours_counter, counter_code

# http://math.stackexchange.com/questions/211645/what-is-the-number-of-all-possible-relations-intersections-of-n-sets
# http://codereview.stackexchange.com/questions/75524/tracking-eye-movements/75550#75550
# chars = '+-', base = 2
def get_codes(chars, base):
    numbers = range(1, base + 1)

    for signs in itertools.product(chars, repeat=base):
        yield "".join("{}{}".format(sign, n) for sign, n in zip(signs, numbers))
