'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
import numpy as np
from scipy.spatial.distance import pdist
#because np.sqrt is slower when we do it on small arrays
from math import sqrt

def get_close_markers(markers,centroids=None, min_distance=20):
    if centroids is None:
        centroids = [m['centroid']for m in markers]
    centroids = np.array(centroids)

    ti = np.triu_indices(centroids.shape[0], 1)
    def full_idx(i):
        #get the pair from condensed matrix index
        #defindend inline because ti changes every time
        return np.array([ti[0][i], ti[1][i]])

    #calculate pairwise distance, return dense distace matrix (upper triangle)
    distances =  pdist(centroids,'euclidean')

    close_pairs = np.where(distances<min_distance)
    return full_idx(close_pairs)



def decode(square_img,grid):
    step = square_img.shape[0]/grid
    start = step/2
    #look only at the center point of each grid cell
    msg = square_img[start::step,start::step]
    # border is: first row - last row and  first column - last column
    if msg[0::grid-1,:].any() or msg[:,0::grid-1].any():
        # logger.debug("This is not a valid marker: \n %s" %msg)
        return None
    # strip border to get the message
    msg = msg[1:-1,1:-1]/255

    # out first bit is encoded in the orientation corners of the marker:
    #               MSB = 0                   MSB = 1
    #               W|*|*|W   ^               B|*|*|B   ^
    #               *|*|*|*  / \              *|*|*|*  / \
    #               *|*|*|*   |  UP           *|*|*|*   |  UP
    #               B|*|*|W   |               W|*|*|B   |
    # 0,0 -1,0 -1,-1, 0,-1
    # angles are counter-clockwise rotation
    corners = msg[0,0], msg[-1,0], msg[-1,-1], msg[0,-1]

    if sum(corners) == 3:
        msg_int = 0
    elif sum(corners) ==1:
        msg_int = 1
        corners = tuple([1-c for c in corners]) #just inversion
    else:
        #this is no valid marker but maybe a maldetected one? We return unknown marker with None rotation
        return None

    #read rotation of marker by now we are guaranteed to have 3w and 1b
    #angle is number of 90deg rotations
    if corners == (0,1,1,1):
        angle = 3
    elif corners == (1,0,1,1):
        angle = 0
    elif corners == (1,1,0,1):
        angle = 1
    else:
        angle = 2

    msg = np.rot90(msg,-angle-2).transpose()
    # Marker Encoding
    #  W |LSB| W      ^
    #  1 | 2 | 3     / \ UP
    # MSB| 4 | W      |
    # print angle
    # print msg    #the message is displayed as you see in the image


    msg = msg.tolist()

    #strip orientation corners from marker
    del msg[0][0]
    del msg[0][-1]
    del msg[-1][0]
    del msg[-1][-1]
    #flatten list
    msg = [item for sublist in msg for item in sublist]
    while msg:
        # [0,1,0,1] -> int [MSB,bit,bit,...,LSB], note the MSB is definde above
        msg_int = (msg_int<<1) + msg.pop()
    return angle,msg_int


def correct_gradient(gray_img,r):
    # used just to increase speed - this simple check is still way to slow
    # lets assume that a marker has a black border
    # we check two pixels one outside, one inside both close to the border
    p1,_,p2,_ = r.reshape(4,2).tolist()
    vector_across = p2[0]-p1[0],p2[1]-p1[1]
    ratio = 5./sqrt(vector_across[0]**2+vector_across[1]**2) #we want to measure 5px away from the border
    vector_across = int(vector_across[0]*ratio) , int(vector_across[1]*ratio)
    #indecies are flipped because numpy is row major
    outer = p1[1] - vector_across[1],  p1[0] - vector_across[0]
    inner = p1[1] + vector_across[1] , p1[0] + vector_across[0]
    try:
        gradient = int(gray_img[outer]) - int(gray_img[inner])
        return gradient > 20 #at least 20 shades darker inside
    except:
        #px outside of img frame, let the other method check
        return True


def detect_markers(gray_img,grid_size,min_marker_perimeter=40,aperture=11,visualize=False):
    edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, aperture, 9)

    _ ,contours, hierarchy = cv2.findContours(edges,
                                    mode=cv2.RETR_TREE,
                                    method=cv2.CHAIN_APPROX_SIMPLE,offset=(0,0)) #TC89_KCOS

    # remove extra encapsulation
    hierarchy = hierarchy[0]
    contours = np.array(contours)
    # keep only contours                        with parents     and      children
    contained_contours = contours[np.logical_and(hierarchy[:,3]>=0, hierarchy[:,2]>=0)]
    # turn on to debug contours
    # cv2.drawContours(gray_img, contours,-1, (0,255,255))
    # cv2.drawContours(gray_img, aprox_contours,-1, (255,0,0))

    # contained_contours = contours #overwrite parent children check

    #filter out rects
    aprox_contours = [cv2.approxPolyDP(c,epsilon=2.5,closed=True) for c in contained_contours]

    # any rectagle will be made of 4 segemnts in its approximation
    # also we dont need to find a marker so small that we cannot read it in the end...
    # also we want all contours to be counter clockwise oriented, we use convex hull fot this:
    rect_cand = [cv2.convexHull(c,clockwise=True) for c in aprox_contours if c.shape[0]==4 and cv2.arcLength(c,closed=True) > min_marker_perimeter]
    # a non convex quadrangle is not what we are looking for.
    rect_cand = [r for r in rect_cand if r.shape[0]==4]

    if visualize:
        cv2.drawContours(gray_img, rect_cand,-1, (255,100,50))


    markers = []
    size = 10*grid_size
    #top left,bottom left, bottom right, top right in image
    mapped_space = np.array( ((0,0),(size,0),(size,size),(0,size)) ,dtype=np.float32).reshape(4,1,2)
    for r in rect_cand:
        if correct_gradient(gray_img,r):
            r = np.float32(r)
            M = cv2.getPerspectiveTransform(r,mapped_space)
            flat_marker_img =  cv2.warpPerspective(gray_img, M, (size,size) )#[, dst[, flags[, borderMode[, borderValue]]]])

            # Otsu documentation here :
            # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
            _ , otsu = cv2.threshold(flat_marker_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # getting a cleaner display of the rectangle marker
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            cv2.erode(otsu,kernel,otsu, iterations=3)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # cv2.dilate(otsu,kernel,otsu, iterations=1)

            marker = decode(otsu, grid_size)
            if marker is not None:
                angle,msg = marker

                # define the criteria to stop and refine the marker verts
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                cv2.cornerSubPix(gray_img,r,(3,3),(-1,-1),criteria)

                centroid = r.sum(axis=0)/4.
                centroid.shape = (2)
                # angle is number of 90deg rotations
                # roll points such that the marker points correspond with oriented marker
                # rolling may not make the verts appear as you expect,
                # but using m_screen_to_marker() will get you the marker with proper rotation.
                r = np.roll(r,angle+1,axis=0) #np.roll is not the fastest when using these tiny arrays...

                r_norm = r/np.float32((gray_img.shape[1],gray_img.shape[0]))
                r_norm[:,:,1] = 1-r_norm[:,:,1]
                marker = {'id':msg,'verts':r,'verts_norm':r_norm,'centroid':centroid,"frames_since_true_detection":0}
                if visualize:
                    marker['img'] = np.rot90(otsu,-angle/90)
                markers.append(marker)

    return markers


def draw_markers(img,markers):
    for m in markers:
        centroid = [m['verts'].sum(axis=0)/4.]
        origin = m['verts'][0]
        hat = np.array([[[0,0],[0,1],[.5,1.25],[1,1],[1,0]]],dtype=np.float32)
        hat = cv2.perspectiveTransform(hat,m_marker_to_screen(m))
        cv2.polylines(img,np.int0(hat),color = (0,0,255),isClosed=True)
        cv2.polylines(img,np.int0(centroid),color = (255,255,0),isClosed=True,thickness=2)
        cv2.putText(img,'id: '+str(m['id']),tuple(np.int0(origin)[0,:]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,100,50))


def m_marker_to_screen(marker):
    #verts need to be sorted counterclockwise stating at bottom left
    #marker coord system:
    # +-----------+
    # |0,1     1,1|  ^
    # |           | / \
    # |           |  |  UP
    # |0,0     1,0|  |
    # +-----------+
    mapped_space_one = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
    return cv2.getPerspectiveTransform(mapped_space_one,marker['verts'])


def m_screen_to_marker(marker):
    #verts need to be sorted counterclockwise stating at bottom left
    #marker coord system:
    # +-----------+
    # |0,1     1,1|  ^
    # |           | / \
    # |           |  |  UP
    # |0,0     1,0|  |
    # +-----------+
    mapped_space_one = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
    return cv2.getPerspectiveTransform(marker['verts'],mapped_space_one)





#persistent vars for detect_markers_robust
lk_params = dict( winSize  = (45, 45),
                  maxLevel = 1,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
prev_img = None
tick = 0

def detect_markers_robust(gray_img,grid_size,prev_markers,min_marker_perimeter=40,aperture=11,visualize=False,true_detect_every_frame = 1):
    global prev_img

    global tick
    if not tick:
        tick = true_detect_every_frame
        new_markers = detect_markers(gray_img,grid_size,min_marker_perimeter,aperture,visualize)
    else:
        new_markers = []
    tick -=1


    if prev_img is not None and prev_markers:

        new_ids = [m['id'] for m in new_markers]

        #any old markers not found in the new list?
        not_found = [m for m in prev_markers if m['id'] not in new_ids and m['id'] >=0]
        if not_found:
            prev_pts = np.array([m['centroid'] for m in not_found])
            # new_pts, flow_found, err = cv2.calcOpticalFlowPyrLK(prev_img, gray_img,prev_pts,winSize=(100,100))
            new_pts, flow_found, err = cv2.calcOpticalFlowPyrLK(prev_img, gray_img,prev_pts,minEigThreshold=0.01,**lk_params)
            for pt,s,e,m in zip(new_pts,flow_found,err,not_found):
                if s: #ho do we ensure that this is a good move?
                    m['verts'] += pt-m['centroid'] #uniformly translate verts by optlical flow offset
                    r_norm = m['verts']/np.float32((gray_img.shape[1],gray_img.shape[0]))
                    r_norm[:,:,1] = 1-r_norm[:,:,1]
                    m['verts_norm'] = r_norm
                    m["frames_since_true_detection"] +=1
                else:
                    m["frames_since_true_detection"] =100


        #cocatenating like this will favour older markers in the doublication deletion process
        markers = [m for m in not_found if m["frames_since_true_detection"] < 10 ]+new_markers
        if 1: #del double detected markers
            min_distace = min_marker_perimeter/4.
            if len(markers)>1:
                remove = set()
                close_markers = get_close_markers(markers,min_distance=min_distace)
                for f,s in close_markers.T:
                    #remove the markers further down in the list
                    remove.add(s)
                remove = list(remove)
                remove.sort(reverse=True)
                for i in remove:
                    del markers[i]
    else:
        markers = new_markers


    prev_img = gray_img.copy()
    return markers



def bench():
    cap = cv2.VideoCapture('/Users/mkassner/Pupil/datasets/markers/many.mov')
    status,img = cap.read()
    markers = []
    while status:
        markers = detect_markers_robust(img,5,markers,true_detect_every_frame=1)
        status,img = cap.read()
        if markers:
            return



if __name__ == '__main__':
    import cProfile,subprocess,os
    cProfile.runctx("bench()",{},locals(),"world.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
    print "created  time graph for  process. Please check out the png next to this file"
