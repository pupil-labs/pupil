'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import cv2
import logging
logger = logging.Logger(__name__)
import numpy as np
from scipy.spatial.distance import pdist
from scipy.interpolate import interp1d
#because np.sqrt is slower when we do it on small arrays
def reversedEnumerate(l):
    return zip(range(len(l)-1, -1, -1), reversed(l))

from math import sqrt
sqrt_2 = sqrt(2)

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
    # msg = square_img[start::step,start::step]

    #resize to grid size
    msg = cv2.resize(square_img,(grid,grid),interpolation=cv2.INTER_LINEAR)
    msg = msg>50 #threshold


    #resample to 4 pixel per gridcell. using linear interpolation
    soft_msg = cv2.resize(square_img,(grid*2,grid*2),interpolation=cv2.INTER_LINEAR)
    #take the area mean to get a soft msg bit.
    soft_msg = cv2.resize(soft_msg,(grid,grid),interpolation=cv2.INTER_AREA)



    # border is: first row - last row and  first column - last column
    if msg[0::grid-1,:].any() or msg[:,0::grid-1].any():
        # logger.debug("This is not a valid marker: \n %s" %msg)
        return None
    # strip border to get the message
    msg = msg[1:-1,1:-1]
    soft_msg = soft_msg[1:-1,1:-1]

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
    soft_msg = np.rot90(soft_msg,-angle-2).transpose()
    # Marker Encoding
    #  W |LSB| W      ^
    #  1 | 2 | 3     / \ UP
    # MSB| 4 | W      |
    # print angle
    # print msg    #the message is displayed as you see in the image


    msg = msg.tolist()
    msb = msg_int

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

    #do the same for the soft msg image
    msg = soft_msg.tolist()
    msg_img = soft_msg
    #strip orientation corners from marker
    del msg[0][0]
    del msg[0][-1]
    del msg[-1][0]
    del msg[-1][-1]

    soft_msg = [item/255. for sublist in msg for item in sublist]+[float(msb)]
    return angle,msg_int,soft_msg,msg_img


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

    _img, contours, hierarchy = cv2.findContours(edges,
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
    size = 20*grid_size
    #top left,bottom left, bottom right, top right in image
    mapped_space = np.array( ((0,0),(size,0),(size,size),(0,size)) ,dtype=np.float32).reshape(4,1,2)
    for r in rect_cand:
        if correct_gradient(gray_img,r):
            r = np.float32(r)
            # define the criteria to stop and refine the marker verts
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            cv2.cornerSubPix(gray_img,r,(3,3),(-1,-1),criteria)

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
                angle,msg,soft_msg,msg_img = marker

                centroid = r.sum(axis=0)/4.
                centroid.shape = (2)
                # angle is number of 90deg rotations
                # roll points such that the marker points correspond with oriented marker
                # rolling may not make the verts appear as you expect,
                # but using m_screen_to_marker() will get you the marker with proper rotation.
                r = np.roll(r,angle+1,axis=0) #np.roll is not the fastest when using these tiny arrays...

                # id_confidence = 2*np.mean (np.abs(np.array(soft_msg)-.5 ))
                id_confidence = 2* min(np.abs(np.array(soft_msg)-.5 ))

                marker = {'id':msg,'id_confidence':id_confidence,'verts':r.tolist(),'soft_id':soft_msg,'perimeter':cv2.arcLength(r,closed=True),'centroid':centroid.tolist(),"frames_since_true_detection":0}
                if visualize:
                    marker['otsu'] = np.rot90(otsu,-angle-2).transpose()
                    marker['img'] = cv2.resize(msg_img,(20*grid_size,20*grid_size),interpolation=cv2.INTER_NEAREST)
                if marker['id'] != 32: #marker 32 sucks because its just a single white spec.
                    markers.append(marker)
    return markers



def draw_markers(img,markers):
    for m in markers:
        centroid = np.array(m['centroid'],dtype=np.float32)
        origin = np.array(m['verts'][0],dtype=np.float32)
        hat = np.array([[[0,0],[0,1],[.5,1.25],[1,1],[1,0]]],dtype=np.float32)
        hat = cv2.perspectiveTransform(hat,m_marker_to_screen(m))
        if m['id_confidence']>.9:
            cv2.polylines(img,np.int0(hat),color = (0,0,255),isClosed=True)
        else:
            cv2.polylines(img,np.int0(hat),color = (0,255,0),isClosed=True)
        # cv2.polylines(img,np.int0(centroid),color = (255,255,int(255*m['id_confidence'])),isClosed=True,thickness=2)
        m_str = 'id: {:d}'.format(m['id'])
        org = origin.copy()
        # cv2.rectangle(img, tuple(np.int0(org+(-5,-13))[0,:]), tuple(np.int0(org+(100,30))[0,:]),color=(0,0,0),thickness=-1)
        cv2.putText(img,m_str,tuple(np.int0(org)[0,:]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,0,255))
        if 'id_confidence' in m:
            m_str = 'idc: {:.3f}'.format(m['id_confidence'])
            org += (0, 12)
            cv2.putText(img,m_str,tuple(np.int0(org)[0,:]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,0,255))
        if 'loc_confidence' in m:
            m_str = 'locc: {:.3f}'.format(m['loc_confidence'])
            org += (0, 12 )
            cv2.putText(img,m_str,tuple(np.int0(org)[0,:]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,0,255))
        if 'frames_since_true_detection' in m:
            m_str = 'otf: {}'.format(m['frames_since_true_detection'])
            org += (0, 12 )
            cv2.putText(img,m_str,tuple(np.int0(org)[0,:]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,0,255))
        if 'opf_vel' in m:
            m_str = 'otf: {}'.format(m['opf_vel'])
            org += (0, 12 )
            cv2.putText(img,m_str,tuple(np.int0(org)[0,:]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,0,255))


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
    return cv2.getPerspectiveTransform(mapped_space_one,np.array(marker['verts'],dtype=np.float32))


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
    return cv2.getPerspectiveTransform(np.array(marker['verts'],dtype=np.float32),mapped_space_one)





#persistent vars for detect_markers_robust
lk_params = dict( winSize  = (45, 45),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))

prev_img = None
tick = 0

def detect_markers_robust(gray_img,grid_size,prev_markers,min_marker_perimeter=40,aperture=11,visualize=False,true_detect_every_frame = 1,invert_image=False):
    global prev_img

    if invert_image:
        gray_img = 255-gray_img

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
            prev_pts = np.array([np.array(m['verts'],dtype=np.float32) for m in not_found])
            prev_pts = np.vstack(prev_pts)
            new_pts, flow_found, err = cv2.calcOpticalFlowPyrLK(
                prev_img, gray_img, prev_pts, None,
                minEigThreshold=.01,**lk_params)
            for marker_idx in range(flow_found.shape[0]//4):
                m = not_found[marker_idx]
                m_slc = slice(marker_idx*4,marker_idx*4+4)
                if flow_found[m_slc].sum() >= 4:
                    found, _  = np.where(flow_found[m_slc])
                    # calculate differences
                    old_verts = prev_pts[m_slc][found,:]
                    new_verts = new_pts[m_slc][found,:]
                    vert_difs = new_verts - old_verts
                    # calc mean dif
                    mean_dif = vert_difs.mean(axis=0)
                    # take n-1 closest difs
                    dist_variance = np.linalg.norm(mean_dif - vert_difs,axis=1)
                    if max(np.abs(dist_variance).flatten())>5:
                        m["frames_since_true_detection"] = 100
                    else:
                        closest_mean_dif = np.argsort(dist_variance,axis=0)[:-1,0]
                        # recalc mean dif
                        mean_dif = vert_difs[closest_mean_dif].mean(axis=0)
                        # apply mean dif
                        proj_verts = prev_pts[m_slc] + mean_dif
                        m['verts'] = new_verts.tolist()
                        m['centroid'] = new_verts.sum(axis=0)/4.
                        m['centroid'].shape = (2)
                        m['centroid'] = m['centroid'].tolist()
                        m["frames_since_true_detection"] +=1
                        # m['opf_vel'] = mean_dif
                else:
                    m["frames_since_true_detection"] = 100


        #cocatenating like this will favour older markers in the doublication deletion process
        markers = [m for m in not_found if m["frames_since_true_detection"] < 5 ]+new_markers
        if markers: #del double detected markers
            min_distace = min([m['perimeter'] for m in markers])/4.
            # min_distace = 50
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



# def bench(folder):
#     from os.path import join
#     from video_capture.av_file_capture import File_Capture
#     cap = File_Capture(join(folder,'marker-test.mp4'))
#
#     tracker = MarkerTracker()
#     detected_count = 0
#     for x in range(500):
#         frame = cap.get_frame()
#         img = frame.img
#         gray_img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
#         markers = tracker.track_in_frame(gray_img,5,visualize=True)
#         draw_markers(img, markers)
#         cv2.imshow('Detected Markers', img)
#         if cv2.waitKey(1) == 27:
#            break
#         detected_count += len(markers)
#
#     print detected_count #3106 #3226


def bench(folder):
    from os.path import join
    from video_capture.av_file_capture import File_Capture
    cap = File_Capture(join(folder,'marker-test.mp4'))
    markers = []
    detected_count = 0

    for x in range(500):
        frame = cap.get_frame()
        img = frame.img
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        markers = detect_markers_robust(gray_img,5,prev_markers=markers,true_detect_every_frame=1,visualize=True)

        draw_markers(img, markers)
        cv2.imshow('Detected Markers', img)

        # for m in markers:
        #     if 'img' in m:
        #         cv2.imshow('id %s'%m['id'], m['img'])
        #         cv2.imshow('otsu %s'%m['id'], m['otsu'])
        if cv2.waitKey(1) == 27:
           break
        detected_count += len(markers)
    print(detected_count) #2900 #3042 #3021


if __name__ == '__main__':
    folder = '/Users/mkassner/Desktop/'
    import cProfile,subprocess,os
    cProfile.runctx("bench(folder)",{'folder':folder},locals(),os.path.join(folder, "world.pstats"))
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("cd {} ; python {} -f pstats world.pstats | dot -Tpng -o world_cpu_time.png".format(folder, gprof2dot_loc), shell=True)
    print("created  time graph for  process. Please check out the png next to this file")
