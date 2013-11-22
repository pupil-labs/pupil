import cv2
import numpy as np

from scipy.spatial.distance import pdist,squareform



def get_close_markers(markers,centroids=None, min_distace=20):
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

    close_pairs = np.where(distances<min_distace)
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
        corners = tuple([1-c for c in corners]) #simple inversion
    else:
        #this is no valid marker but maybe a maldetected one? We return unknown marker with None rotation
        return None, -1

    #read rotation of marker by now we are guaranteed to have 3w and 1b
    if corners == (0,1,1,1):
        angle = 270
    elif corners == (1,0,1,1):
        angle = 0
    elif corners == (1,1,0,1):
        angle = 90
    else:
        angle = 180

    msg = np.rot90(msg,-angle/90)
    # W  |LSB| 1 |W      ^
    # 2  | 3 | 4 |5     / \
    # 6  | 7 | 8 |9      |  UP
    # MSB| 10| 11|W      |
    # print angle
    # print msg.transpose() # we align the output of print to align with pixels that you see
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



def detect_markers(gray_img,grid_size,min_marker_perimeter=40,aperture=11,visualize=False):
    edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, aperture, 9)

    contours, hierarchy = cv2.findContours(edges,
                                    mode=cv2.RETR_TREE,
                                    method=cv2.CHAIN_APPROX_SIMPLE,offset=(0,0)) #TC89_KCOS

    # remove extra encapsulation
    hierarchy = hierarchy[0]
    # turn outmost list into array
    contours =  np.array(contours)
    # keep only contours                        with parents     and      children
    contained_contours = contours[np.logical_and(hierarchy[:,3]>=0, hierarchy[:,2]>=0)]
    # turn on to debug contours
    # cv2.drawContours(gray_img, contours,-1, (0,255,255))
    # cv2.drawContours(gray_img, aprox_contours,-1, (255,0,0))

    # contained_contours = contours #overwrite parent children check

    #filter out rects
    aprox_contours = [cv2.approxPolyDP(c,epsilon=2.5,closed=True) for c in contained_contours]


    # any rectagle will be made of 4 segemnts in its approximation we dont need to find a marker so small that we cannot read it in the end...
    #also we want all contours to be counter clockwise oriented, we use convex hull fot this:
    rect_cand = [cv2.convexHull(c,clockwise=True) for c in aprox_contours if c.shape[0]==4 and cv2.arcLength(c,closed=True) > min_marker_perimeter]
    # a covex quadrangle is not what we are looking for.
    rect_cand = [r for r in rect_cand if r.shape[0]==4]

    if visualize:
        cv2.drawContours(gray_img, rect_cand,-1, (255,100,50))

    # subpixel corner fitting
    rects = np.array(rect_cand,dtype=np.float32)
    rects_shape = rects.shape
    rects.shape = (-1,2) #flatten for rectsubPix
    # define the criteria to stop and refine the rects
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    cv2.cornerSubPix(gray_img,rects,(3,3),(-1,-1),criteria)
    rects.shape = rects_shape #back to old layout [[rect],[rect],[rect]...] with rect = [corner,corner,corncer,corner]

    markers = []
    centroids = []
    for r in rects:

        size = 20*grid_size
        M = cv2.getPerspectiveTransform(r,np.array(((0,0),(size,0),(size,size),(0,size)),dtype=np.float32) ) #bottom left,top left, top right, bottom right in image
        flat_marker_img =  cv2.warpPerspective(gray_img, M, (size,size) )#[, dst[, flags[, borderMode[, borderValue]]]])

        # Otsu documentation here :
        # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
        _ , otsu = cv2.threshold(flat_marker_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # cosmetics -- getting a cleaner display of the rectangle marker
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        cv2.erode(otsu,kernel,otsu, iterations=3)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # cv2.dilate(otsu,kernel,otsu, iterations=1)

        marker = decode(otsu, grid_size)
        if marker is not None:
            angle,msg = marker
            centroid = r.sum(axis=0)/4.
            centroid.shape = (2)
            # roll points such that the marker points correspond with oriented marker
            if angle is not None:
                r = np.roll(r,angle/90+1,axis=0) #not the fastest when using these tiny arrays...

            # this way we get the matrix transform with rotation included
            marker_to_screen = cv2.getPerspectiveTransform(np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32),r)
            screen_to_marker = cv2.getPerspectiveTransform(r,np.array(((0.,0.),(0.,1),(1,1),(1,0.)),dtype=np.float32))
            #marker coord system:
            # +-----------+
            # |0,1     1,1|  ^
            # |           | / \
            # |           |  |  UP
            # |0,0     1,0|  |
            # +-----------+
            # marker to be returned/broadcast out -- accessible to world
            # verts are sorted counterclockwise with vert[0]=0,0 (origin) vert[1]= 1,0 vert[2] = 1,1 vert[3] 0,1
            marker = {'id':msg,'verts':r,'marker_to_screen':marker_to_screen,'screen_to_marker':screen_to_marker,'centroid':centroid,"frames_since_true_detection":0}
            if visualize and angle is not None:
                marker['img'] = np.rot90(otsu,-angle/90)
            markers.append(marker)
            centroids.append(centroid)

    if 1: #del double detected markers
        min_distace = min_marker_perimeter/4
        if len(centroids)>1:
                remove = set()
                close_markers = get_close_markers(markers,centroids,min_distace)
                for f,s in close_markers.T:
                    if cv2.arcLength(markers[f]['verts'],closed=True) < cv2.arcLength(markers[s]['verts'],closed=True):
                        remove.add(f)
                    else:
                        remove.add(s)
                remove = list(remove)
                remove.sort(reverse=True)
                for i in remove:
                    del markers[i]
    return markers


def draw_markers(img,markers):
    for m in markers:
        centroid = [m['verts'].sum(axis=0)/4.]
        origin = m['verts'][0]
        hat = np.array([[[0,0],[0,1],[.5,1.5],[1,1],[1,0]]],dtype=np.float32)
        hat = cv2.perspectiveTransform(hat,m['marker_to_screen'])
        cv2.polylines(img,np.int0(hat),color = (0,0,255),isClosed=True)
        cv2.polylines(img,np.int0(centroid),color = (255,255,0),isClosed=True,thickness=2)
        cv2.putText(img,'id: '+str(m['id']),tuple(np.int0(origin)[0,:]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,100,50))


lk_params = dict( winSize  = (55, 55),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

prev_img = None

def detect_markers_simple(img,grid_size,min_marker_perimeter=40,aperture=11,visualize=False):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return detect_markers(gray_img,grid_size,min_marker_perimeter,aperture,visualize)


def detect_markers_robust(img,grid_size,prev_markers,min_marker_perimeter=40,aperture=11,visualize=False):
    global prev_img
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    new_markers = detect_markers(gray_img,grid_size,min_marker_perimeter,aperture,visualize)


    if prev_img is not None and prev_markers:

        #not looking of stub markers yet
        new_ids = [m['id'] for m in new_markers]

        #any old markers not found in the new list? - we ignore stub markers
        not_found = [m for m in prev_markers if m['id'] not in new_ids and m['id'] >=0]
        if not_found:
            prev_pts = np.array([m['centroid'] for m in not_found])
            # print 'before',prev_pts

            # we could use  a forward backward check as error mesure...
            # p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            # p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            # d = abs(p0-p0r).reshape(-1, 2).max(-1)
            # good = d < 1

            #we use err in this configurtation it is simple the disance the pt has moved/pix in window
            new_pts, flow_found, err = cv2.calcOpticalFlowPyrLK(prev_img, gray_img,prev_pts,**lk_params)
            for pt,s,e,m in zip(new_pts,flow_found,err,not_found):
                if s and e<3:
                    m['verts'] += pt-m['centroid']
                    m['marker_to_screen']= cv2.getPerspectiveTransform(np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32),m['verts'])
                    m['screen_to_marker'] = cv2.getPerspectiveTransform(m['verts'],np.array(((0.,0.),(0.,1),(1,1),(1,0.)),dtype=np.float32))
                    m["frames_since_true_detection"] +=1
                else:
                    m["frames_since_true_detection"] =100

        markers = new_markers+[m for m in not_found if m["frames_since_true_detection"] < 20 ]

    else:
        markers = new_markers


    prev_img = gray_img
    return markers



# class Marker(object):
#     """docstring for marker"""
#     def __init__(self, name,verts,marker_to_screen,screen_to_marker):
#         super(marker, self).__init__()
#         self.name = name
#         self.verts = verts
#         self.marker_to_screen = marker_to_screen
#         self.screen_to_marker = screen_to_marker
#         self.img = None

#     def gl_draw(self):
#         hat = np.array([[[0,0],[1,0],[1.5,.5],[1,1],[0,1],[0,0]]],dtype=np.float32)
#         hat = cv2.perspectiveTransform(hat,m['marker_to_screen'])
#         draw_gl_polyline(hat.reshape((6,2)),(0.1,1.,1.,.5))



class Reference_Surface(object):
    """docstring for Reference Surface"""
    def __init__(self, marker_names):
        self.marker_names = marker_names




if __name__ == '__main__':
    pass