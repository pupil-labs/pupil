'''
(*)~----------------------------------------------------------------------------------

Find approximated contours
Draw convex parent contours of inner most contours 
if  gaze_point  is inside contour:
    draw circle a
else
    draw circle b

tested on black circles on white background
enviroment iluminated by fluorecent (127v 30watt) unclosed lamp on 4m high ceiling


slow performance for real time app  with standard find_edges_2 parameters

Author: Carlos Picanco.
Hack of simple_circle.py from Pupil - eye tracking platform (v0.3.7.4)
    
----------------------------------------------------------------------------------~(*)
'''
import sys,os
import cv2
import numpy as np

# hierarchy tree is defined by the cv2.RETR_TREE from cv2.findContours
# it is an index tree, not an array (contour) tree, duh...

# just to remember that it is from cv2.RETR_TREE
_RETR_TREE = 0

# Constants for the hierarchy[_RETR_TREE][contour][{next,back,child,parent}]
_ID_NEXT = 0
_ID_BACK = 1
_ID_CHILD = 2
_ID_PARENT = 3  

# squares
#def angle_cos(p0, p1, p2):
#    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
#    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def is_circle(cnt):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    radius = w / 2
    isc = abs(1 - (w / h)) <= 1 and abs(1 - (area / (np.pi * pow(radius, 2)))) <= 20 #20, adjusted by experimentation
    #if isc:
    #    print str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) 
    return isc

def idx(depth, prime, hierarchy): #get parent contour index
    if not depth == 0:
        next_level = hierarchy[_RETR_TREE][prime][_ID_PARENT]
        return idx(depth -1, next_level, hierarchy)
    else:
        return hierarchy[_RETR_TREE][prime][_ID_PARENT]  

def get_approx_and_draw (img, index, contours): #approximate and draw contour by its index
    cnt = contours[index]
    epsilon = 0.007 * cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    if cv2.isContourConvex(approx):
        cv2.drawContours(img,[approx],0 ,(255, 0 ,255),1) #-------------------------------------------------------------------pink
        return approx
    else:
        approx = []
        return approx

def get_form_contours(img, contours, hierarchy): #get and draw contours
    #drawing = np.zeros(frame.shape,np.uint8) # Image to draw the contours
    form_contours = []
    for i, cnt in enumerate(contours):
        # x, y, w, h = cv2.boundingRect(cnt) # putText arguments
        # hull = cv2.convexHull(cnt)
        # epsilon = 0.007 * cv2.arcLength(hull,True)            

        epsilon = 0.007 * cv2.arcLength(cnt,True) # 0.007, adjusted by experimentation       
        # epsilon = 3
        approx = cv2.approxPolyDP(cnt,epsilon,True)

        if hierarchy[_RETR_TREE][i][_ID_CHILD] == -1: # if the contour has no child
            if cv2.isContourConvex(approx): 
                if len(approx) > 5:
                    if is_circle(approx) and cv2.contourArea(approx) > 1000: #-------------------------------------------------red 
                        #cv2.drawContours(img,[approx],0 ,(0, 0, 255),1)
                        #form_contours.append(approx)

                        approx = get_approx_and_draw(img, idx(2, i, hierarchy), contours)
                        if len(approx) > 0: 
                            form_contours.append(approx)
                        # cv2.putText(img, 'o', (int(x + w), int(y + h)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                    #else:
                        #cv2.drawContours(img,[approx],0 ,(255,255,0),1) #--------------------------------------------------yellow
                        # cv2.putText(img, 'D', (int(x + w), int(y + h)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255)) 
                # squares
                #elif len(approx) == 4:
                #    approx = approx.reshape(-1, 2)
                #    max_cos = np.max([angle_cos( approx[i], approx[(i + 1) % 4], approx[(i + 2) % 4] ) for i in xrange(4)])
                #    if max_cos < 0.1:                        
                #        cv2.drawContours(img,[approx],0 ,(200, 33, 50),1)
                        # cv2.putText(img, 'u"\u25A0"', (int(x + w), int(y + h)), cv2.FONT_HERSHEY_PLAIN, 1, (15, 255, 128))
            #else: #----------------------------------------------------------------------------------------------------------------green
                #cv2.drawContours(img,[approx],0 ,(0, 255, 0),1) 
                # cv2.putText(img, 'c', (int(x + w), int(y + h)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
              
    return img, form_contours

def find_edges_2(img, cv2_thresh_mode):
    blur = cv2.GaussianBlur(img,(5,5),0)
    #gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
 
    for gray in cv2.split(blur):
        for thrs in xrange(0, 255, 26): # slow point
            if thrs == 0:
                edges = cv2.Canny(gray, 0, 50, apertureSize = 5)
                edges = cv2.dilate(edges, None)
            else:
                retval, edges = cv2.threshold(gray, thrs, 255, cv2_thresh_mode)
                #retval, edges = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return edges

def denormalize(pos, width, height, flip_y=True):
    """
    denormalize
    """
    x = pos[0]
    y = pos[1]
    x *= width
    if flip_y:
        y = 1-y
    y *= height
    return int(x),int(y)

def get_data_folder():
    try:
        data_folder = sys.argv[1]
    except:
        data_folder = '/home/recordings/000'

    if not os.path.isdir(data_folder):
        raise Exception("Please supply a recording directory as first argument, ex.: python 'exemple.py' 'directory'")
    else:
        print "Assigned path '" + data_folder + "'"
        return data_folder  

def main():

    save_video = True

    data_folder = get_data_folder()

    video_path = data_folder + "/world.avi"
    timestamps_path = data_folder + "/timestamps.npy"
    gaze_positions_path = data_folder + "/gaze_positions.npy"
    record_path = data_folder + "/world_simple_circle_if_inside_contour.avi"

    cap = cv2.VideoCapture(video_path)
    gaze_list = list(np.load(gaze_positions_path))
    timestamps = list(np.load(timestamps_path))
    # gaze_list: gaze x | gaze y | pupil x | pupil y | timestamp
    # timestamps timestamp

    # this takes the timestamps list and makes a list
    # with the length of the number of recorded frames.
    # Each slot conains a list that will have 0, 1 or more assosiated gaze postions.
    positions_by_frame = [[] for i in timestamps]


    no_frames = len(timestamps)
    frame_idx = 0
    data_point = gaze_list.pop(0)
    gaze_point = data_point[:2]
    gaze_timestamp = data_point[4]
    while gaze_list:
        # if the current gaze point is before the mean of the current world frame timestamp and the next worldframe timestamp
        if gaze_timestamp <= (timestamps[frame_idx] + timestamps[frame_idx + 1])/2.:
            positions_by_frame[frame_idx].append({'x': gaze_point[0],'y':gaze_point[1], 'timestamp':gaze_timestamp})
            data_point = gaze_list.pop(0)
            gaze_point = data_point[:2]
            gaze_timestamp = data_point[4]
        else:
            if frame_idx >= no_frames-2:
                break
            frame_idx+=1

    status, img = cap.read()
    height, width = img.shape[0:2]
    frame = 0

    fps = cap.get(5)
    # wait =  int((1./fps)*1000)

    if save_video:
        #FFV1 -- good speed lossless big file
        #DIVX -- good speed good compression medium file
        writer = cv2.VideoWriter(record_path, cv2.cv.CV_FOURCC(*'DIVX'), fps, (img.shape[1], img.shape[0]))

    past_gaze = []

    #print str(no_frames)
    while status and frame < no_frames:

        #find and draw contours
        edges = find_edges_2(img, cv2.THRESH_OTSU)
        contours,hierarchy = cv2.findContours(edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img, contours = get_form_contours(img, contours, hierarchy)

        #print 'frame ' + str(frame) + 'c: '+  str(len(fcontours))
        # gaze points if inside else outside contours
        current_gaze = positions_by_frame[frame]
        print str(current_gaze)
        for gaze_point in current_gaze:
            x_screen, y_screen = denormalize((gaze_point['x'], gaze_point['y']), width, height)
            for contour in contours:
                IsInside = cv2.pointPolygonTest(contour, (x_screen, y_screen), False)
                #for some reason circle a is being shown inside circle b
                if not IsInside == -1:
                    # circle a
                    cv2.circle(img, (x_screen, y_screen), 10, (60, 20, 220), 2, cv2.cv.CV_AA)
                else:
                    # circle b
                    cv2.circle(img, (x_screen, y_screen), 30, (60, 20, 220), 1, cv2.cv.CV_AA)

        cv2.imshow("world", img)

        if save_video:
            writer.write(img)

        status, img = cap.read()
        frame += 1
        ch = cv2.waitKey(1)
        if ch == 27:
            break

if __name__ == '__main__':
    main()
