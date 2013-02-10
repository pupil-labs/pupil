import cv2 as cv
import numpy as np
import cProfile
import time

def main():
    save_video = False


    # change this path to point to the data folder you would like to play
    data_folder = "/Users/mkassner/Desktop/pupil_sample_videos/01/data000"

    video_path = data_folder + "/world.avi"
    gaze_positions_path = data_folder + "/gaze_positions.npy"
    record_path = data_folder + "/world_viz.avi"

    cap = cv.VideoCapture(video_path)
    gaze_list = list(np.load(gaze_positions_path))

    # this takes the gaze list and makes a list
    # with the length of the number of recorded frames.
    # Each slot conains a list that has 0, 1 or more assosiated gaze postions.
    positions_by_frame = [[] for frame in range(int(gaze_list[-1][-1]) + 1)]
    while gaze_list:
        s = gaze_list.pop(0)
        frame = int(s[-1])
        positions_by_frame[frame].append({'x': s[0], 'y': s[1], 'dt': s[2]})


    status, img = cap.read()
    prevgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = img.shape[0:2]
    frame = 0
    past_gaze = []
    t = time.time()

    fps = cap.get(5)
    wait =  int((1./fps)*1000)

    if save_video:
        #FFV1 -- good speed lossless big file
        #DIVX -- good speed good compression medium file
        writer = cv.VideoWriter(record_path, cv.cv.CV_FOURCC(*'DIVX'), fps, (img.shape[1], img.shape[0]))


    while status:
        nt = time.time()
        # print nt-t
        t = nt
        # apply optical flow displacement to previous gaze
        if past_gaze:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            prevPts = np.array(past_gaze,dtype=np.float32)
            nextPts, status, err = cv.calcOpticalFlowPyrLK(prevgray, gray,prevPts)
            prevgray = gray
            past_gaze = list(nextPts)

            #contrain gaze positions to
            c_gaze = []
            for x,y in past_gaze:
                if x >0 and x<width and y >0 and y <height:
                    c_gaze.append([x,y])
            past_gaze = c_gaze


        #load and map current gaze postions and append to the past_gaze list
        current_gaze = positions_by_frame[frame]
        for gaze_point in current_gaze:
            x,y = denormalize((gaze_point['x'], gaze_point['y']), width, height, flip_y=False)
            if x >0 and x<width and y >0 and y <height:
                past_gaze.append([x,y])


        vap = 10 #Visual_Attention_Span
        window_string = "the last %i frames of visual attention" %vap
        overlay = np.ones(img.shape[:-1],dtype=img.dtype)

        # remove everything but the last "vap" number of gaze postions from the list of past_gazes
        for x in xrange(len(past_gaze)-vap):
            past_gaze.pop(0)


        # draw recent gaze postions as white dots on an overlay image.
        for gaze_point in past_gaze[::-1]:
            try:
                overlay[int(gaze_point[1]),int(gaze_point[0])] = 0
            except:
                pass

        out = cv.distanceTransform(overlay,cv.cv.CV_DIST_L2, 5)

        wide = 1/(out/50+1)
        # narrow =  1/(out/20+1)

        img *=cv.cvtColor(wide,cv.COLOR_GRAY2RGB)

        cv.imshow(window_string, img)

        if save_video:
            writer.write(img)


        status, img = cap.read()
        frame += 1
        ch = cv.waitKey(wait)
        if ch == 27:
            break


def denormalize(pos, width, height, flip_y=True):
    """
    denormalize and return as int
    """
    x = pos[0]
    y = pos[1]
    if flip_y:
        y= -y
    x = (x * width / 2.) + (width / 2.)
    y = (y * height / 2.) + (height / 2.)
    return x,y


if __name__ == '__main__':
    main()