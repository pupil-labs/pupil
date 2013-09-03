'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import sys,os
import cv2 as cv
import numpy as np


def main():

    save_video = False

    try:
        data_folder = sys.argv[1]
    except:
        print "You did not supply a datafolder when you called this script. \
               \nI will use the path hardcoded into the script instead."
        data_folder = "/Users/mkassner/Desktop/002"

    if not os.path.isdir(data_folder):
        raise Exception("Please supply a recording directory")


    video_path = data_folder + "/world.avi"
    timestamps_path = data_folder + "/timestamps.npy"
    gaze_positions_path = data_folder + "/gaze_positions.npy"
    record_path = data_folder + "/world_viz.avi"

    cap = cv.VideoCapture(video_path)
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
        if gaze_timestamp <= (timestamps[frame_idx]+timestamps[frame_idx+1])/2.:
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
    fps = 20
    wait =  int((1./fps)*1000)

    if save_video:
        #FFV1 -- good speed lossless big file
        #DIVX -- good speed good compression medium file
        writer = cv.VideoWriter(record_path, cv.cv.CV_FOURCC(*'DIVX'), fps, (img.shape[1], img.shape[0]))


    while status and frame < no_frames:

        # all gaze points of the current frame
        current_gaze = positions_by_frame[frame]
        for gaze_point in current_gaze:
            x_screen, y_screen = denormalize((gaze_point['x'], gaze_point['y']), width, height)
            cv.circle(img, (x_screen, y_screen), 30, (255, 255, 255), 2, cv.cv.CV_AA)

        cv.imshow("world", img)

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
        y=-y
    x = (x * width / 2.) + (width / 2.)
    y = (y * height / 2.) + (height / 2.)
    return int(x), int(y)

if __name__ == '__main__':
    main()