'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
#logging
import logging
logger = logging.getLogger(__name__)
from file_methods import save_object


def correlate_data(data,timestamps):
    '''
    data:  dict of data :
        will have at least:
            timestamp: float

    timestamps: timestamps list to correlate  data to

    this takes a data list and a timestamps list and makes a new list
    with the length of the number of timestamps.
    Each slot contains a list that will have 0, 1 or more assosiated data points.

    Finally we add an index field to the data_point with the assosiated index
    '''
    timestamps = list(timestamps)
    data_by_frame = [[] for i in timestamps]

    frame_idx = 0
    data_index = 0


    while True:
        try:
            datum = data[data_index]
            # we can take the midpoint between two frames in time: More appropriate for SW timestamps
            ts = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
            # or the time of the next frame: More appropriate for Sart Of Exposure Timestamps (HW timestamps).
            # ts = timestamps[frame_idx+1]
        except IndexError:
            # we might loose a data point at the end but we dont care
            break

        if datum['timestamp'] <= ts:
            datum['index'] = frame_idx
            data_by_frame[frame_idx].append(datum)
            data_index +=1
        else:
            frame_idx+=1

    return data_by_frame



def update_recording_0v4_to_current(rec_dir):
    logger.info("Updatig recording from v0.4x format to current version")
    gaze_array = np.load(os.path.join(rec_dir,'gaze_positions.npy'))
    pupil_array = np.load(os.path.join(rec_dir,'pupil_positions.npy'))
    gaze_list = []
    pupil_list = []

    for datum in pupil_array:
        ts, confidence, id, x, y, diameter = datum[:6]
        pupil_list.append({'timestamp':ts,'confidence':confidence,'id':id,'norm_pos':[x,y],'diameter':diameter})

    pupil_by_ts = dict([(p['timestamp'],p) for p in pupil_list])

    for datum in gaze_array:
        ts,confidence,x,y, = datum
        gaze_list.append({'timestamp':ts,'confidence':confidence,'norm_pos':[x,y],'base':[pupil_by_ts.get(ts,None)]})

    pupil_data = {'pupil_positions':pupil_list,'gaze_positions':gaze_list}
    try:
        save_object(pupil_data,os.path.join(rec_dir, "pupil_data"))
    except IOError:
        pass

def update_recording_0v3_to_current(rec_dir):
    logger.info("Updatig recording from v0.3x format to current version")
    pupilgaze_array = np.load(os.path.join(rec_dir,'gaze_positions.npy'))
    gaze_list = []
    pupil_list = []

    for datum in pupilgaze_array:
        gaze_x,gaze_y,pupil_x,pupil_y,ts,confidence = datum
        #some bogus size and confidence as we did not save it back then
        pupil_list.append({'timestamp':ts,'confidence':confidence,'id':0,'norm_pos':[pupil_x,pupil_y],'diameter':50})
        gaze_list.append({'timestamp':ts,'confidence':confidence,'norm_pos':[gaze_x,gaze_y],'base':[pupil_list[-1]]})

    pupil_data = {'pupil_positions':pupil_list,'gaze_positions':gaze_list}
    try:
        save_object(pupil_data,os.path.join(rec_dir, "pupil_data"))
    except IOError:
        pass

def is_pupil_rec_dir(rec_dir):
    if not os.path.isdir(rec_dir):
        logger.error("No valid dir supplied")
        return False
    meta_info_path = os.path.join(rec_dir,"info.csv")
    try:
        with open(meta_info_path) as info:
            meta_info = dict( ((line.strip().split('\t')) for line in info.readlines() ) )
            info = meta_info["Capture Software Version"]
    except:
        logger.error("Could not read info.csv file: Not a valid Pupil recording.")
        return False
    return True




def transparent_circle(img,center,radius,color,thickness):
    center = tuple(map(int,center))
    rgb = [255*c for c in color[:3]] # convert to 0-255 scale for OpenCV
    alpha = color[-1]
    radius = int(radius)
    if thickness > 0:
        pad = radius + 2 + thickness
    else:
        pad = radius + 3
    roi = slice(center[1]-pad,center[1]+pad),slice(center[0]-pad,center[0]+pad)

    try:
        overlay = img[roi].copy()
        cv2.circle(overlay,(pad,pad), radius=radius, color=rgb, thickness=thickness, lineType=cv2.cv.CV_AA)
        opacity = alpha
        cv2.addWeighted(overlay, opacity, img[roi], 1. - opacity, 0, img[roi])
    except:
        logger.debug("transparent_circle would have been partially outsize of img. Did not draw it.")


def transparent_image_overlay(pos,overlay_img,img,alpha):
    """
    Overlay one image with another with alpha blending
    In player this will be used to overlay the eye (as overlay_img) over the world image (img)
    Arguments:
        pos: (x,y) position of the top left corner in numpy row,column format from top left corner (numpy coord system)
        overlay_img: image to overlay
        img: destination image
        alpha: 0.0-1.0
    """
    roi = slice(pos[1],pos[1]+overlay_img.shape[0]),slice(pos[0],pos[0]+overlay_img.shape[1])
    try:
        cv2.addWeighted(overlay_img,alpha,img[roi],1.-alpha,0,img[roi])
    except:
        logger.debug("transparent_image_overlay was outside of the world image and was not drawn")
    pass

