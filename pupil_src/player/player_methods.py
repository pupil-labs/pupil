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
#logging
import logging
logger = logging.getLogger(__name__)


def correlate_data(data,timestamps):
    '''
    data:  dict of data :
        will have at least:
            timestamp: float

    timestamps: timestamps list to correlate  data to

    this takes a data list and a timestamps list and makes a new list
    with the length of the number of timestamps.
    Each slot conains a list that will have 0, 1 or more assosiated data points.
    '''
    timestamps = list(timestamps)
    data_by_frame = [[] for i in timestamps]

    frame_idx = 0
    data_index = 0


    while True:
        try:
            datum = data[data_index]
            t_between_frames = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
        except IndexError:
            # we might loose a data point at the end but we dont care
            break

        if datum['timestamp'] <= t_between_frames:
            data_by_frame[frame_idx].append(datum)
            data_index +=1
        else:
            frame_idx+=1

    return data_by_frame





def is_pupil_rec_dir(data_dir):
    if not os.path.isdir(data_dir):
        logger.error("No valid dir supplied")
        return False
    required_files = ["info.csv", "gaze_positions.npy"]
    for f in required_files:
        if not os.path.isfile(os.path.join(data_dir,f)):
            logger.debug("Did not find required file: %s in data folder %s" %(f, data_dir))
            return False

    logger.debug("%s contains %s and is therefore considered a valid rec dir."%(data_dir,required_files))
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

