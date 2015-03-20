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


def correlate_gaze(gaze_list,timestamps):
    '''
    gaze_list: timestamp | confidence | gaze x | gaze y |
    timestamps timestamps to correlate gaze data to


    this takes a gaze positions list and a timestamps list and makes a new list
    with the length of the number of recorded frames.
    Each slot conains a list that will have 0, 1 or more assosiated gaze postions.
    '''
    gaze_list = list(gaze_list)
    timestamps = list(timestamps)

    positions_by_frame = [[] for i in timestamps]

    frame_idx = 0
    try:
        data_point = gaze_list.pop(0)
    except:
        logger.warning("No gaze positons in this recording.")
        return positions_by_frame

    gaze_timestamp = data_point[0]

    while gaze_list:
        # if the current gaze point is before the mean of the current world frame timestamp and the next worldframe timestamp
        try:
            t_between_frames = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
        except IndexError:
            break
        if gaze_timestamp <= t_between_frames:
            ts,confidence,x,y, = data_point
            positions_by_frame[frame_idx].append({'norm_gaze':(x,y), 'confidence':confidence, 'timestamp':ts})
            data_point = gaze_list.pop(0)
            gaze_timestamp = data_point[0]
        else:
            frame_idx+=1

    return positions_by_frame


def correlate_gaze_legacy(gaze_list,timestamps):
    '''
    gaze_list: gaze x | gaze y | pupil x | pupil y | timestamp
    timestamps timestamps to correlate gaze data to


    this takes a gaze positions list and a timestamps list and makes a new list
    with the length of the number of recorded frames.
    Each slot conains a list that will have 0, 1 or more assosiated gaze postions.
    load gaze information
    '''
    gaze_list = list(gaze_list)
    timestamps = list(timestamps)

    positions_by_frame = [[] for i in timestamps]

    frame_idx = 0
    try:
        data_point = gaze_list.pop(0)
    except:
        logger.warning("No gaze positons in this recording.")
        return positions_by_frame

    gaze_timestamp = data_point[4]

    while gaze_list:
        # if the current gaze point is before the mean of the current world frame timestamp and the next worldframe timestamp
        try:
            t_between_frames = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
        except IndexError:
            break
        if gaze_timestamp <= t_between_frames:
            positions_by_frame[frame_idx].append({'norm_gaze':(data_point[0],data_point[1]),'norm_pupil': (data_point[2],data_point[3]), 'timestamp':data_point[4],'confidence':data_point[5]})
            data_point = gaze_list.pop(0)
            gaze_timestamp = data_point[4]
        else:
            frame_idx+=1

    return positions_by_frame



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

# backwards compatibility tools:

def patch_meta_info(rec_dir):
    #parse info.csv file

    '''
    This is how we need it:

    Recording Name  2014_01_21
    Start Date  21.01.2014
    Start Time  11:42:24
    Duration Time   00:00:29
    World Camera Frames 710
    World Camera Resolution 1280x720
    Capture Software Version    v0.3.7
    User    testing
    Platform    Linux
    Machine brosnan
    Release 3.5.0-45-generic
    Version #68~precise1-Ubuntu SMP Wed Dec 4 16:18:46 UTC 2013
    '''
    proper_names = ['Recording Name',
                    'Start Date',
                    'Start Time',
                    'Duration Time',
                    'World Camera Frames',
                    'World Camera Resolution',
                    'Capture Software Version',
                    'User',
                    'Platform',
                    'Release',
                    'Version']

    with open(rec_dir + "/info.csv") as info:
        meta_info = [line.strip().split('\t') for line in info.readlines() ]

    for entry in meta_info:
        for proper_name in proper_names:
            if proper_name == entry[0]:
                break
            elif proper_name in entry[0]:
                logger.info("Permanently updated info.csv field name: '%s' to '%s'."%(entry[0],proper_name))
                entry[0]=proper_name
                break

    new_info = ''
    for e in meta_info:
        new_info += e[0]+'\t'+e[1]+'\n'

    with open(rec_dir + "/info.csv",'w') as info:
        info.write(new_info)

def convert_gaze_pos(gaze_list,capture_version):
    '''
    util fn to update old gaze pos files to new coordsystem. UNTESTED!
    '''
    #lets make a copy here so that we are not making inplace edits of the passed array
    gaze_list = gaze_list.copy()
    if capture_version < .36:
        logger.info("Gaze list is from old Recoding, I will update the data to work with new coordinate system.")
        gaze_list[:,:4] += 1. #broadcasting
        gaze_list[:,:4] /= 2. #broadcasting
    return gaze_list


def transparent_circle(img,center,radius,color,thickness):
    center = tuple(map(int,center))
    rgb = [255*c for c in color[:3]] # convert to 0-255 scale for OpenCV
    alpha = color[-1]
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

