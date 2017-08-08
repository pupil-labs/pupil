'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os, cv2, csv_utils
import numpy as np
from scipy.interpolate import interp1d
import collections
import glob
import av

# logging
import logging
logger = logging.getLogger(__name__)
from file_methods import save_object, load_object, UnpicklingError
from version_utils import VersionFormat
from version_utils import read_rec_version
from calibration_routines.camera_intrinsics_estimation import pre_recorded_calibrations, idealized_camera_calibration


def correlate_data(data, timestamps):
    '''
    data:  list of data :
        each datum is a dict with at least:
            timestamp: float

    timestamps: timestamps list to correlate  data to

    this takes a data list and a timestamps list and makes a new list
    with the length of the number of timestamps.
    Each slot contains a list that will have 0, 1 or more assosiated data points.

    Finally we add an index field to the datum with the associated index
    '''
    timestamps = list(timestamps)
    data_by_frame = [[] for i in timestamps]

    frame_idx = 0
    data_index = 0

    data.sort(key=lambda d: d['timestamp'])

    while True:
        try:
            datum = data[data_index]
            # we can take the midpoint between two frames in time: More appropriate for SW timestamps
            ts = (timestamps[frame_idx]+timestamps[frame_idx+1]) / 2.
            # or the time of the next frame: More appropriate for Sart Of Exposure Timestamps (HW timestamps).
            # ts = timestamps[frame_idx+1]
        except IndexError:
            # we might loose a data point at the end but we dont care
            break

        if datum['timestamp'] <= ts:
            datum['index'] = frame_idx
            data_by_frame[frame_idx].append(datum)
            data_index += 1
        else:
            frame_idx += 1

    return data_by_frame


def update_recording_to_recent(rec_dir):

    meta_info = load_meta_info(rec_dir)
    update_meta_info(rec_dir, meta_info)

    if (meta_info.get('Capture Software', 'Pupil Capture') == 'Pupil Mobile'
            and 'Data Format Version' not in meta_info):
        convert_pupil_mobile_recording_to_v094(rec_dir)
        meta_info['Data Format Version'] = 'v0.9.4'
        update_meta_info(rec_dir, meta_info)

    # Reference format: v0.7.4
    rec_version = read_rec_version(meta_info)

    # Convert python2 to python3
    if rec_version <= VersionFormat('0.8.7'):
        update_recording_bytes_to_unicode(rec_dir)

    if rec_version >= VersionFormat('0.7.4'):
        pass
    elif rec_version >= VersionFormat('0.7.3'):
        update_recording_v073_to_v074(rec_dir)
    elif rec_version >= VersionFormat('0.5'):
        update_recording_v05_to_v074(rec_dir)
    elif rec_version >= VersionFormat('0.4'):
        update_recording_v04_to_v074(rec_dir)
    elif rec_version >= VersionFormat('0.3'):
        update_recording_v03_to_v074(rec_dir)
    else:
        logger.Error("This recording is too old. Sorry.")
        return

    # Incremental format updates
    if rec_version < VersionFormat('0.8.2'):
        update_recording_v074_to_v082(rec_dir)
    if rec_version < VersionFormat('0.8.3'):
        update_recording_v082_to_v083(rec_dir)
    if rec_version < VersionFormat('0.8.6'):
        update_recording_v083_to_v086(rec_dir)
    if rec_version < VersionFormat('0.8.7'):
        update_recording_v086_to_v087(rec_dir)
    if rec_version < VersionFormat('0.9.1'):
        update_recording_v087_to_v091(rec_dir)
    if rec_version < VersionFormat('0.9.3'):
        update_recording_v091_to_v093(rec_dir)
    if rec_version < VersionFormat('0.9.4'):
        update_recording_v093_to_v094(rec_dir)
    if rec_version < VersionFormat('0.9.13'):
        update_recording_v094_to_v0913(rec_dir)

    # How to extend:
    # if rec_version < VersionFormat('FUTURE FORMAT'):
    #    update_recording_v081_to_FUTURE(rec_dir)


def load_meta_info(rec_dir):
    meta_info_path = os.path.join(rec_dir,"info.csv")
    with open(meta_info_path,'r',encoding='utf-8') as csvfile:
        meta_info = csv_utils.read_key_value_file(csvfile)
    return meta_info


def update_meta_info(rec_dir, meta_info):
    logger.info('Updating meta info')
    meta_info_path = os.path.join(rec_dir,"info.csv")
    with open(meta_info_path,'w',newline='') as csvfile:
        csv_utils.write_key_value_file(csvfile,meta_info)


def convert_pupil_mobile_recording_to_v094(rec_dir):
    logger.info("Converting Pupil Mobile recording to v0.9.4 format")
    # convert time files and rename corresponding videos
    time_pattern = os.path.join(rec_dir, '*.time')
    for time_loc in glob.glob(time_pattern):
        time_file_name = os.path.split(time_loc)[1]
        time_name, time_ext = os.path.splitext(time_file_name)

        potential_locs = [os.path.join(rec_dir, time_name+ext) for ext in ('.mjpeg', '.mp4','.m4a')]
        existing_locs = [loc for loc in potential_locs if os.path.exists(loc)]
        if not existing_locs:
            continue
        else:
            video_loc = existing_locs[0]

        if time_name in ('Pupil Cam1 ID0', 'Pupil Cam1 ID1'):
            time_name = 'eye'+time_name[-1]  # rename eye files
        elif time_name in ('Pupil Cam1 ID2', 'Logitech Webcam C930e'):
            cam_calib_loc = os.path.join(rec_dir, 'camera_calibration')
            try:
                camera_calibration = load_object(cam_calib_loc)
            except:
                # no camera calibration found
                video = av.open(video_loc, 'r')
                frame_size = video.streams.video[0].format.width, video.streams.video[0].format.height
                del video
                try:
                    camera_calibration = pre_recorded_calibrations[time_name][frame_size]
                except KeyError:

                    camera_calibration = idealized_camera_calibration(frame_size)
                    logger.warning('Camera calibration not found. Will assume idealized camera.')
                save_object(camera_calibration, cam_calib_loc)

            time_name = 'world'  # assume world file
        elif time_name.startswith('audio_'):
            time_name = 'audio'

        timestamps = np.fromfile(time_loc, dtype='>f8')
        timestamp_loc = os.path.join(rec_dir, '{}_timestamps.npy'.format(time_name))
        logger.info('Creating "{}"'.format(os.path.split(timestamp_loc)[1]))
        np.save(timestamp_loc, timestamps)

        if time_name == 'audio':
            video_dst = os.path.join(rec_dir, time_name) + '.mp4'
            logger.info('Renaming "{}" to "{}"'.format(os.path.split(video_loc)[1], os.path.split(video_dst)[1]))
            os.rename(video_loc, video_dst)
        else:
            video_dst = os.path.join(rec_dir, time_name) + os.path.splitext(video_loc)[1]
            logger.info('Renaming "{}" to "{}"'.format(os.path.split(video_loc)[1], os.path.split(video_dst)[1]))
            os.rename(video_loc, video_dst)

    pupil_data_loc = os.path.join(rec_dir, 'pupil_data')
    if not os.path.exists(pupil_data_loc):
        logger.info('Creating "pupil_data"')
        save_object({'pupil_positions': [],
                     'gaze_positions': [],
                     'notifications': []}, pupil_data_loc)


def update_recording_v074_to_v082(rec_dir):
    meta_info_path = os.path.join(rec_dir,"info.csv")
    with open(meta_info_path,'r',encoding='utf-8') as csvfile:
        meta_info = csv_utils.read_key_value_file(csvfile)
        meta_info['Data Format Version'] = 'v0.8.2'
    update_meta_info(rec_dir, meta_info)


def update_recording_v082_to_v083(rec_dir):
    logger.info("Updating recording from v0.8.2 format to v0.8.3 format")
    pupil_data = load_object(os.path.join(rec_dir, "pupil_data"))
    meta_info_path = os.path.join(rec_dir,"info.csv")

    for d in pupil_data['gaze_positions']:
        if 'base' in d:
            d['base_data'] = d.pop('base')

    save_object(pupil_data,os.path.join(rec_dir, "pupil_data"))

    with open(meta_info_path,'r',encoding='utf-8') as csvfile:
        meta_info = csv_utils.read_key_value_file(csvfile)
        meta_info['Data Format Version'] = 'v0.8.3'

    update_meta_info(rec_dir, meta_info)


def update_recording_v083_to_v086(rec_dir):
    logger.info("Updating recording from v0.8.3 format to v0.8.6 format")
    pupil_data = load_object(os.path.join(rec_dir, "pupil_data"))
    meta_info_path = os.path.join(rec_dir,"info.csv")

    for topic in pupil_data.keys():
        for d in pupil_data[topic]:
            d['topic'] = topic

    save_object(pupil_data,os.path.join(rec_dir, "pupil_data"))

    with open(meta_info_path,'r',encoding='utf-8') as csvfile:
        meta_info = csv_utils.read_key_value_file(csvfile)
        meta_info['Data Format Version'] = 'v0.8.6'

    update_meta_info(rec_dir, meta_info)


def update_recording_v086_to_v087(rec_dir):
    logger.info("Updating recording from v0.8.6 format to v0.8.7 format")
    pupil_data = load_object(os.path.join(rec_dir, "pupil_data"))
    meta_info_path = os.path.join(rec_dir,"info.csv")

    def _clamp_norm_point(pos):
        '''realisitic numbers for norm pos should be in this range.
            Grossly bigger or smaller numbers are results bad exrapolation
            and can cause overflow erorr when denormalized and cast as int32.
        '''
        return min(100.,max(-100.,pos[0])),min(100.,max(-100.,pos[1]))

    for g in pupil_data.get('gaze_positions', []):
        if 'topic' not in g:
            # we missed this in one gaze mapper
            g['topic'] = 'gaze'
        g['norm_pos'] = _clamp_norm_point(g['norm_pos'])

    save_object(pupil_data,os.path.join(rec_dir, "pupil_data"))

    with open(meta_info_path,'r',encoding='utf-8') as csvfile:
        meta_info = csv_utils.read_key_value_file(csvfile)
        meta_info['Data Format Version'] = 'v0.8.7'

    update_meta_info(rec_dir, meta_info)


def update_recording_v087_to_v091(rec_dir):
    logger.info("Updating recording from v0.8.7 format to v0.9.1 format")
    meta_info_path = os.path.join(rec_dir,"info.csv")

    with open(meta_info_path,'r',encoding='utf-8') as csvfile:
        meta_info = csv_utils.read_key_value_file(csvfile)
        meta_info['Data Format Version'] = 'v0.9.1'

    update_meta_info(rec_dir, meta_info)


def update_recording_v091_to_v093(rec_dir):
    logger.info("Updating recording from v0.9.1 format to v0.9.3 format")
    meta_info_path = os.path.join(rec_dir,"info.csv")
    pupil_data = load_object(os.path.join(rec_dir, "pupil_data"))
    for g in pupil_data.get('gaze_positions', []):
        # fixing recordings made with bug https://github.com/pupil-labs/pupil/issues/598
        g['norm_pos'] = float(g['norm_pos'][0]), float(g['norm_pos'][1])

    save_object(pupil_data, os.path.join(rec_dir, "pupil_data"))

    with open(meta_info_path, 'r', encoding='utf-8') as csvfile:
        meta_info = csv_utils.read_key_value_file(csvfile)
        meta_info['Data Format Version'] = 'v0.9.3'
    update_meta_info(rec_dir, meta_info)


def update_recording_v093_to_v094(rec_dir):
    logger.info("Updating recording from v0.9.3 to v0.9.4.")
    meta_info_path = os.path.join(rec_dir, "info.csv")

    for file in os.listdir(rec_dir):
        if file.startswith('.') or os.path.splitext(file)[1] in ('.mp4', '.avi'):
            continue
        rec_file = os.path.join(rec_dir, file)

        try:
            rec_object = load_object(rec_file,allow_legacy=False)
            save_object(rec_object, rec_file)
        except:
            try:
                rec_object = load_object(rec_file,allow_legacy=True)
                save_object(rec_object, rec_file)
                logger.info('Converted `{}` from pickle to msgpack'.format(file))
            except:
                logger.warning("did not convert {}".format(rec_file))

    with open(meta_info_path, 'r', encoding='utf-8') as csvfile:
        meta_info = csv_utils.read_key_value_file(csvfile)
        meta_info['Data Format Version'] = 'v0.9.4'
    update_meta_info(rec_dir, meta_info)


def update_recording_v094_to_v0913(rec_dir, retry_on_averror=True):
    try:
        logger.info("Updating recording from v0.9.4 to v0.9.13")
        meta_info_path = os.path.join(rec_dir, "info.csv")

        wav_file_loc = os.path.join(rec_dir, 'audio.wav')
        aac_file_loc = os.path.join(rec_dir, 'audio.mp4')
        audio_ts_loc = os.path.join(rec_dir, 'audio_timestamps.npy')
        backup_ts_loc = os.path.join(rec_dir, 'audio_timestamps_old.npy')
        if os.path.exists(wav_file_loc) and os.path.exists(audio_ts_loc):
            in_container = av.open(wav_file_loc)
            in_stream = in_container.streams.audio[0]
            in_frame_size = 0
            in_frame_num = 0

            out_container = av.open(aac_file_loc, 'w')
            out_stream = out_container.add_stream('aac')

            for in_packet in in_container.demux():
                for audio_frame in in_packet.decode():
                    if not in_frame_size:
                        in_frame_size = audio_frame.samples
                    in_frame_num += 1
                    out_packet = out_stream.encode(audio_frame)
                    if out_packet is not None:
                        out_container.mux(out_packet)

            # flush encoder
            out_packet = out_stream.encode(None)
            while out_packet is not None:
                out_container.mux(out_packet)
                out_packet = out_stream.encode(None)

            out_frame_size = out_stream.frame_size
            out_frame_num = out_stream.frames
            out_frame_rate = out_stream.rate
            in_frame_rate = in_stream.rate

            out_container.close()

            old_ts = np.load(audio_ts_loc)
            np.save(backup_ts_loc, old_ts)

            if len(old_ts) != in_frame_num:
                in_frame_size /= len(old_ts) / in_frame_num
                logger.debug('Provided audio frame size is inconsistent with amount of timestamps. Correcting frame size to {}'.format(in_frame_size))

            old_ts_idx = np.arange(0, len(old_ts) * in_frame_size, in_frame_size) * out_frame_rate / in_frame_rate
            new_ts_idx = np.arange(0, out_frame_num * out_frame_size, out_frame_size)
            interpolate = interp1d(old_ts_idx, old_ts, bounds_error=False, fill_value='extrapolate')
            new_ts = interpolate(new_ts_idx)

            # raise RuntimeError
            np.save(audio_ts_loc, new_ts)

        with open(meta_info_path, 'r', encoding='utf-8') as csvfile:
            meta_info = csv_utils.read_key_value_file(csvfile)
            meta_info['Data Format Version'] = 'v0.9.13'
        update_meta_info(rec_dir, meta_info)
    except av.AVError as averr:
        # Try to catch `libav.aac : Input contains (near) NaN/+-Inf` errors
        # Unfortunately, the above error is only logged not raised. Instead
        # `averr`, an `Invalid Argument` error with error number 22, is raised.
        if retry_on_averror and averr.errno == 22:
            # unfortunately
            logger.error('Encountered AVError. Retrying to update recording.')
            out_container.close()
            # Only retry once:
            update_recording_v094_to_v0913(rec_dir, retry_on_averror=False)
        else:
            raise  # re-raise exception


def update_recording_bytes_to_unicode(rec_dir):
    logger.info("Updating recording from bytes to unicode.")

    def convert(data):
        if isinstance(data, bytes):
            return data.decode()
        elif isinstance(data, str) or isinstance(data, np.ndarray):
            return data
        elif isinstance(data, collections.Mapping):
            return dict(map(convert, data.items()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(convert, data))
        else:
            return data

    for file in os.listdir(rec_dir):
        if file.startswith('.') or os.path.splitext(file)[1] in ('.mp4', '.avi'):
            continue
        rec_file = os.path.join(rec_dir, file)
        try:
            rec_object = load_object(rec_file)
            converted_object = convert(rec_object)
            if converted_object != rec_object:
                logger.info('Converted `{}` from bytes to unicode'.format(file))
                save_object(converted_object, rec_file)
        except (UnpicklingError, IsADirectoryError):
            continue

    # manually convert k v dicts.
    meta_info_path = os.path.join(rec_dir, "info.csv")
    with open(meta_info_path, 'r', encoding='utf-8') as csvfile:
        meta_info = csv_utils.read_key_value_file(csvfile)
    with open(meta_info_path, 'w', newline='') as csvfile:
        csv_utils.write_key_value_file(csvfile, meta_info)


def update_recording_v073_to_v074(rec_dir):
    logger.info("Updating recording from v0.7x format to v0.7.4 format")
    pupil_data = load_object(os.path.join(rec_dir, "pupil_data"))
    modified = False
    for p in pupil_data['pupil_positions']:
        if p['method'] == "3D c++":
            p['method'] = "3d c++"
            try:
                p['projected_sphere'] = p.pop('projectedSphere')
            except:
                p['projected_sphere'] = {'center':(0,0),'angle':0,'axes':(0,0)}
            p['model_confidence'] = p.pop('modelConfidence')
            p['model_id'] = p.pop('modelID')
            p['circle_3d'] = p.pop('circle3D')
            p['diameter_3d'] = p.pop('diameter_3D')
            modified = True
    if modified:
        save_object(load_object(os.path.join(rec_dir, "pupil_data")),os.path.join(rec_dir, "pupil_data_old"))
    try:
        save_object(pupil_data,os.path.join(rec_dir, "pupil_data"))
    except IOError:
        pass


def update_recording_v05_to_v074(rec_dir):
    logger.info("Updating recording from v0.5x/v0.6x/v0.7x format to v0.7.4 format")
    pupil_data = load_object(os.path.join(rec_dir, "pupil_data"))
    save_object(pupil_data,os.path.join(rec_dir, "pupil_data_old"))
    for p in pupil_data['pupil_positions']:
        p['method'] = '2d python'
    try:
        save_object(pupil_data,os.path.join(rec_dir, "pupil_data"))
    except IOError:
        pass

def update_recording_v04_to_v074(rec_dir):
    logger.info("Updating recording from v0.4x format to v0.7.4 format")
    gaze_array = np.load(os.path.join(rec_dir,'gaze_positions.npy'))
    pupil_array = np.load(os.path.join(rec_dir,'pupil_positions.npy'))
    gaze_list = []
    pupil_list = []

    for datum in pupil_array:
        ts, confidence, id, x, y, diameter = datum[:6]
        pupil_list.append({'timestamp':ts,'confidence':confidence,'id':id,'norm_pos':[x,y],'diameter':diameter,
                          'method':'2d python','ellipse': {'angle': 0.0, 'center': [0.0, 0.0], 'axes': [0.0, 0.0]}})

    pupil_by_ts = dict([(p['timestamp'],p) for p in pupil_list])

    for datum in gaze_array:
        ts,confidence,x,y, = datum
        gaze_list.append({'timestamp':ts,'confidence':confidence,'norm_pos':[x,y],'base':[pupil_by_ts.get(ts,None)]})

    pupil_data = {'pupil_positions':pupil_list,'gaze_positions':gaze_list}
    try:
        save_object(pupil_data,os.path.join(rec_dir, "pupil_data"))
    except IOError:
        pass

def update_recording_v03_to_v074(rec_dir):
    logger.info("Updating recording from v0.3x format to v0.7.4 format")
    pupilgaze_array = np.load(os.path.join(rec_dir,'gaze_positions.npy'))
    gaze_list = []
    pupil_list = []

    for datum in pupilgaze_array:
        gaze_x,gaze_y,pupil_x,pupil_y,ts,confidence = datum
        #some bogus size and confidence as we did not save it back then
        pupil_list.append({'timestamp':ts,'confidence':confidence,'id':0,'norm_pos':[pupil_x,pupil_y],'diameter':50,'method':'2d python'})
        gaze_list.append({'timestamp':ts,'confidence':confidence,'norm_pos':[gaze_x,gaze_y],'base':[pupil_list[-1]]})

    pupil_data = {'pupil_positions':pupil_list,'gaze_positions':gaze_list}
    try:
        save_object(pupil_data,os.path.join(rec_dir, "pupil_data"))
    except IOError:
        pass

    ts_path     = os.path.join(rec_dir,"world_timestamps.npy")
    ts_path_old = os.path.join(rec_dir,"timestamps.npy")
    if not os.path.isfile(ts_path) and os.path.isfile(ts_path_old):
        os.rename(ts_path_old, ts_path)


def is_pupil_rec_dir(rec_dir):
    if not os.path.isdir(rec_dir):
        logger.error("No valid dir supplied")
        return False
    try:
        meta_info = load_meta_info(rec_dir)
        meta_info["Recording Name"]  # Test key existence
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
        cv2.circle(img,center,radius,rgb, thickness=thickness, lineType=cv2.LINE_AA)
        opacity = alpha
        cv2.addWeighted(src1=img[roi], alpha=opacity, src2=overlay, beta=1. - opacity, gamma=0, dst=img[roi])
    except:
        logger.debug("transparent_circle would have been partially outside of img. Did not draw it.")


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
