'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os, sys
import cv2
import atb
import numpy as np
from plugin import Plugin
from time import strftime,localtime,time,gmtime
from ctypes import create_string_buffer
from shutil import copy2
from glob import glob
from audio import Audio_Capture
#logging
import logging
logger = logging.getLogger(__name__)

class Recorder(Plugin):
    """Capture Recorder"""
    def __init__(self,g_pool, session_str, fps, img_shape, record_eye, eye_tx,audio = -1):
        Plugin.__init__(self)
        self.g_pool = g_pool
        self.session_str = session_str
        self.record_eye = record_eye
        self.frame_count = 0
        self.timestamps = []
        self.gaze_list = []
        self.eye_tx = eye_tx
        self.start_time = time()

        session = os.path.join(self.g_pool.rec_dir, self.session_str)
        try:
            os.mkdir(session)
            logger.debug("Created new recordings session dir %s"%session)

        except:
            logger.debug("Recordings session dir %s already exists, using it." %session)

        # set up self incrementing folder within session folder
        counter = 0
        while True:
            self.rec_path = os.path.join(session, "%03d/" % counter)
            try:
                os.mkdir(self.rec_path)
                logger.debug("Created new recording dir %s"%self.rec_path)
                break
            except:
                logger.debug("We dont want to overwrite data, incrementing counter & trying to make new data folder")
                counter += 1

        self.meta_info_path = os.path.join(self.rec_path, "info.csv")

        with open(self.meta_info_path, 'w') as f:
            f.write("Recording Name\t"+self.session_str+ "\n")
            f.write("Start Date\t"+ strftime("%d.%m.%Y", localtime(self.start_time))+ "\n")
            f.write("Start Time\t"+ strftime("%H:%M:%S", localtime(self.start_time))+ "\n")


        if audio >=0:
            audio_src = audio
            audio_path = os.path.join(self.rec_path, "world.wav")
            self.audio_writer = Audio_Capture(audio_src,audio_path)
        else:
            self.audio_writer = None

        video_path = os.path.join(self.rec_path, "world.avi")
        self.writer = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), fps, (img_shape[1], img_shape[0]))
        self.height = img_shape[0]
        self.width = img_shape[1]
        # positions path to eye process
        if self.record_eye:
            self.eye_tx.send(self.rec_path)

        atb_pos = (10, 540)
        self._bar = atb.Bar(name = self.__class__.__name__, label='REC: '+session_str,
            help="capture recording control", color=(220, 0, 0), alpha=150,
            text='light', position=atb_pos,refresh=.3, size=(300, 80))
        self._bar.add_var("rec time",create_string_buffer(512), getter=lambda: create_string_buffer(self.get_rec_time_str(),512), readonly=True)
        self._bar.add_button("stop",self.on_stop, key="s", help="stop recording")
        self._bar.define("contained=true")

    def get_rec_time_str(self):
        rec_time = gmtime(time()-self.start_time)
        return strftime("%H:%M:%S", rec_time)

    def update(self,frame,recent_pupil_positons,events):
        # cv2.putText(frame.img, "Frame %s"%self.frame_count,(200,200), cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,100))
        for p in recent_pupil_positons:
            if p['norm_pupil'] is not None:
                gaze_pt = p['norm_gaze'][0],p['norm_gaze'][1],p['norm_pupil'][0],p['norm_pupil'][1],p['timestamp'],p['confidence']
                self.gaze_list.append(gaze_pt)
        self.timestamps.append(frame.timestamp)
        self.writer.write(frame.img)
        self.frame_count += 1


    def stop_and_destruct(self):
        #explicit release of VideoWriter
        self.writer.release()
        self.writer = None

        if self.record_eye:
            try:
                self.eye_tx.send(None)
            except:
                logger.warning("Could not stop eye-recording. Please report this bug!")
                
        gaze_list_path = os.path.join(self.rec_path, "gaze_positions.npy")
        np.save(gaze_list_path,np.asarray(self.gaze_list))

        timestamps_path = os.path.join(self.rec_path, "timestamps.npy")
        np.save(timestamps_path,np.array(self.timestamps))

        try:
            copy2(os.path.join(self.g_pool.user_dir,"surface_definitions"),os.path.join(self.rec_path,"surface_definitions"))
        except:
            logger.info("No surface_definitions data found. You may want this if you do marker tracking.")

        try:
            copy2(os.path.join(self.g_pool.user_dir,"cal_pt_cloud.npy"),os.path.join(self.rec_path,"cal_pt_cloud.npy"))
        except:
            logger.warning("No calibration data found. Please calibrate first.")

        try:
            copy2(os.path.join(self.g_pool.user_dir,"camera_matrix.npy"),os.path.join(self.rec_path,"camera_matrix.npy"))
            copy2(os.path.join(self.g_pool.user_dir,"dist_coefs.npy"),os.path.join(self.rec_path,"dist_coefs.npy"))
        except:
            logger.info("No camera intrinsics found.")


        try:
            with open(self.meta_info_path, 'a') as f:
                f.write("Duration Time\t"+ self.get_rec_time_str()+ "\n")
                f.write("World Camera Frames\t"+ str(self.frame_count)+ "\n")
                f.write("World Camera Resolution\t"+ str(self.width)+"x"+str(self.height)+"\n")
                f.write("Capture Software Version\t"+ self.g_pool.version + "\n")
                f.write("User\t"+os.getlogin()+"\n")
                try:
                    sysname, nodename, release, version, machine = os.uname()
                except:
                    sysname, nodename, release, version, machine = sys.platform,None,None,None,None
                f.write("Platform\t"+sysname+"\n")
                f.write("Machine\t"+nodename+"\n")
                f.write("Release\t"+release+"\n")
                f.write("Version\t"+version+"\n")
        except Exception:
            logger.Exception("Could not save metadata. Please report this bug!")

        if self.audio_writer:
            self.audio_writer = None

        self.alive = False


    def on_stop(self):
        """
        get called from _bar to init termination.
        """
        self.alive= False


    def cleanup(self):
        """gets called when the plugin get terminated.
           either volunatily or forced.
        """
        self.stop_and_destruct()
        self._bar.destroy()



def get_auto_name():
    return strftime("%Y_%m_%d", localtime())
