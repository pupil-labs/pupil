'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os, sys
import cv2
from pyglui import ui
import numpy as np
from scipy.interpolate import UnivariateSpline
from plugin import Plugin
from time import strftime,localtime,time,gmtime
from shutil import copy2
from glob import glob
from audio import Audio_Capture
#logging
import logging
logger = logging.getLogger(__name__)


def get_auto_name():
    return strftime("%Y_%m_%d", localtime())

def sanitize_timestamps(ts):
    logger.debug("Checking %s timestamps for monotony in direction and smoothness"%ts.shape[0])
    avg_frame_time = (ts[-1] - ts[0])/ts.shape[0]
    logger.debug('average_frame_time: %s'%(1./avg_frame_time))

    raw_ts = ts #only needed for visualization
    runs = 0
    while True:
        #forward check for non monotonic increasing behaviour
        clean = np.ones((ts.shape[0]),dtype=np.bool)
        damper  = 0
        for idx in range(ts.shape[0]-1):
            if ts[idx] >= ts[idx+1]: #not monotonically increasing timestamp
                damper = 50
            clean[idx] = damper <= 0
            damper -=1

        #backward check to smooth timejumps forward
        damper  = 0
        for idx in range(ts.shape[0]-1)[::-1]:
            if ts[idx+1]-ts[idx]>1: #more than one second forward jump
                damper = 50
            clean[idx] &= damper <= 0
            damper -=1

        if clean.all() == True:
            if runs >0:
                logger.debug("Timestamps were bad but are ok now. Correction runs: %s"%runs)
                # from matplotlib import pyplot as plt
                # plt.plot(frames,raw_ts)
                # plt.plot(frames,ts)
                # # plt.scatter(frames[~clean],ts[~clean])
                # plt.show()
            else:
                logger.debug("Timestamps are clean.")
            return ts

        runs +=1
        if runs > 4:
            logger.error("Timestamps could not be fixed!")
            return ts

        logger.warning("Timestamps are not sane. We detected non monotitc or jumpy timestamps. Fixing them now")
        frames = np.arange(len(ts))
        s = UnivariateSpline(frames[clean],ts[clean],s=0)
        ts = s(frames)



class Recorder(Plugin):
    """Capture Recorder"""
    def __init__(self,g_pool,session_name = get_auto_name(), record_eye = False, audio_src = -1, menu_conf = {}):
        super(Recorder, self).__init__(g_pool)
        self.order = .9
        self.record_eye = record_eye
        self.session_name = session_name
        self.audio_src = audio_src
        self.running = False
        self.menu = None
        self.button = None
        self.menu_conf = menu_conf


    def get_init_dict(self):
        d = {}
        d['record_eye'] = self.record_eye
        d['audio_src'] = self.audio_src
        if self.menu:
            d['menu_conf'] = self.menu.configuration
        else:
            d['menu_conf'] = self.menu_conf
        return d


    def init_gui(self):
        self.menu = ui.Growing_Menu('Recorder')
        self.menu.configuration = self.menu_conf
        self.g_pool.sidebar.append(self.menu)
        self.menu.append(ui.Info_Text('This is the recorder info text. It should explain some non obvious settings.'))
        self.menu.append(ui.TextInput('rec_dir',self.g_pool,setter=self.set_rec_dir,label='Recording Path'))
        self.menu.append(ui.TextInput('session_name',self,setter=self.set_session_name,label='Session'))
        self.menu.append(ui.Switch('record_eye',self,on_val=True,off_val=False,label='Record Eye'))

        self.button = ui.Thumb('running',self,setter=self.toggle,label='Record',hotkey='r')
        self.button.on_color[:] = (1,.0,.0,.8)
        self.g_pool.quickbar.append(self.button)


    def deinit_gui(self):
        if self.menu:
            self.menu_conf = self.menu.configuration
            self.g_pool.sidebar.remove(self.menu)

            self.menu = None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None

    def toggle(self,val):
        if val:
            self.start()
        else:
            self.stop()

    def get_rec_time_str(self):
        rec_time = gmtime(time()-self.start_time)
        return strftime("%H:%M:%S", rec_time)

    def start(self):
        self.timestamps = []
        self.pupil_list = []
        self.gaze_list = []
        self.frame_count = 0
        self.running = True
        self.menu.read_only = True
        self.start_time = time()

        session = os.path.join(self.g_pool.rec_dir, self.session_name)
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
            f.write("Recording Name\t"+self.session_name+ "\n")
            f.write("Start Date\t"+ strftime("%d.%m.%Y", localtime(self.start_time))+ "\n")
            f.write("Start Time\t"+ strftime("%H:%M:%S", localtime(self.start_time))+ "\n")


        if self.audio_src >=0:
            audio_path = os.path.join(self.rec_path, "world.wav")
            self.audio_writer = Audio_Capture(self.audio_src,audio_path)
        else:
            self.audio_writer = None

        self.video_path = os.path.join(self.rec_path, "world.mkv")
        self.writer = None
        # positions path to eye process
        if self.record_eye:
            self.g_pool.eye_tx.send(self.rec_path)

    def update(self,frame,events):
        if self.running:
            if not self.writer:
                self.writer = cv2.VideoWriter(self.video_path, cv2.cv.CV_FOURCC(*'DIVX'), float(self.g_pool.capture.frame_rate), (frame.width,frame.height))
                self.height, self.width = frame.height,frame.width

            # cv2.putText(frame.img, "Frame %s"%self.frame_count,(200,200), cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,100))
            for p in events['pupil_positions']:
                pupil_pos = p['norm_pos'][0],p['norm_pos'][1],p['diameter'],p['timestamp'],p['confidence'],p['id']

                self.pupil_list.append(pupil_pos)

            for g in events.get('gaze',[]):
                gaze_pos = g['norm_pos'][0],g['norm_pos'][1],g['confidence'],g['timestamp']
                self.gaze_list.append(gaze_pos)

            self.timestamps.append(frame.timestamp)
            self.writer.write(frame.img)
            self.frame_count += 1

            self.button.status_text = self.get_rec_time_str()

    def stop(self):
        #explicit release of VideoWriter
        self.writer.release()
        self.writer = None

        if self.record_eye:
            try:
                self.g_pool.eye_tx.send(None)
            except:
                logger.warning("Could not stop eye-recording. Please report this bug!")

        gaze_list_path = os.path.join(self.rec_path, "gaze_positions.npy")
        np.save(gaze_list_path,np.asarray(self.gaze_list))

        pupil_list_path = os.path.join(self.rec_path, "pupil_positions.npy")
        np.save(pupil_list_path,np.asarray(self.pupil_list))

        timestamps_path = os.path.join(self.rec_path, "world_timestamps.npy")
        ts = sanitize_timestamps(np.array(self.timestamps))
        np.save(timestamps_path,ts)

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
            logger.exception("Could not save metadata. Please report this bug!")

        if self.audio_writer:
            self.audio_writer = None

        self.running = False
        self.menu.read_only = False
        self.button.status_text = ''



    def cleanup(self):
        """gets called when the plugin get terminated.
           either volunatily or forced.
        """
        if self.running:
            self.stop()
        self.deinit_gui()


    def set_rec_dir(self,val):
        try:
            n_path = os.path.expanduser(val)
            logger.debug("Expanded user path.")
        except:
            n_path = val

        if not n_path:
            logger.warning("Please specify a path.")
        elif not os.path.isdir(n_path):
            logger.warning("This is not a valid path.")
        else:
            self.g_pool.rec_dir = n_path

    def set_session_name(self, val):
        if not val:
            self.session_name = get_auto_name()
        else:
            self.session_name = val




