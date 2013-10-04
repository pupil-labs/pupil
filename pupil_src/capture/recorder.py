import os, sys
import cv2
import atb
import numpy as np
from plugin import Plugin
from time import strftime,localtime,time,gmtime
from ctypes import create_string_buffer

class Recorder(Plugin):
    """Capture Recorder"""
    def __init__(self,g_pool, session_str, fps, img_shape, record_eye, eye_tx):
        Plugin.__init__(self)
        self.g_pool = g_pool
        self.session_str = session_str
        self.record_eye = record_eye
        self.frame_count = 0
        self.timestamps = []
        self.gaze_list = []
        self.eye_tx = eye_tx
        self.start_time = time()
        # set up base folder called "recordings"


        if getattr(sys, 'frozen', False):
            # we are running in a |PyInstaller| bundle
            self.base_path = os.path.join(sys._MEIPASS.rsplit(os.path.sep,1)[0],"recordings")
        else:
            # we are running in a normal Python environment
            self.base_path = os.path.join(os.path.abspath(__file__).rsplit('pupil_src', 1)[0], "recordings")



        try:
            os.mkdir(self.base_path)
        except:
            print "recordings folder already exists, using existing."

        session = os.path.join(self.base_path, self.session_str)
        try:
            os.mkdir(session)
        except:
            print "recordings session folder already exists, using existing."

        # set up self incrementing folder within session folder
        counter = 0
        while True:
            self.path = os.path.join(self.base_path, session, "%03d/" % counter)
            try:
                os.mkdir(self.path)
                break
            except:
                print "We dont want to overwrite data, incrementing counter & trying to make new data folder"
                counter += 1

        self.meta_info_path = os.path.join(self.path, "info.csv")

        with open(self.meta_info_path, 'w') as f:
            f.write("Pupil Recording Name:\t"+self.session_str+ "\n")
            f.write("Start Date: \t"+ strftime("%d.%m.%Y", localtime(self.start_time))+ "\n")
            f.write("Start Time: \t"+ strftime("%H:%M:%S", localtime(self.start_time))+ "\n")



        video_path = os.path.join(self.path, "world.avi")
        self.writer = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), fps, (img_shape[1], img_shape[0]))
        self.height = img_shape[0]
        self.width = img_shape[1]
        # positions path to eye process
        if self.record_eye:
            self.eye_tx.send(self.path)

        atb_pos = (10, 540)
        self._bar = atb.Bar(name = self.__class__.__name__, label='REC: '+session_str,
            help="capture recording control", color=(220, 0, 0), alpha=150,
            text='light', position=atb_pos,refresh=.3, size=(300, 80))
        self._bar.rec_name = create_string_buffer(512)
        self._bar.add_var("rec time",self._bar.rec_name, getter=lambda: create_string_buffer(self.get_rec_time_str(),512), readonly=True)
        self._bar.add_button("stop",self.on_stop, key="s", help="stop recording")
        self._bar.define("contained=true")

    def get_rec_time_str(self):
        rec_time = gmtime(time()-self.start_time)
        return strftime("%H:%M:%S", rec_time)

    def update(self, frame,recent_pupil_positons):
        self.frame_count += 1
        for p in recent_pupil_positons:
            if p['norm_pupil'] is not None:
                gaze_pt = p['norm_gaze'][0],p['norm_gaze'][1],p['norm_pupil'][0],p['norm_pupil'][1],p['timestamp']
                self.gaze_list.append(gaze_pt)
        self.timestamps.append(frame.timestamp)
        self.writer.write(frame.img)

    def stop_and_destruct(self):
        if self.record_eye:
            try:
                self.eye_tx.send(None)
            except:
                print "WARNING: Could not stop eye-recording. Please report this bug!"
        gaze_list_path = os.path.join(self.path, "gaze_positions.npy")
        np.save(gaze_list_path,np.asarray(self.gaze_list))

        timestamps_path = os.path.join(self.path, "timestamps.npy")
        np.save(timestamps_path,np.array(self.timestamps))


        try:
            cal_pt_cloud = np.load(os.path.join(self.g_pool.user_dir,"cal_pt_cloud.npy"))
            cal_pt_cloud_path = os.path.join(self.path, "cal_pt_cloud.npy")
            np.save(cal_pt_cloud_path, cal_pt_cloud)
        except:
            print "WARNING: No calibration data found. Please calibrate first."

        try:
            camera_matrix = np.load(os.path.join(self.g_pool.user_dir,"camera_matrix.npy"))
            dist_coefs = np.load(os.path.join(self.g_pool.user_dir,"dist_coefs.npy"))
            cam_path = os.path.join(self.path, "camera_matrix.npy")
            dist_path = os.path.join(self.path, "dist_coefs.npy")
            np.save(cam_path, camera_matrix)
            np.save(dist_path, dist_coefs)
        except:
            print "No camera intrinsics found, will not copy them into recordings folder."


        try:
            with open(self.meta_info_path, 'a') as f:
                f.write("Duration Time: \t"+ self.get_rec_time_str()+ "\n")
                f.write("World Camera Frames: \t"+ str(self.frame_count)+ "\n")
                f.write("World Camera Resolution: \t"+ str(self.width)+"x"+str(self.height)+"\n")
                f.write("Capture Software Version: \t"+ self.g_pool.version + "\n")
                f.write("user:\t"+os.getlogin()+"\n")
                try:
                    sysname, nodename, release, version, machine = os.uname()
                except:
                    sysname, nodename, release, version, machine = sys.platform,None,None,None,None
                f.write("Platform:\t"+sysname+"\n")
                f.write("Machine:\t"+nodename+"\n")
                f.write("Release:\t"+release+"\n")
                f.write("Version:\t"+version+"\n")
        except:
            print "Could not save metadata. Please report this bug!"

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
