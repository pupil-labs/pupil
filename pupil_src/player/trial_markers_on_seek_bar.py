'''
(*)~----------------------------------------------------------------------------------

Pupil Player Plugin

Author: Carlos Picanco.
Hacked from player/seek_bar.py from Pupil - eye tracking platform (v0.3.7.4)

Distributed under the terms of the CC BY-NC-SA License.
License details are in the file license.txt, distributed as part of this software.
    
----------------------------------------------------------------------------------~(*)
'''


from gl_utils import draw_gl_polyline,draw_gl_point
from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D

# from ctypes import c_int,c_float,c_bool

from glfw import glfwGetWindowSize, glfwGetCurrentContext
from plugin import Plugin
from uvc_capture import autoCreateCapture
from player_methods import correlate_gaze

from methods import denormalize

import logging
import platform
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# plugins
from vis_circle_on_contours import Circle_on_Contours
from vis_circle import Vis_Circle
from vis_polyline import Vis_Polyline
from vis_light_points import Vis_Light_Points
from scan_path import Scan_Path

plugin_by_index =  (    Vis_Circle,
                        Vis_Polyline,
                        Scan_Path,
                        Vis_Light_Points,
                        Circle_on_Contours)

name_by_index = [p.__name__ for p in plugin_by_index]
index_by_name = dict(zip(name_by_index,range(len(name_by_index))))
plugin_by_name = dict(zip(name_by_index,plugin_by_index))
additive_plugins = (Vis_Circle,Vis_Polyline, Circle_on_Contours)


class Trial_Markers_on_Seek_Bar(Plugin):
    """
    Draw markers in the seek bar based on trial (a_code) events.
    Trial format: temp = [0, 1, 2] = (trial_index, timestamp, a_code).   
    Export metadata 
    """
    def __init__(self,
                 g_pool = None):
        super(Trial_Markers_on_Seek_Bar, self).__init__()
 
        # testing........................................................................................................
        # print self.cap.timestamps
        # print np.array(self.cap.timestamps)
        self.idx_begin_trial = []
        self.idx_end_limited_hold = []
        self.idx_first_response = []
        self.pos_begin_trial = []
        self.pos_first_response = []
        self.pos_end_limited_hold = []
        self.pos_else = []

        self.g_pool = g_pool
        self.order = .5
        # self.cap = capture

        #display layout
        self.padding = 20.
        logger.info('Circle_on_Contours plugin initialization on ' + platform.system())

    def init_gui(self):
        import atb
        atb_pos = 320, 115    
        
        # Creating an AntTweakBar.
        atb_label = "Trial Markers on Seek_Bar"
        self._bar = atb.Bar(
            name = self.__class__.__name__,
            label = atb_label,
            help = "draw trial markers on seek bar",
            color = (50, 50, 50),
            alpha = 100,
            text = 'light',
            position = atb_pos,
            refresh = .3,
            size = (300, 100))

        # Exporter
        self._bar.add_button('Export Metadata',self.export_metadata)
        self._bar.add_separator('s0')

        # Window parameters
        self._bar.add_button('remove',self.unset_alive)


    def export_metadata(self):
        print 'Exporting metadata...'
        from os import sep, path

        # choose a filename for the output metadata
        data_path = self.current_path + sep
        n = 1
        metadata_file_path = data_path + 'metadata' + str(n)
        while True:
            if path.isfile(metadata_file_path):
                n += 1
                metadata_file_path = data_path + 'metadata' + str(n)
            else:
                metadata = open(metadata_file_path, 'w+')
                break
        logger.debug("metadata will be save to: %s" %metadata_file_path)

        # load data 
        gaze_list = np.load(data_path + "gaze_positions.npy")
        timestamps = np.load(data_path + 'timestamps.npy')
        
        # correlate data
        positions_by_frame = correlate_gaze(gaze_list,timestamps)

        # load video file
        cap = autoCreateCapture(src = data_path + 'world.avi',
                                # size=(640, 480),
                                # fps=24
                                timestamps = data_path + 'timestamps.npy')

        # writer setup
        width,height = cap.get_size()
        writer = cv2.VideoWriter(data_path + 'last_world_viz.avi',
                                 cv2.cv.CV_FOURCC(*'DIVX'), 24, (width, height))
        
        # load initialized plugins:
        initialized_plugins = []
        for p in self.g_pool.plugins:
            try:
                p_initializer = p.get_class_name(),p.get_init_dict()
                initialized_plugins.append(p_initializer)
            except AttributeError:
                pass
        
        plugins = []
        for initialized in initialized_plugins:
            name, args = initialized
            logger.debug("Loading plugin: %s with settings %s"%(name, args))
            try:
                p = plugin_by_name[name](**args)
                plugins.append(p)
            except:
                logger.warning("Plugin '%s' failed to load." %name)

        # create an auto trim reference - trim_sections - based on trial events
        # total trials == len(begin_frames) == len(end_frames)
        # zip(begin_frames, end_frames)
        trim_sections = zip(self.idx_first_response, self.idx_end_limited_hold)

        # metadata output file structure, we use tabs to split columns and \n to lines
        new_line = 'idxframe' + '\t' + 'idxtrial' + '\t' + 'contours' + '\t' + 'cntcount' + '\t' + 'cntellps' + '\t' + 'elpcount'  '\t' + 'pupilsxy' + '\t' + 'pxycount' + '\t' + 'pxytests' + '\t' + 'elp_alfa' + '\n'
        
        # write header strings to file
        metadata.write(new_line)

        # create containers for each variables

        # frame index
        idxframe = 0

        # trial index
        idxtrial = 0

        # list of detected contours (findcontours -> fitellipse -> ellipse2poly*)
        contours = []

        # number of detected contours 
        cntcount = 0

        # list of fitted ellipses (findcontours -> fitellipse* -> ellipse2poly)  
        cntellps = []

        # number of fitted ellipses
        elpcount = 0

        # list of xy coordenate of pupil positions (normalized -> denormalized*)
        pupilsxy = []

        # number of pupils
        pxycount = 0

        # list of point polygon test
        pxytests = []

        # axis alpha used for control ellipses size
        elp_alfa = 0

        # begin the trim process
        for trim_section in trim_sections:
            # set start-end frames
            start_frame = trim_section[0]
            end_frame = trim_section[1]

            # seek to start frame
            cap.seek_to_frame(start_frame)

            # begin the record process
            for x in xrange(start_frame, end_frame):
                # get current frame
                new_frame = cap.get_frame()

                # video logic ending: pause at last frame.
                if not new_frame:
                    logger.error("Could not read all the frames.")
                    #explicit release of VideoWriter
                    writer.release()
                    writer = None
                    return False
                else:
                    frame = new_frame

                idxframe = frame.index

                #new positons and events
                current_pupil_positions = positions_by_frame[frame.index]
                events = None

                # allow each Plugin to do its work.
                for p in plugins:
                    p.update(frame,current_pupil_positions,events)
                    if hasattr(p,'get_variables'):
                        contours, cntcount, cntellps, elpcount, pupilsxy, pxycount, pxytests, elp_alfa = p.get_var


                frame.img 
                writer.write(frame.img)
                new_line = str(idxframe) + '\t' + str(idxtrial) + '\t' + str(contours) + '\t' + str(cntcount) + '\t' + str(cntellps) + '\t' + str(elpcount) + '\t' + str(pupilsxy) + '\t' + str(pxycount) + '\t' + str(pxytests) + '\t' + str(elp_alfa)
                metadata.write(new_line)
        
        writer.release()
        writer = None
        metadata.close;
        logger.info("Dear machine, thank you for your collaboration.")


    def unset_alive(self):
        self.alive = False

    def trial_from_frame(self, frame):
        timestamp = self.timestamps[frame]


    def set_var(self, timestamps, current_path):
        from os import sep
        from ast import literal_eval

        self.timestamps = timestamps
        self.frame_count = len(self.timestamps)
        self.current_path = current_path

        timestamps_by_trial_path = current_path + sep + 'timestamps'
        timestamps_by_trial = [[]]
        with open(timestamps_by_trial_path) as f:
            for line in f:
                temp = literal_eval(line)

                # legend
                # temp [0, 1, 2] == (trial_index, timestamp, a_code)
                i = int(temp[0])
                timestamp = (temp [1], temp[2])

                if i > len(timestamps_by_trial):
                    timestamps_by_trial.append([])
                timestamps_by_trial[i - 1].append(timestamp)
            f.close
        self.timestamps_by_trial = timestamps_by_trial

        # Add Main Events

        # frame_index = np.abs(np.array(self.timestamps) - float(timestamp)).argmin()
        # timestamp = self.timestamps[frame_index]
        # frame_index = self.timestamps.index(timestamp)

        for x in timestamps_by_trial:
            # begin trial/show marker
            timestamp = x[0][0]
            frame_index = np.abs(self.timestamps - float(timestamp)).argmin()
            self.idx_begin_trial.append(frame_index)
            self.pos_begin_trial.append(frame_index/float(self.frame_count))

            # first response after marker
            timestamp = x[1][0]
            frame_index = np.abs(self.timestamps - float(timestamp)).argmin()
            self.idx_first_response.append(frame_index)
            self.pos_first_response.append(frame_index/float(self.frame_count))

            # consequence is end_limited_hold
            if 'EndBloc' in x[-1]:
                timestamp = x[-2][0]
            else:
                timestamp = x[-1][0]             
            frame_index = np.abs(self.timestamps - float(timestamp)).argmin()
            self.idx_end_limited_hold.append(frame_index)
            self.pos_end_limited_hold.append(frame_index/float(self.frame_count))
        
        # Add End Event
        timestamp = self.timestamps_by_trial[-1][-1][0]
        frame_index = np.abs(self.timestamps - float(timestamp)).argmin()
        self.idx_begin_trial.append(frame_index)
        self.pos_begin_trial.append(frame_index/float(self.frame_count))

    def update(self,frame,recent_pupil_positions,events):
        pass
        #self.norm_seek_pos = self.current_frame_index/float(self.frame_count)

    def gl_display(self):

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        width,height = glfwGetWindowSize(glfwGetCurrentContext())
        h_pad = self.padding / width
        v_pad = self.padding / height
        gluOrtho2D(-h_pad, 1 + h_pad, -v_pad, 1 + v_pad) # gl coord convention
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        #Draw...............................................................................
         
        for pos_to_draw in self.pos_begin_trial:   
            draw_gl_point((pos_to_draw, 0), size = 5, color = (.5, .5, .5, .5))
            draw_gl_polyline( [(pos_to_draw,.05),(pos_to_draw,0)], color = (.5, .5, .5, .5))

        for pos_to_draw in self.pos_first_response:
            draw_gl_point((pos_to_draw, 0), size = 5, color = (.0, .0, .5, 1.))    
            draw_gl_polyline( [(pos_to_draw,.025),(pos_to_draw,0)], color = (.0, .0, .7, 1.))

        for pos_to_draw in self.pos_end_limited_hold:
            draw_gl_point((pos_to_draw, 0), size = 5, color = (.5, .0, .0, 1.)) 
            draw_gl_polyline( [(pos_to_draw,.025),(pos_to_draw,0)], color = (.7, .0, .0, 1.))
        
        for x in xrange(len(self.pos_first_response)):
            draw_gl_polyline( [(self.pos_end_limited_hold[x],.025),(self.pos_first_response[x],.025)], color = (1., .5, .0, 1.))    

        #Draw...............................................................................

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def get_init_dict(self):
        return { }

    def clone(self):
        return Trial_Markers_on_Seek_Bar(**self.get_init_dict())

    def cleanup(self):
        """gets called when the plugin get terminated.
        This happends either volunatily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        # if self._window:
        #     self.close_window()
        self._bar.destroy()