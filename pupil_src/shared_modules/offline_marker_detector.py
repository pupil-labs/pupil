'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os,platform
import cv2
import numpy as np
import csv


if platform.system() == 'Darwin':
    from billiard import Process,Queue,forking_enable
    from billiard.sharedctypes import Value
else:
    from multiprocessing import Process, Queue
    forking_enable = lambda x: x #dummy fn
    from multiprocessing.sharedctypes import Value
from ctypes import c_bool


from itertools import chain
from OpenGL.GL import *
from methods import normalize,denormalize
from file_methods import Persistent_Dict,save_object
from cache_list import Cache_List
from glfw import *
from pyglui import ui
from pyglui.cygl.utils import *

from plugin import Plugin
#logging
import logging
logger = logging.getLogger(__name__)

from marker_detector import Marker_Detector
from square_marker_detect import detect_markers_robust, draw_markers,m_marker_to_screen
from offline_reference_surface import Offline_Reference_Surface
from math import sqrt


class Offline_Marker_Detector(Marker_Detector):
    """
    Special version of marker detector for use with videofile source.
    It uses a seperate process to search all frames in the world video file for markers.
     - self.cache is a list containing marker positions for each frame.
     - self.surfaces[i].cache is a list containing surface positions for each frame
    Both caches are build up over time. The marker cache is also session persistent.
    See marker_tracker.py for more info on this marker tracker.
    """

    def __init__(self,g_pool,mode="Show Markers and Frames"):
        super(Offline_Marker_Detector, self).__init__(g_pool)
        self.order = .2


        # all markers that are detected in the most recent frame
        self.markers = []
        # all registered surfaces

        if g_pool.app == 'capture':
           raise Exception('For Player only.')
        #in player we load from the rec_dir: but we have a couple options:
        self.surface_definitions = None
        self.surfaces = None
        self.load_surface_definitions_from_file()

        # ui mode settings
        self.mode = mode
        self.min_marker_perimeter = 20  #if we make this a slider we need to invalidate the cache on change.
        # edit surfaces
        self.edit_surfaces = []


        #check if marker cache is available from last session
        self.persistent_cache = Persistent_Dict(os.path.join(g_pool.rec_dir,'square_marker_cache'))
        self.cache = Cache_List(self.persistent_cache.get('marker_cache',[False for _ in g_pool.timestamps]))
        logger.debug("Loaded marker cache %s / %s frames had been searched before"%(len(self.cache)-self.cache.count(False),len(self.cache)) )
        self.init_marker_cacher()

        #debug vars
        self.show_surface_idx = c_int(0)

        self.img_shape = None
        self.img = None

    def load_surface_definitions_from_file(self):
        self.surface_definitions = Persistent_Dict(os.path.join(self.g_pool.rec_dir,'surface_definitions'))
        if self.surface_definitions.get('offline_square_marker_surfaces',[]) != []:
            logger.debug("Found ref surfaces defined or copied in previous session.")
            self.surfaces = [Offline_Reference_Surface(self.g_pool,saved_definition=d) for d in self.surface_definitions.get('offline_square_marker_surfaces',[]) if isinstance(d,dict)]
        elif self.surface_definitions.get('realtime_square_marker_surfaces',[]) != []:
            logger.debug("Did not find ref surfaces def created or used by the user in player from earlier session. Loading surfaces defined during capture.")
            self.surfaces = [Offline_Reference_Surface(self.g_pool,saved_definition=d) for d in self.surface_definitions.get('realtime_square_marker_surfaces',[]) if isinstance(d,dict)]
        else:
            logger.debug("No surface defs found. Please define using GUI.")
            self.surfaces = []

    def init_gui(self):
        self.menu = ui.Scrolling_Menu('Offline Marker Tracker')
        self.g_pool.gui.append(self.menu)


        self.add_button = ui.Thumb('add_surface',setter=self.add_surface,getter=lambda:False,label='Add Surface',hotkey='a')
        self.g_pool.quickbar.append(self.add_button)
        self.update_gui_markers()

        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu= None
        if self.add_button:
            self.g_pool.quickbar.remove(self.add_button)
            self.add_button = None

    def update_gui_markers(self):
        pass
        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text('The offline marker tracker will look for markers in the entire video. By default it uses surfaces defined in capture. You can change and add more surfaces here.'))
        self.menu.append(ui.Info_Text('Please note: Unlike the real-time marker detector the offline marker detector works with a fixed min_marker_perimeter of 20.'))
        self.menu.append(ui.Selector('mode',self,label='Mode',selection=["Show Markers and Frames","Show marker IDs", "Surface edit mode","Show Heatmaps","Show Metrics"] ))
        self.menu.append(ui.Info_Text('To see heatmap or surface metrics visualizations, click (re)-calculate gaze distributions. Set "X size" and "Y size" for each surface to see heatmap visualizations.'))
        self.menu.append(ui.Button("(Re)-calculate gaze distributions", self.recalculate))
        self.menu.append(ui.Button("Export gaze and surface data", self.save_surface_statsics_to_file))
        self.menu.append(ui.Button("Add surface", lambda:self.add_surface('_')))
        for s in self.surfaces:
            idx = self.surfaces.index(s)
            s_menu = ui.Growing_Menu("Surface %s"%idx)
            s_menu.collapsed=True
            s_menu.append(ui.Text_Input('name',s))
            s_menu.append(ui.Text_Input('x',s.real_world_size,label='X size'))
            s_menu.append(ui.Text_Input('y',s.real_world_size,label='Y size'))
            s_menu.append(ui.Button('Open Debug Window',s.open_close_window))
            #closure to encapsulate idx
            def make_remove_s(i):
                return lambda: self.remove_surface(i)
            remove_s = make_remove_s(idx)
            s_menu.append(ui.Button('remove',remove_s))
            self.menu.append(s_menu)


    def on_notify(self,notification):
        if notification['subject'] == 'gaze_positions_changed':
            logger.info('Gaze postions changed. Recalculating.')
            self.recalculate()

    def on_window_resize(self,window,w,h):
        self.win_size = w,h


    def add_surface(self,_):
        self.surfaces.append(Offline_Reference_Surface(self.g_pool))
        self.update_gui_markers()



    def recalculate(self):

        in_mark = self.g_pool.trim_marks.in_mark
        out_mark = self.g_pool.trim_marks.out_mark
        section = slice(in_mark,out_mark)

        # calc heatmaps
        for s in self.surfaces:
            if s.defined:
                s.generate_heatmap(section)

        # calc distirbution accross all surfaces.
        results = []
        for s in self.surfaces:
            gaze_on_srf  = s.gaze_on_srf_in_section(section)
            results.append(len(gaze_on_srf))
            self.metrics_gazecount = len(gaze_on_srf)

        if results == []:
            logger.warning("No surfaces defined.")
            return
        max_res = max(results)
        results = np.array(results,dtype=np.float32)
        if not max_res:
            logger.warning("No gaze on any surface for this section!")
        else:
            results *= 255./max_res
        results = np.uint8(results)
        results_c_maps = cv2.applyColorMap(results, cv2.COLORMAP_JET)

        for s,c_map in zip(self.surfaces,results_c_maps):
            heatmap = np.ones((1,1,4),dtype=np.uint8)*125
            heatmap[:,:,:3] = c_map
            s.metrics_texture = Named_Texture()
            s.metrics_texture.update_from_ndarray(heatmap)


    def update(self,frame,events):
        self.img = frame.img
        self.img_shape = frame.img.shape
        self.update_marker_cache()
        self.markers = self.cache[frame.index]
        if self.markers == False:
            self.markers = []
            self.seek_marker_cacher(frame.index) # tell precacher that it better have every thing from here on analyzed

        # locate surfaces
        for s in self.surfaces:
            if not s.locate_from_cache(frame.index):
                s.locate(self.markers)
            if s.detected:
                pass
                # events.append({'type':'marker_ref_surface','name':s.name,'uid':s.uid,'m_to_screen':s.m_to_screen,'m_from_screen':s.m_from_screen, 'timestamp':frame.timestamp,'gaze_on_srf':s.gaze_on_srf})

        if self.mode == "Show marker IDs":
            draw_markers(frame.img,self.markers)

        # edit surfaces by user
        if self.mode == "Surface edit mode":
            window = glfwGetCurrentContext()
            pos = glfwGetCursorPos(window)
            pos = normalize(pos,glfwGetWindowSize(window),flip_y=True)

            for s,v_idx in self.edit_surfaces:
                if s.detected:
                    new_pos =  s.img_to_ref_surface(np.array(pos))
                    s.move_vertex(v_idx,new_pos)
                    s.cache = None
                    self.heatmap = None
        else:
            # update srf with no or invald cache:
            for s in self.surfaces:
                if s.cache == None:
                    s.init_cache(self.cache)


        #map recent gaze onto detected surfaces used for pupil server
        for s in self.surfaces:
            if s.detected:
                s.gaze_on_srf = []
                for p in events.get('gaze_positions',[]):
                    gp_on_s = tuple(s.img_to_ref_surface(np.array(p['norm_pos'])))
                    p['realtime gaze on ' + s.name] = gp_on_s
                    s.gaze_on_srf.append(gp_on_s)


        #allow surfaces to open/close windows
        for s in self.surfaces:
            if s.window_should_close:
                s.close_window()
            if s.window_should_open:
                s.open_window()


    def init_marker_cacher(self):
        forking_enable(0) #for MacOs only
        from marker_detector_cacher import fill_cache
        visited_list = [False if x == False else True for x in self.cache]
        video_file_path =  self.g_pool.capture.src
        timestamps = self.g_pool.capture.timestamps
        self.cache_queue = Queue()
        self.cacher_seek_idx = Value('i',0)
        self.cacher_run = Value(c_bool,True)
        self.cacher = Process(target=fill_cache, args=(visited_list,video_file_path,timestamps,self.cache_queue,self.cacher_seek_idx,self.cacher_run,self.min_marker_perimeter))
        self.cacher.start()

    def update_marker_cache(self):
        while not self.cache_queue.empty():
            idx,c_m = self.cache_queue.get()
            self.cache.update(idx,c_m)
            for s in self.surfaces:
                s.update_cache(self.cache,idx=idx)

    def seek_marker_cacher(self,idx):
        self.cacher_seek_idx.value = idx

    def close_marker_cacher(self):
        self.update_marker_cache()
        self.cacher_run.value = False
        self.cacher.join()

    def gl_display(self):
        """
        Display marker and surface info inside world screen
        """
        self.gl_display_cache_bars()
        for s in self.surfaces:
            s.gl_display_in_window(self.g_pool.image_tex)

        if self.mode == "Show Markers and Frames":
            for m in self.markers:
                hat = np.array([[[0,0],[0,1],[1,1],[1,0],[0,0]]],dtype=np.float32)
                hat = cv2.perspectiveTransform(hat,m_marker_to_screen(m))
                draw_polyline(hat.reshape((5,2)),color=RGBA(0.1,1.,1.,.3),line_type=GL_POLYGON)
                draw_polyline(hat.reshape((5,2)),color=RGBA(0.1,1.,1.,.6))

            for s in self.surfaces:
                s.gl_draw_frame(self.img_shape)

        if self.mode == "Surface edit mode":
            for s in self.surfaces:
                s.gl_draw_frame(self.img_shape)
                s.gl_draw_corners()

        if self.mode == "Show Heatmaps":
            for s in  self.surfaces:
                s.gl_display_heatmap()
        if self.mode == "Show Metrics":
            #todo: draw a backdrop to represent the gaze that is not on any surface
            for s in self.surfaces:
                #draw a quad on surface with false color of value.
                s.gl_display_metrics()

    def gl_display_cache_bars(self):
        """
        """
        padding = 20.

       # Lines for areas that have been cached
        cached_ranges = []
        for r in self.cache.visited_ranges: # [[0,1],[3,4]]
            cached_ranges += (r[0],0),(r[1],0) #[(0,0),(1,0),(3,0),(4,0)]

        # Lines where surfaces have been found in video
        cached_surfaces = []
        for s in self.surfaces:
            found_at = []
            if s.cache is not None:
                for r in s.cache.positive_ranges: # [[0,1],[3,4]]
                    found_at += (r[0],0),(r[1],0) #[(0,0),(1,0),(3,0),(4,0)]
                cached_surfaces.append(found_at)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        width,height = self.win_size
        h_pad = padding * (self.cache.length-2)/float(width)
        v_pad = padding* 1./(height-2)
        glOrtho(-h_pad,  (self.cache.length-1)+h_pad, -v_pad, 1+v_pad,-1,1) # ranging from 0 to cache_len-1 (horizontal) and 0 to 1 (vertical)


        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        color = RGBA(.8,.6,.2,.8)
        draw_polyline(cached_ranges,color=color,line_type=GL_LINES,thickness=4)

        color = RGBA(0,.7,.3,.8)

        for s in cached_surfaces:
            glTranslatef(0,.02,0)
            draw_polyline(s,color=color,line_type=GL_LINES,thickness=2)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()


    def save_surface_statsics_to_file(self):

        in_mark = self.g_pool.trim_marks.in_mark
        out_mark = self.g_pool.trim_marks.out_mark


        """
        between in and out mark

            report: gaze distribution:
                    - total gazepoints
                    - gaze points on surface x
                    - gaze points not on any surface

            report: surface visisbility

                - total frames
                - surface x visible framecount

            surface events:
                frame_no, ts, surface "name", "id" enter/exit

            for each surface:
                fixations_on_name.csv
                gaze_on_name_id.csv
                positions_of_name_id.csv

        """
        section = slice(in_mark,out_mark)


        metrics_dir = os.path.join(self.g_pool.rec_dir,"metrics_%s-%s"%(in_mark,out_mark))
        logger.info("exporting metrics to %s"%metrics_dir)
        if os.path.isdir(metrics_dir):
            logger.info("Will overwrite previous export for this section")
        else:
            try:
                os.mkdir(metrics_dir)
            except:
                logger.warning("Could not make metrics dir %s"%metrics_dir)
                return


        with open(os.path.join(metrics_dir,'surface_visibility.csv'),'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)

            # surface visibility report
            frame_count = len(self.g_pool.timestamps[section])

            csv_writer.writerow(('frame_count',frame_count))
            csv_writer.writerow((''))
            csv_writer.writerow(('surface_name','visible_frame_count'))
            for s in self.surfaces:
                if s.cache == None:
                    logger.warning("The surface is not cached. Please wait for the cacher to collect data.")
                    return
                visible_count  = s.visible_count_in_section(section)
                csv_writer.writerow( (s.name, visible_count) )
            logger.info("Created 'surface_visibility.csv' file")


        with open(os.path.join(metrics_dir,'surface_gaze_distribution.csv'),'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)

            # gaze distribution report
            gaze_in_section = list(chain(*self.g_pool.gaze_positions_by_frame[section]))
            not_on_any_srf = set([gp['timestamp'] for gp in gaze_in_section])

            csv_writer.writerow(('total_gaze_point_count',len(gaze_in_section)))
            csv_writer.writerow((''))
            csv_writer.writerow(('surface_name','gaze_count'))

            for s in self.surfaces:
                gaze_on_srf  = s.gaze_on_srf_in_section(section)
                gaze_on_srf = set([gp['base']['timestamp'] for gp in gaze_on_srf])
                not_on_any_srf -= gaze_on_srf
                csv_writer.writerow( (s.name, len(gaze_on_srf)) )

            csv_writer.writerow(('not_on_any_surface', len(not_on_any_srf) ) )
            logger.info("Created 'surface_gaze_distribution.csv' file")



        with open(os.path.join(metrics_dir,'surface_events.csv'),'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)

            # surface events report
            csv_writer.writerow(('frame_number','timestamp','surface_name','surface_uid','event_type'))

            events = []
            for s in self.surfaces:
                for enter_frame_id,exit_frame_id in s.cache.positive_ranges:
                    events.append({'frame_id':enter_frame_id,'srf_name':s.name,'srf_uid':s.uid,'event':'enter'})
                    events.append({'frame_id':exit_frame_id,'srf_name':s.name,'srf_uid':s.uid,'event':'exit'})

            events.sort(key=lambda x: x['frame_id'])
            for e in events:
                csv_writer.writerow( ( e['frame_id'],self.g_pool.timestamps[e['frame_id']],e['srf_name'],e['srf_uid'],e['event'] ) )
            logger.info("Created 'surface_events.csv' file")


        for s in self.surfaces:
            # per surface names:
            surface_name = '_'+s.name.replace('/','')+'_'+s.uid


            # save surface_positions as pickle file
            save_object(s.cache.to_list(),os.path.join(metrics_dir,'srf_positions'+surface_name))

            #save surface_positions as csv
            with open(os.path.join(metrics_dir,'srf_positons'+surface_name+'.csv'),'wb') as csvfile:
                csv_writer =csv.writer(csvfile, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(('frame_idx','timestamp','m_to_screen','m_from_screen','detected_markers'))
                for idx,ts,ref_srf_data in zip(range(len(self.g_pool.timestamps)),self.g_pool.timestamps,s.cache):
                    if in_mark <= idx <= out_mark:
                        if ref_srf_data is not None and ref_srf_data is not False:
                            csv_writer.writerow( (idx,ts,ref_srf_data['m_to_screen'],ref_srf_data['m_from_screen'],ref_srf_data['detected_markers']) )


            # save gaze on srf as csv.
            with open(os.path.join(metrics_dir,'gaze_positions_on_surface'+surface_name+'.csv'),'wb') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(('world_timestamp','world_frame_idx','gaze_timestamp','x_norm','y_norm','x_scaled','y_scaled','on_srf'))
                for idx,ts,ref_srf_data in zip(range(len(self.g_pool.timestamps)),self.g_pool.timestamps,s.cache):
                    if in_mark <= idx <= out_mark:
                        if ref_srf_data is not None and ref_srf_data is not False:
                            for gp in s.gaze_on_srf_by_frame_idx(idx,ref_srf_data['m_from_screen']):
                                csv_writer.writerow( (ts,idx,gp['base']['timestamp'],gp['norm_pos'][0],gp['norm_pos'][1],gp['norm_pos'][0]*s.real_world_size['x'],gp['norm_pos'][1]*s.real_world_size['y'],gp['on_srf']) )


            # save fixation on srf as csv.
            with open(os.path.join(metrics_dir,'fixations_on_surface'+surface_name+'.csv'),'wb') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(('id','start_timestamp','duration','start_frame','end_frame','norm_pos_x','norm_pos_y','x_scaled','y_scaled','on_srf'))
                fixations_on_surface = []
                for idx,ref_srf_data in zip(range(len(self.g_pool.timestamps)),s.cache):
                    if in_mark <= idx <= out_mark:
                        if ref_srf_data is not None and ref_srf_data is not False:
                            for f in s.fixations_on_srf_by_frame_idx(idx,ref_srf_data['m_from_screen']):
                                fixations_on_surface.append(f)

                removed_dublicates = dict([(f['base']['id'],f) for f in fixations_on_surface]).values()
                for f_on_s in removed_dublicates:
                    f = f_on_s['base']
                    f_x,f_y = f_on_s['norm_pos']
                    f_on_srf = f_on_s['on_srf']
                    csv_writer.writerow( (f['id'],f['timestamp'],f['duration'],f['start_frame_index'],f['end_frame_index'],f_x,f_y,f_x*s.real_world_size['x'],f_y*s.real_world_size['y'],f_on_srf) )


            logger.info("Saved surface positon gaze and fixation data for '%s' with uid:'%s'"%(s.name,s.uid))

            if s.heatmap is not None:
                logger.info("Saved Heatmap as .png file.")
                cv2.imwrite(os.path.join(metrics_dir,'heatmap'+surface_name+'.png'),s.heatmap)


        logger.info("Done exporting reference surface data.")
        # if s.detected and self.img is not None:
        #     #let save out the current surface image found in video

        #     #here we get the verts of the surface quad in norm_coords
        #     mapped_space_one = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32).reshape(-1,1,2)
        #     screen_space = cv2.perspectiveTransform(mapped_space_one,s.m_to_screen).reshape(-1,2)
        #     #now we convert to image pixel coods
        #     screen_space[:,1] = 1-screen_space[:,1]
        #     screen_space[:,1] *= self.img.shape[0]
        #     screen_space[:,0] *= self.img.shape[1]
        #     s_0,s_1 = s.real_world_size
        #     #no we need to flip vertically again by setting the mapped_space verts accordingly.
        #     mapped_space_scaled = np.array(((0,s_1),(s_0,s_1),(s_0,0),(0,0)),dtype=np.float32)
        #     M = cv2.getPerspectiveTransform(screen_space,mapped_space_scaled)
        #     #here we do the actual perspactive transform of the image.
        #     srf_in_video = cv2.warpPerspective(self.img,M, (int(s.real_world_size['x']),int(s.real_world_size['y'])) )
        #     cv2.imwrite(os.path.join(metrics_dir,'surface'+surface_name+'.png'),srf_in_video)
        #     logger.info("Saved current image as .png file.")
        # else:
        #     logger.info("'%s' is not currently visible. Seek to appropriate frame and repeat this command."%s.name)


    def get_init_dict(self):
        return {'mode':self.mode}


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """

        self.surface_definitions["offline_square_marker_surfaces"] = [rs.save_to_dict() for rs in self.surfaces if rs.defined]
        self.surface_definitions.close()

        self.close_marker_cacher()
        self.persistent_cache["marker_cache"] = self.cache.to_list()
        self.persistent_cache.close()

        for s in self.surfaces:
            s.close_window()
        self.deinit_gui()

