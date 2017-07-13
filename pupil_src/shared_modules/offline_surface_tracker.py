'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import sys, os
import cv2
import numpy as np
import csv

from ctypes import c_bool


from itertools import chain
from OpenGL.GL import *
from methods import normalize
from file_methods import Persistent_Dict
from cache_list import Cache_List
from glfw import *
from pyglui import ui
from pyglui.cygl.utils import *

from plugin import Analysis_Plugin_Base
#logging
import logging
logger = logging.getLogger(__name__)

from surface_tracker import Surface_Tracker
from square_marker_detect import draw_markers,m_marker_to_screen
from calibration_routines.camera_intrinsics_estimation import load_camera_calibration
from offline_reference_surface import Offline_Reference_Surface


import multiprocessing
mp = multiprocessing.get_context("fork")


class Offline_Surface_Tracker(Surface_Tracker, Analysis_Plugin_Base):
    """
    Special version of surface tracker for use with videofile source.
    It uses a seperate process to search all frames in the world video file for markers.
     - self.cache is a list containing marker positions for each frame.
     - self.surfaces[i].cache is a list containing surface positions for each frame
    Both caches are build up over time. The marker cache is also session persistent.
    See marker_tracker.py for more info on this marker tracker.
    """

    def __init__(self,g_pool,mode="Show Markers and Surfaces",min_marker_perimeter = 100,invert_image=False,robust_detection=True):
        super().__init__(g_pool,mode,min_marker_perimeter,invert_image,robust_detection,)
        self.order = .2
        self.marker_cache_version = 2
        self.min_marker_perimeter_cacher = 20  #find even super small markers. The surface locater will filter using min_marker_perimeter
        if g_pool.app == 'capture':
           raise Exception('For Player only.')

        self.load_marker_cache()
        self.init_marker_cacher()
        for s in self.surfaces:
            s.init_cache(self.cache,self.camera_calibration,self.min_marker_perimeter,self.min_id_confidence)
        self.recalculate()


    def load_marker_cache(self):
        #check if marker cache is available from last session
        self.persistent_cache = Persistent_Dict(os.path.join(self.g_pool.rec_dir,'square_marker_cache'))
        version = self.persistent_cache.get('version',0)
        cache = self.persistent_cache.get('marker_cache',None)
        if cache is None:
            self.cache = Cache_List([False for _ in self.g_pool.timestamps])
            self.persistent_cache['version'] = self.marker_cache_version
            self.persistent_cache['inverted_markers'] = self.invert_image
        elif version != self.marker_cache_version:
            self.persistent_cache['version'] = self.marker_cache_version
            self.invert_image = self.persistent_cache.get('inverted_markers',False)
            self.cache = Cache_List([False for _ in self.g_pool.timestamps])
            logger.debug("Marker cache version missmatch. Rebuilding marker cache.")
        else:
            self.cache = Cache_List(cache)
            #we overwrite the inverted_image setting from init with the one save in the marker cache.
            self.invert_image = self.persistent_cache.get('inverted_markers',False)
            logger.debug("Loaded marker cache {} / {} frames had been searched before".format(len(self.cache)-self.cache.count(False),len(self.cache)) )

    def clear_marker_cache(self):
        self.cache = Cache_List([False for _ in self.g_pool.timestamps])
        self.persistent_cache['version'] = self.marker_cache_version

    def load_surface_definitions_from_file(self):
        self.surface_definitions = Persistent_Dict(os.path.join(self.g_pool.rec_dir,'surface_definitions'))
        if self.surface_definitions.get('offline_square_marker_surfaces',[]) != []:
            logger.debug("Found ref surfaces defined or copied in previous session.")
            self.surfaces = [Offline_Reference_Surface(self.g_pool,saved_definition=d) for d in self.surface_definitions.get('offline_square_marker_surfaces',[])]
        elif self.surface_definitions.get('realtime_square_marker_surfaces',[]) != []:
            logger.debug("Did not find ref surfaces def created or used by the user in player from earlier session. Loading surfaces defined during capture.")
            self.surfaces = [Offline_Reference_Surface(self.g_pool,saved_definition=d) for d in self.surface_definitions.get('realtime_square_marker_surfaces',[])]
        else:
            logger.debug("No surface defs found. Please define using GUI.")
            self.surfaces = []


    def init_gui(self):
        self.menu = ui.Scrolling_Menu('Offline Surface Tracker')
        self.g_pool.gui.append(self.menu)
        self.add_button = ui.Thumb('add_surface',setter=lambda x: self.add_surface(),getter=lambda:False,label='A',hotkey='a')
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
        def close():
            self.alive=False

        def set_min_marker_perimeter(val):
            self.min_marker_perimeter = val
            self.notify_all({'subject':'min_marker_perimeter_changed','delay':1})

        def set_invert_image(val):
            self.invert_image = val
            self.invalidate_marker_cache()
            self.invalidate_surface_caches()

        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',close))
        self.menu.append(ui.Switch('invert_image',self,setter=set_invert_image,label='Use inverted markers'))
        self.menu.append(ui.Slider('min_marker_perimeter',self,min=20,max=500,step=1,setter=set_min_marker_perimeter))
        self.menu.append(ui.Info_Text('The offline surface tracker will look for markers in the entire video. By default it uses surfaces defined in capture. You can change and add more surfaces here.'))
        self.menu.append(ui.Info_Text("Press the export button or type 'e' to start the export."))
        self.menu.append(ui.Selector('mode',self,label='Mode',selection=["Show Markers and Surfaces","Show marker IDs","Show Heatmaps","Show Metrics"] ))
        self.menu.append(ui.Info_Text('To see heatmap or surface metrics visualizations, click (re)-calculate gaze distributions. Set "X size" and "Y size" for each surface to see heatmap visualizations.'))
        self.menu.append(ui.Button("(Re)-calculate gaze distributions", self.recalculate))
        self.menu.append(ui.Button("Add surface", lambda:self.add_surface()))
        for s in self.surfaces:
            idx = self.surfaces.index(s)
            s_menu = ui.Growing_Menu("Surface {}".format(idx))
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
        if notification['subject'] == 'min_data_confidence_changed':
            logger.info('Min_data_confidence changed. Recalculating.')
            self.recalculate()
        elif notification['subject'] == 'surfaces_changed':
            logger.info('Surfaces changed. Recalculating.')
            self.recalculate()
        elif notification['subject'] == 'min_marker_perimeter_changed':
            logger.info('Min marker perimeter adjusted. Re-detecting surfaces.')
            self.invalidate_surface_caches()
        elif notification['subject'] == "should_export":
            self.save_surface_statsics_to_file(notification['range'],notification['export_dir'])


    def on_window_resize(self,window,w,h):
        self.win_size = w,h


    def add_surface(self):
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


    def invalidate_surface_caches(self):
        for s in self.surfaces:
            s.cache = None

    def recent_events(self,events):
        frame = events.get('frame')
        if not frame:
            return
        self.img_shape = frame.img.shape
        self.update_marker_cache()
        # self.markers = [m for m in self.cache[frame.index] if m['perimeter'>=self.min_marker_perimeter]
        self.markers = self.cache[frame.index]
        if self.markers == False:
            self.markers = []
            self.seek_marker_cacher(frame.index) # tell precacher that it better have every thing from here on analyzed


        events['surfaces'] = []
        # locate surfaces
        for s in self.surfaces:
            if not s.locate_from_cache(frame.index):
                s.locate(self.markers,self.camera_calibration,self.min_marker_perimeter,self.min_id_confidence)
            if s.detected:
                events['surfaces'].append({'name':s.name,'uid':s.uid,'m_to_screen':s.m_to_screen.tolist(),'m_from_screen':s.m_from_screen.tolist(),'gaze_on_srf': s.gaze_on_srf, 'timestamp':frame.timestamp,'camera_pose_3d':s.camera_pose_3d.tolist() if s.camera_pose_3d is not None else None})

        if self.mode == "Show marker IDs":
            draw_markers(frame.img,self.markers)

        elif self.mode == "Show Markers and Surfaces":
            # edit surfaces by user
            if self.edit_surf_verts:
                window = glfwGetCurrentContext()
                pos = glfwGetCursorPos(window)
                pos = normalize(pos,glfwGetWindowSize(window),flip_y=True)
                for s,v_idx in self.edit_surf_verts:
                    if s.detected:
                        new_pos =  s.img_to_ref_surface(np.array(pos))
                        s.move_vertex(v_idx,new_pos)
            else:
                # update srf with no or invald cache:
                for s in self.surfaces:
                    if s.cache == None:
                        s.init_cache(self.cache,self.camera_calibration,self.min_marker_perimeter,self.min_id_confidence)
                        self.notify_all({'subject':'surfaces_changed','delay':1})



        #allow surfaces to open/close windows
        for s in self.surfaces:
            if s.window_should_close:
                s.close_window()
            if s.window_should_open:
                s.open_window()

    def invalidate_marker_cache(self):
        self.close_marker_cacher()
        self.clear_marker_cache()
        self.init_marker_cacher()

    def init_marker_cacher(self):
        from marker_detector_cacher import fill_cache
        visited_list = [False if x == False else True for x in self.cache]
        video_file_path =  self.g_pool.capture.source_path
        timestamps = self.g_pool.capture.timestamps
        self.cache_queue = mp.Queue()
        self.cacher_seek_idx = mp.Value('i',0)
        self.cacher_run = mp.Value(c_bool,True)
        self.cacher = mp.Process(target=fill_cache, args=(visited_list,video_file_path,timestamps,self.cache_queue,self.cacher_seek_idx,self.cacher_run,self.min_marker_perimeter_cacher,self.invert_image))
        self.cacher.start()

    def update_marker_cache(self):
        while not self.cache_queue.empty():
            idx,c_m = self.cache_queue.get()
            self.cache.update(idx,c_m)
            for s in self.surfaces:
                s.update_cache(self.cache,camera_calibration=self.camera_calibration,min_marker_perimeter=self.min_marker_perimeter,min_id_confidence=self.min_id_confidence,idx=idx)
            if self.cacher_run.value == False:
                self.recalculate()

    def seek_marker_cacher(self,idx):
        self.cacher_seek_idx.value = idx

    def close_marker_cacher(self):
        self.update_marker_cache()
        self.cacher_run.value = False
        self.cacher.join(1.0)
        if self.cacher.is_alive():
            logger.error("Marker cacher unresponsive - terminating.")
            self.cacher.terminate()

    def gl_display(self):
        """
        Display marker and surface info inside world screen
        """
        self.gl_display_cache_bars()

        super().gl_display()

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
        padding = 30.

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


    def save_surface_statsics_to_file(self,export_range,export_dir):
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
        metrics_dir = os.path.join(export_dir,'surfaces')
        section = slice(*export_range)
        in_mark = section.start
        out_mark = section.stop
        logger.info("exporting metrics to {}".format(metrics_dir))
        if os.path.isdir(metrics_dir):
            logger.info("Will overwrite previous export for this section")
        else:
            try:
                os.mkdir(metrics_dir)
            except:
                logger.warning("Could not make metrics dir {}".format(metrics_dir))
                return


        with open(os.path.join(metrics_dir,'surface_visibility.csv'),'w',encoding='utf-8',newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')

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


        with open(os.path.join(metrics_dir,'surface_gaze_distribution.csv'),'w',encoding='utf-8',newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')

            # gaze distribution report
            gaze_in_section = list(chain(*self.g_pool.gaze_positions_by_frame[section]))
            not_on_any_srf = set([gp['timestamp'] for gp in gaze_in_section])

            csv_writer.writerow(('total_gaze_point_count',len(gaze_in_section)))
            csv_writer.writerow((''))
            csv_writer.writerow(('surface_name','gaze_count'))

            for s in self.surfaces:
                gaze_on_srf  = s.gaze_on_srf_in_section(section)
                gaze_on_srf = set([gp['base_data']['timestamp'] for gp in gaze_on_srf])
                not_on_any_srf -= gaze_on_srf
                csv_writer.writerow( (s.name, len(gaze_on_srf)) )

            csv_writer.writerow(('not_on_any_surface', len(not_on_any_srf) ) )
            logger.info("Created 'surface_gaze_distribution.csv' file")



        with open(os.path.join(metrics_dir,'surface_events.csv'),'w',encoding='utf-8',newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')

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

            #save surface_positions as csv
            with open(os.path.join(metrics_dir,'srf_positons'+surface_name+'.csv'),'w',encoding='utf-8',newline='') as csvfile:
                csv_writer =csv.writer(csvfile, delimiter=',')
                csv_writer.writerow(('frame_idx','timestamp','m_to_screen','m_from_screen','detected_markers'))
                for idx,ts,ref_srf_data in zip(range(len(self.g_pool.timestamps)),self.g_pool.timestamps,s.cache):
                    if in_mark <= idx <= out_mark:
                        if ref_srf_data is not None and ref_srf_data is not False:
                            csv_writer.writerow( (idx,ts,ref_srf_data['m_to_screen'],ref_srf_data['m_from_screen'],ref_srf_data['detected_markers']) )


            # save gaze on srf as csv.
            with open(os.path.join(metrics_dir,'gaze_positions_on_surface'+surface_name+'.csv'),'w',encoding='utf-8',newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                csv_writer.writerow(('world_timestamp','world_frame_idx','gaze_timestamp','x_norm','y_norm','x_scaled','y_scaled','on_srf'))
                for idx,ts,ref_srf_data in zip(range(len(self.g_pool.timestamps)),self.g_pool.timestamps,s.cache):
                    if in_mark <= idx <= out_mark:
                        if ref_srf_data is not None and ref_srf_data is not False:
                            for gp in s.gaze_on_srf_by_frame_idx(idx,ref_srf_data['m_from_screen']):
                                csv_writer.writerow( (ts,idx,gp['base_data']['timestamp'],gp['norm_pos'][0],gp['norm_pos'][1],gp['norm_pos'][0]*s.real_world_size['x'],gp['norm_pos'][1]*s.real_world_size['y'],gp['on_srf']) )


            # save fixation on srf as csv.
            with open(os.path.join(metrics_dir,'fixations_on_surface'+surface_name+'.csv'),'w',encoding='utf-8',newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                csv_writer.writerow(('id','start_timestamp','duration','start_frame','end_frame','norm_pos_x','norm_pos_y','x_scaled','y_scaled','on_srf'))
                fixations_on_surface = []
                for idx,ref_srf_data in zip(range(len(self.g_pool.timestamps)),s.cache):
                    if in_mark <= idx <= out_mark:
                        if ref_srf_data is not None and ref_srf_data is not False:
                            for f in s.fixations_on_srf_by_frame_idx(idx,ref_srf_data['m_from_screen']):
                                fixations_on_surface.append(f)

                removed_duplicates = dict([(f['base_data']['id'],f) for f in fixations_on_surface]).values()
                for f_on_s in removed_duplicates:
                    f = f_on_s['base_data']
                    f_x,f_y = f_on_s['norm_pos']
                    f_on_srf = f_on_s['on_srf']
                    csv_writer.writerow( (f['id'],f['timestamp'],f['duration'],f['start_frame_index'],f['end_frame_index'],f_x,f_y,f_x*s.real_world_size['x'],f_y*s.real_world_size['y'],f_on_srf) )


            logger.info("Saved surface positon gaze and fixation data for '{}' with uid:'{}'".format(s.name,s.uid))

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


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """

        self.surface_definitions["offline_square_marker_surfaces"] = [rs.save_to_dict() for rs in self.surfaces if rs.defined]
        self.surface_definitions.close()

        self.close_marker_cacher()
        self.persistent_cache['inverted_markers'] = self.invert_image
        self.persistent_cache["marker_cache"] = self.cache.to_list()
        self.persistent_cache.close()

        for s in self.surfaces:
            s.close_window()
        self.deinit_gui()
