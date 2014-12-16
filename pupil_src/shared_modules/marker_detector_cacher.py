'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''




def fill_cache(visited_list,video_file_path,q,seek_idx,run):
    '''
    this function is part of marker_detector it is run as a separate process.
    it must be kept in a separate file for namespace sanitisation.
    '''
    import os
    import logging
    logger = logging.getLogger(__name__+' with pid: '+str(os.getpid()) )
    logger.debug('Started cacher process for Marker Detector')
    import cv2
    from uvc_capture import autoCreateCapture, EndofVideoFileError,FileSeekError
    from square_marker_detect import detect_markers_robust,detect_markers_simple
    min_marker_perimeter = 80
    aperture = 9
    markers = []

    cap = autoCreateCapture(video_file_path)

    def next_unvisited_idx(frame_idx):
        try:
            visited = visited_list[frame_idx]
        except IndexError:
            visited = True # trigger search

        if not visited:
            next_unvisited = frame_idx
        else:
            # find next unvisited site in the future
            try:
                next_unvisited = visited_list.index(False,frame_idx)
            except ValueError:
                # any thing in the past?
                try:
                    next_unvisited = visited_list.index(False,0,frame_idx)
                except ValueError:
                    #no unvisited sites left. Done!
                    logger.debug("Caching completed.")
                    next_unvisited = None
        return next_unvisited

    def handle_frame(next):
        if next != cap.get_frame_index():
            #we need to seek:
            logger.debug("Seeking to Frame %s" %next)
            try:
                cap.seek_to_frame(next)
            except FileSeekError:
                #could not seek to requested position
                logger.warning("Could not evaluate frame: %s."%next)
                visited_list[next] = True # this frame is now visited.
                q.put((next,[])) # we cannot look at the frame, report no detection
                return
            #seeking invalidates prev markers for the detector
            markers[:] = []

        try:
            frame = cap.get_frame()
        except EndofVideoFileError:
            logger.debug("Video File's last frame(s) not accesible")
             #could not read frame
            logger.warning("Could not evaluate frame: %s."%next)
            visited_list[next] = True # this frame is now visited.
            q.put((next,[])) # we cannot look at the frame, report no detection
            return

        markers[:] = detect_markers_robust(frame.img,
                                        grid_size = 5,
                                        prev_markers=markers,
                                        min_marker_perimeter=min_marker_perimeter,
                                        aperture=aperture,
                                        visualize=0,
                                        true_detect_every_frame=1)

        # markers[:] = detect_markers_simple(frame.img,
        #                         grid_size = 5,
        #                         min_marker_perimeter=min_marker_perimeter,
        #                         aperture=aperture,
        #                         visualize=0)
        visited_list[frame.index] = True
        q.put((frame.index,markers[:])) #object passed will only be pickeld when collected from other process! need to make a copy ot avoid overwrite!!!

    while run.value:
        next = cap.get_frame_index()
        if seek_idx.value != -1:
            next = seek_idx.value
            seek_idx.value = -1
            logger.debug("User required seek. Marker caching at Frame: %s"%next)


        #check the visited list
        next = next_unvisited_idx(next)
        if next == None:
            #we are done here:
            break
        else:
            handle_frame(next)


    logger.debug("Closing Cacher Process")
    cap.close()
    q.close()
    return
