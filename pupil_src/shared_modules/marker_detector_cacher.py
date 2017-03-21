'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''


class Global_Container(object):
    pass


def fill_cache(visited_list,video_file_path,timestamps,q,seek_idx,run,min_marker_perimeter,invert_image):
    '''
    this function is part of marker_detector it is run as a seperate process.
    it must be kept in a seperate file for namespace sanatisation
    '''
    import os
    import logging
    logger = logging.getLogger(__name__+' with pid: '+str(os.getpid()) )
    logger.debug('Started cacher process for Marker Detector')
    import cv2
    from video_capture import File_Source, EndofVideoFileError,FileSeekError
    from square_marker_detect import detect_markers_robust
    aperture = 9
    markers = []
    cap = File_Source(Global_Container(), video_file_path,timestamps=timestamps)

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

    def handle_frame(next_frame):
        if next_frame != cap.get_frame_index():
            #we need to seek:
            logger.debug("Seeking to Frame {}".format(next_frame))
            try:
                cap.seek_to_frame(next_frame)
            except FileSeekError:
                #could not seek to requested position
                logger.warning("Could not evaluate frame: {}.".format(next_frame))
                visited_list[next_frame] = True # this frame is now visited.
                q.put((next_frame,[])) # we cannot look at the frame, report no detection
                return
            #seeking invalidates prev markers for the detector
            markers[:] = []

        try:
            frame = cap.get_frame()
        except EndofVideoFileError:
            logger.debug("Video File's last frame(s) not accesible")
             #could not read frame
            logger.warning("Could not evaluate frame: {}.".format(next_frame))
            visited_list[next_frame] = True # this frame is now visited.
            q.put((next_frame,[])) # we cannot look at the frame, report no detection
            return

        markers[:] = detect_markers_robust(frame.gray,
                                        grid_size = 5,
                                        prev_markers=markers,
                                        min_marker_perimeter=min_marker_perimeter,
                                        aperture=aperture,
                                        visualize=0,
                                        true_detect_every_frame=1,
                                        invert_image= invert_image)

        visited_list[frame.index] = True
        q.put((frame.index,markers[:])) #object passed will only be pickeled when collected from other process! need to make a copy ot avoid overwrite!!!

    while run.value:
        next_frame = cap.get_frame_index()
        if seek_idx.value != -1:
            next_frame = seek_idx.value
            seek_idx.value = -1
            logger.debug("User required seek. Marker caching at Frame: {}".format(next_frame))

        #check the visited list
        next_frame = next_unvisited_idx(next_frame)
        if next_frame is None:
            #we are done here:
            break
        else:
            handle_frame(next_frame)

    logger.debug("Closing Cacher Process")
    cap.cleanup()
    q.close()
    run.value = False
    return
