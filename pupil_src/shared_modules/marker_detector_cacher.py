

def fill_cache(visited_list,video_file_path,q,seek_idx,run):
    import logging,os
    logger = logging.getLogger(__name__+' with pid: '+str(os.getpid()) )
    logger.debug('Started Cacher Process for Marker Detector')
    import cv2
    from uvc_capture import autoCreateCapture, EndofVideoFileError
    from square_marker_detect import detect_markers_robust
    min_marker_perimeter = 80
    aperture = 11
    markers = []

    cap = autoCreateCapture(video_file_path)
    frame_idx = 0

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
            except IndexError:
                # any thing in the past?
                try:
                    next_unvisited = visited_list.index(False,0,frame_idx)
                except IndexError:
                    #no unvisited sites left. Done!
                    logger.debug("Caching Completed")
                    next_unvisited = None
        return next_unvisited


    while run.value:

        if seek_idx.value != -1:
            frame_idx = seek_idx.value
            seek_idx.value = -1

        #check the visited list
        next = next_unvisited_idx(frame_idx)
        if next == None:
            #we are done here:
            break
        elif next != frame_idx:
            #we need to seek:
            logger.debug("seeking to Frame %s" %next)
            cap.seek_to_frame(next)
            frame_idx = next
        else:
            #next frame is unvisited
            pass


        try:
            frame = cap.get_frame()
        except EndofVideoFileError:
            logger.debug('Reached end of video. Rewinding.')
            frame = None
            cap.seek_to_frame(0)
            frame_idx = 0
        else:
            markers = detect_markers_robust(frame.img,
                                            grid_size = 5,
                                            prev_markers=markers,
                                            min_marker_perimeter=min_marker_perimeter,
                                            aperture=aperture,
                                            visualize=0,
                                            true_detect_every_frame=1)
            visited_list[frame.index] = True
            logger.debug("adding Frame %s"%frame.index)
            q.put((frame.index,markers))
            frame_idx +=1
    logger.debug("Closing Cacher Process")
    cap.close()
    q.close()
    return