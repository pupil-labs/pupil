'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import os, sys, platform


class Global_Container(object):
    pass

def service(timebase,eyes_are_alive,ipc_pub_url,ipc_sub_url,user_dir,version,cap_src):
    """service
    Maps pupil to gaze data
    Can run various plug-ins.
    """

    # We defer the imports because of multiprocessing.
    # Otherwise the world process each process also loads the other imports.
    # This is not harmful but unnecessary.

    #general imports
    from time import time,sleep
    import numpy as np
    import logging
    import zmq
    import zmq_tools
    #zmq ipc setup
    zmq_ctx = zmq.Context()
    ipc_pub = zmq_tools.Msg_Dispatcher(zmq_ctx,ipc_pub_url)
    pupil_sub = zmq_tools.Msg_Receiver(zmq_ctx,ipc_sub_url,topics=('pupil',))
    notify_sub = zmq_tools.Msg_Receiver(zmq_ctx,ipc_sub_url,topics=('notify',))

    poller =  zmq.Poller()
    poller.register(pupil_sub.socket)
    poller.register(notify_sub.socket)

    #log setup
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx,ipc_pub_url))
    # create logger for the context of this function
    logger = logging.getLogger(__name__)

    #monitoring
    import psutil

    # helpers/utils
    from file_methods import Persistent_Dict
    from methods import normalize, denormalize, delta_t, get_system_info
    from version_utils import VersionFormat
    import audio

    #trigger pupil detector cpp build:
    import pupil_detectors
    del pupil_detectors

    # Plug-ins
    from plugin import Plugin_List,import_runtime_plugins
    from calibration_routines import calibration_plugins, gaze_mapping_plugins
    from pupil_sync import Pupil_Sync
    from pupil_remote import Pupil_Remote

    logger.info('Application Version: %s'%version)
    logger.info('System Info: %s'%get_system_info())


    #g_pool holds variables for this process they are accesible to all plugins
    g_pool = Global_Container()
    g_pool.app = 'service'
    g_pool.user_dir = user_dir
    g_pool.version = version
    g_pool.timebase = timebase
    g_pool.zmq_ctx = zmq_ctx
    g_pool.ipc_pub = ipc_pub
    g_pool.ipc_pub_url = ipc_pub_url
    g_pool.ipc_sub_url = ipc_sub_url
    g_pool.eyes_are_alive = eyes_are_alive


    #manage plugins
    runtime_plugins = import_runtime_plugins(os.path.join(g_pool.user_dir,'plugins'))
    plugin_by_index =  runtime_plugins+calibration_plugins+gaze_mapping_plugins
    name_by_index = [p.__name__ for p in plugin_by_index]
    plugin_by_name = dict(zip(name_by_index,plugin_by_index))
    default_plugins = [('Dummy_Gaze_Mapper',{}),('HMD_Calibration',{}),('Pupil_Remote',{})]


    tick = delta_t()
    def get_dt():
        return next(tick)

    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_service'))
    if session_settings.get("version",VersionFormat('0.0')) < g_pool.version:
        logger.info("Session setting are from older version of this app. I will not use those.")
        session_settings.clear()


    g_pool.detection_mapping_mode = session_settings.get('detection_mapping_mode','2d')
    g_pool.active_calibration_plugin = None
    g_pool.active_gaze_mapping_plugin = None

    audio.audio_mode = session_settings.get('audio_mode',audio.default_audio_mode)

    def launch_eye_process(eye_id,delay=0):
        if eyes_are_alive[eye_id].value:
            logger.error("Eye%s process already running."%eye_id)
            return
        n = {'subject':'eye_process.should_start','eye_id':eye_id,'delay':delay}
        ipc_pub.notify(n)

    def stop_eye_process(eye_id):
        n = {'subject':'eye_process.should_stop','eye_id':eye_id}
        ipc_pub.notify(n)


    def handle_notifications(n):
        subject = n['subject']
        if subject == 'set_detection_mapping_mode':
            if n['mode'] == '2d':
                if "Vector_Gaze_Mapper" in g_pool.active_gaze_mapping_plugin.class_name:
                    logger.warning("The gaze mapper is not supported in 2d mode. Please recalibrate.")
                    g_pool.plugins.add(plugin_by_name['Dummy_Gaze_Mapper'])
            g_pool.detection_mapping_mode = n['mode']
        elif subject == 'start_plugin':
            g_pool.plugins.add(plugin_by_name[n['name']],args=n.get('args',{}) )
        elif subject == 'eye_process.started':
            n = {'subject':'set_detection_mapping_mode','mode':g_pool.detection_mapping_mode}
            ipc_pub.notify(n)


    if session_settings.get('eye1_process_alive',False):
        launch_eye_process(1,delay=0.3)
    if session_settings.get('eye0_process_alive',True):
        launch_eye_process(0,delay=0.0)

    ipc_pub.notify({'subject':'service_process.started'})
    logger.warning('Process started.')

    # Event loop
    while True:
        socks = dict(poller.poll())
        if pupil_sub.socket in socks:
            t,p = pupil_sub.recv()
            pupil_graph.add(p['confidence'])
            recent_pupil_data.append(p)
            new_gaze_data = g_pool.active_gaze_mapping_plugin.on_pupil_datum(p)
            for g in new_gaze_data:
                ipc_pub.send('gaze',g)
                recent_gaze_data += new_gaze_data
        elif notify_sub.socket in socks:
            t,n = notify_sub.recv()
            handle_notifications(p)
            for p in plugins:
                p.on_notify(n)



        #check if a plugin need to be destroyed
        g_pool.plugins.clean()


    session_settings['loaded_plugins'] = g_pool.plugins.get_initializers()
    session_settings['version'] = g_pool.version
    session_settings['eye0_process_alive'] = eyes_are_alive[0].value
    session_settings['eye1_process_alive'] = eyes_are_alive[1].value
    session_settings['detection_mapping_mode'] = g_pool.detection_mapping_mode
    session_settings['audio_mode'] = audio.audio_mode
    session_settings.close()

    # de-init all running plugins
    for p in g_pool.plugins:
        p.alive = False
    g_pool.plugins.clean()


    #shut down eye processes:
    stop_eye_process(0)
    stop_eye_process(1)

    logger.info("Process shutting down.")
    ipc_pub.notify({'subject':'service_process.stopped'})

    #shut down launcher
    n = {'subject':'launcher_process.should_stop'}
    ipc_pub.notify(n)


def world_profiled(timebase,eyes_are_alive,ipc_pub_url,ipc_sub_url,user_dir,version,cap_src):
    import cProfile,subprocess,os
    from world import world
    cProfile.runctx("world(timebase,eyes_are_alive,ipc_pub_url,ipc_sub_url,user_dir,version,cap_src)",{'timebase':timebase,'eyes_are_alive':eyes_are_alive,'ipc_pub_url':ipc_pub_url,'ipc_sub_url':ipc_sub_url,'user_dir':user_dir,'version':version,'cap_src':cap_src},locals(),"world.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
    print "created cpu time graph for world process. Please check out the png next to the world.py file"