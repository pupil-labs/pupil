'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
import os
# import sys, platform


class Global_Container(object):
    pass


def service(timebase, eyes_are_alive, ipc_pub_url, ipc_sub_url, ipc_push_url, user_dir, version):
    """Maps pupil to gaze data, can run various plug-ins.

    Reacts to notifications:
       ``set_detection_mapping_mode``: Sets detection method
       ``start_plugin``: Starts given plugin with the given arguments
       ``eye_process.started``: Sets the detection method eye process
       ``service_process.should_stop``: Stops the service process

    Emits notifications:
        ``eye_process.should_start``
        ``eye_process.should_stop``
        ``set_detection_mapping_mode``
        ``service_process.started``
        ``service_process.stopped``
        ``launcher_process.should_stop``

    Emits data:
        ``gaze``: Gaze data from current gaze mapping plugin.``
        ``*``: any other plugin generated data in the events that it not [dt,pupil,gaze].
    """

    # We defer the imports because of multiprocessing.
    # Otherwise the service process each process also loads the other imports.
    # This is not harmful but unnecessary.

    # general imports
    import logging
    import zmq
    import zmq_tools

    # zmq ipc setup
    zmq_ctx = zmq.Context()
    ipc_pub = zmq_tools.Msg_Dispatcher(zmq_ctx, ipc_push_url)
    gaze_pub = zmq_tools.Msg_Streamer(zmq_ctx, ipc_pub_url)
    pupil_sub = zmq_tools.Msg_Receiver(zmq_ctx, ipc_sub_url, topics=('pupil',))
    notify_sub = zmq_tools.Msg_Receiver(zmq_ctx, ipc_sub_url, topics=('notify',))

    poller = zmq.Poller()
    poller.register(pupil_sub.socket)
    poller.register(notify_sub.socket)

    # log setup
    logging.getLogger("OpenGL").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx, ipc_push_url))
    # create logger for the context of this function
    logger = logging.getLogger(__name__)

    # helpers/utils
    from file_methods import Persistent_Dict
    from methods import delta_t, get_system_info
    from version_utils import VersionFormat
    import audio
    from uvc import get_time_monotonic

    # trigger pupil detector cpp build:
    import pupil_detectors
    del pupil_detectors

    # Plug-ins
    from plugin import Plugin, Plugin_List, import_runtime_plugins
    from calibration_routines import calibration_plugins, gaze_mapping_plugins
    from pupil_remote import Pupil_Remote
    from pupil_groups import Pupil_Groups

    logger.info('Application Version: {}'.format(version))
    logger.info('System Info: {}'.format(get_system_info()))

    # g_pool holds variables for this process they are accesible to all plugins
    g_pool = Global_Container()
    g_pool.app = 'service'
    g_pool.user_dir = user_dir
    g_pool.version = version
    g_pool.get_now = get_time_monotonic
    g_pool.zmq_ctx = zmq_ctx
    g_pool.ipc_pub = ipc_pub
    g_pool.ipc_pub_url = ipc_pub_url
    g_pool.ipc_sub_url = ipc_sub_url
    g_pool.ipc_push_url = ipc_push_url
    g_pool.eyes_are_alive = eyes_are_alive
    g_pool.timebase = timebase

    def get_timestamp():
        return get_time_monotonic()-g_pool.timebase.value
    g_pool.get_timestamp = get_timestamp

    # manage plugins
    runtime_plugins = import_runtime_plugins(os.path.join(g_pool.user_dir, 'plugins'))
    user_launchable_plugins = [Pupil_Groups, Pupil_Remote]+runtime_plugins
    plugin_by_index = runtime_plugins+calibration_plugins+gaze_mapping_plugins+user_launchable_plugins
    name_by_index = [p.__name__ for p in plugin_by_index]
    plugin_by_name = dict(zip(name_by_index, plugin_by_index))
    default_plugins = [('Dummy_Gaze_Mapper', {}), ('HMD_Calibration', {}), ('Pupil_Remote', {})]

    tick = delta_t()

    def get_dt():
        return next(tick)

    # load session persistent settings
    session_settings = Persistent_Dict(os.path.join(g_pool.user_dir, 'user_settings_service'))
    if session_settings.get("version", VersionFormat('0.0')) < g_pool.version:
        logger.info("Session setting are from older version of this app. I will not use those.")
        session_settings.clear()

    g_pool.detection_mapping_mode = session_settings.get('detection_mapping_mode', '2d')
    g_pool.active_calibration_plugin = None
    g_pool.active_gaze_mapping_plugin = None

    audio.audio_mode = session_settings.get('audio_mode', audio.default_audio_mode)

    # plugins that are loaded based on user settings from previous session
    g_pool.plugins = Plugin_List(g_pool, plugin_by_name, session_settings.get('loaded_plugins', default_plugins))

    def launch_eye_process(eye_id, delay=0):
        n = {'subject': 'eye_process.should_start', 'eye_id': eye_id, 'delay': delay}
        ipc_pub.notify(n)

    def stop_eye_process(eye_id):
        n = {'subject': 'eye_process.should_stop', 'eye_id': eye_id}
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
            g_pool.plugins.add(plugin_by_name[n['name']], args=n.get('args', {}))
        elif subject == 'eye_process.started':
            n = {'subject': 'set_detection_mapping_mode', 'mode': g_pool.detection_mapping_mode}
            ipc_pub.notify(n)
        elif subject == 'service_process.should_stop':
            g_pool.service_should_run = False
        elif subject.startswith('meta.should_doc'):
            ipc_pub.notify({
                'subject': 'meta.doc',
                'actor': g_pool.app,
                'doc': service.__doc__})
            for p in g_pool.plugins:
                if p.on_notify.__doc__ and p.__class__.on_notify != Plugin.on_notify:
                    ipc_pub.notify({
                        'subject': 'meta.doc',
                        'actor': p.class_name,
                        'doc': p.on_notify.__doc__})

    if session_settings.get('eye1_process_alive', False):
        launch_eye_process(1, delay=0.3)
    if session_settings.get('eye0_process_alive', True):
        launch_eye_process(0, delay=0.0)

    ipc_pub.notify({'subject': 'service_process.started'})
    logger.warning('Process started.')
    g_pool.service_should_run = True

    # Event loop
    while g_pool.service_should_run:
        socks = dict(poller.poll())
        if pupil_sub.socket in socks:
            t, p = pupil_sub.recv()
            new_gaze_data = g_pool.active_gaze_mapping_plugin.on_pupil_datum(p)
            for g in new_gaze_data:
                gaze_pub.send('gaze', g)

            events = {}
            events['gaze_positions'] = new_gaze_data
            events['pupil_positions'] = [p]
            for plugin in g_pool.plugins:
                plugin.recent_events(events=events)

        if notify_sub.socket in socks:
            t, n = notify_sub.recv()
            handle_notifications(n)
            for plugin in g_pool.plugins:
                plugin.on_notify(n)

        # check if a plugin need to be destroyed
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

    # shut down eye processes:
    stop_eye_process(0)
    stop_eye_process(1)

    logger.info("Process shutting down.")
    ipc_pub.notify({'subject': 'service_process.stopped'})

    # shut down launcher
    n = {'subject': 'launcher_process.should_stop'}
    ipc_pub.notify(n)


def service_profiled(timebase, eyes_are_alive, ipc_pub_url, ipc_sub_url, ipc_push_url, user_dir, version):
    import cProfile, subprocess, os
    from .service import service
    cProfile.runctx("service(timebase,eyes_are_alive,ipc_pub_url,ipc_sub_url,ipc_push_url,user_dir,version)",
                    {'timebase': timebase, 'eyes_are_alive': eyes_are_alive, 'ipc_pub_url': ipc_pub_url,
                     'ipc_sub_url': ipc_sub_url, 'ipc_push_url': ipc_push_url, 'user_dir': user_dir,
                     'version': version}, locals(), "service.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules', 'gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats service.pstats | dot -Tpng -o service_cpu_time.png", shell=True)
    print("created cpu time graph for service process. Please check out the png next to the service.py file")
