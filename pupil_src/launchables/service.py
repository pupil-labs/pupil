"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os
import signal

# import sys, platform
from types import SimpleNamespace


def service(
    timebase,
    eye_procs_alive,
    ipc_pub_url,
    ipc_sub_url,
    ipc_push_url,
    user_dir,
    version,
    preferred_remote_port,
    hide_ui,
    debug,
):
    """Maps pupil to gaze data, can run various plug-ins.

    Reacts to notifications:
       ``start_plugin``: Starts given plugin with the given arguments
       ``eye_process.started``: Sets the detection method eye process
       ``service_process.should_stop``: Stops the service process

    Emits notifications:
        ``eye_process.should_start``
        ``eye_process.should_stop``
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
    from time import sleep

    import zmq
    import zmq_tools

    # zmq ipc setup
    zmq_ctx = zmq.Context()
    ipc_pub = zmq_tools.Msg_Dispatcher(zmq_ctx, ipc_push_url)
    gaze_pub = zmq_tools.Msg_Streamer(zmq_ctx, ipc_pub_url)
    pupil_sub = zmq_tools.Msg_Receiver(zmq_ctx, ipc_sub_url, topics=("pupil",))
    notify_sub = zmq_tools.Msg_Receiver(zmq_ctx, ipc_sub_url, topics=("notify",))

    poller = zmq.Poller()
    poller.register(pupil_sub.socket)
    poller.register(notify_sub.socket)

    # log setup
    logging.getLogger("OpenGL").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx, ipc_push_url))
    # create logger for the context of this function
    logger = logging.getLogger(__name__)

    def launch_eye_process(eye_id, delay=0):
        n = {"subject": "eye_process.should_start", "eye_id": eye_id, "delay": delay}
        ipc_pub.notify(n)

    def stop_eye_process(eye_id):
        n = {"subject": "eye_process.should_stop", "eye_id": eye_id}
        ipc_pub.notify(n)

    try:

        # helpers/utils
        import audio
        from background_helper import IPC_Logging_Task_Proxy
        from blink_detection import Blink_Detection
        from calibration_choreography import (
            available_calibration_choreography_plugins,
            patch_loaded_plugins_with_choreography_plugin,
        )
        from file_methods import Persistent_Dict
        from fixation_detector import Fixation_Detector
        from gaze_mapping import registered_gazer_classes
        from methods import delta_t, get_system_info
        from network_api import NetworkApiPlugin

        # Plug-ins
        from plugin import Plugin, Plugin_List, import_runtime_plugins
        from pupil_groups import Pupil_Groups
        from service_ui import Service_UI
        from uvc import get_time_monotonic
        from version_utils import parse_version

        IPC_Logging_Task_Proxy.push_url = ipc_push_url

        process_was_interrupted = False

        def interrupt_handler(sig, frame):
            import traceback

            trace = traceback.format_stack(f=frame)
            logger.debug(f"Caught signal {sig} in:\n" + "".join(trace))
            nonlocal process_was_interrupted
            process_was_interrupted = True

        signal.signal(signal.SIGINT, interrupt_handler)

        logger.info(f"Application Version: {version}")
        logger.info(f"System Info: {get_system_info()}")
        logger.debug(f"Debug flag: {debug}")

        # g_pool holds variables for this process they are accesible to all plugins
        g_pool = SimpleNamespace()
        g_pool.debug = debug
        g_pool.app = "service"
        g_pool.process = "service"
        g_pool.user_dir = user_dir
        g_pool.version = version
        g_pool.get_now = get_time_monotonic
        g_pool.zmq_ctx = zmq_ctx
        g_pool.ipc_pub = ipc_pub
        g_pool.ipc_pub_url = ipc_pub_url
        g_pool.ipc_sub_url = ipc_sub_url
        g_pool.ipc_push_url = ipc_push_url
        g_pool.eye_procs_alive = eye_procs_alive
        g_pool.timebase = timebase
        g_pool.preferred_remote_port = preferred_remote_port
        g_pool.hide_ui = hide_ui

        def get_timestamp():
            return get_time_monotonic() - g_pool.timebase.value

        g_pool.get_timestamp = get_timestamp

        # manage plugins
        runtime_plugins = import_runtime_plugins(
            os.path.join(g_pool.user_dir, "plugins")
        )
        user_launchable_plugins = [
            Service_UI,
            Pupil_Groups,
            NetworkApiPlugin,
            Blink_Detection,
        ] + runtime_plugins
        plugin_by_index = (
            runtime_plugins
            + available_calibration_choreography_plugins()
            + registered_gazer_classes()
            + user_launchable_plugins
        )
        name_by_index = [pupil_datum.__name__ for pupil_datum in plugin_by_index]
        plugin_by_name = dict(zip(name_by_index, plugin_by_index))
        default_plugins = [
            ("Service_UI", {}),
            # Calibration choreography plugin is added bellow by calling
            # patch_world_session_settings_with_choreography_plugin
            ("NetworkApiPlugin", {}),
            ("Blink_Detection", {}),
            ("Fixation_Detector", {}),
        ]
        g_pool.plugin_by_name = plugin_by_name

        tick = delta_t()

        def get_dt():
            return next(tick)

        # load session persistent settings
        session_settings = Persistent_Dict(
            os.path.join(g_pool.user_dir, "user_settings_service")
        )
        if parse_version(session_settings.get("version", "0.0")) < g_pool.version:
            logger.info(
                "Session setting are from older version of this app. I will not use those."
            )
            session_settings.clear()

        g_pool.min_calibration_confidence = session_settings.get(
            "min_calibration_confidence", 0.8
        )

        audio.set_audio_mode(
            session_settings.get("audio_mode", audio.get_default_audio_mode())
        )

        ipc_pub.notify({"subject": "service_process.started"})
        logger.warning("Process started.")
        g_pool.service_should_run = True

        loaded_plugins = session_settings.get("loaded_plugins", default_plugins)

        # Resolve the active calibration choreography plugin
        loaded_plugins = patch_loaded_plugins_with_choreography_plugin(
            loaded_plugins, app=g_pool.app
        )
        session_settings["loaded_plugins"] = loaded_plugins

        # plugins that are loaded based on user settings from previous session
        g_pool.plugins = Plugin_List(g_pool, loaded_plugins)

        # NOTE: The NetworkApiPlugin plugin fails to load when the port is already in use
        # and will set this variable to false. Then we should not even start the eye
        # processes. Otherwise we would have to wait for their initialization before
        # attempting cleanup in Service.
        if g_pool.service_should_run:
            if session_settings.get("eye1_process_alive", True):
                launch_eye_process(1, delay=0.3)
            if session_settings.get("eye0_process_alive", True):
                launch_eye_process(0, delay=0.0)

        def handle_notifications(n):
            subject = n["subject"]
            if subject == "start_plugin":
                try:
                    g_pool.plugins.add(
                        plugin_by_name[n["name"]], args=n.get("args", {})
                    )
                except KeyError as err:
                    logger.error(f"Attempt to load unknown plugin: {err}")
            elif subject == "service_process.should_stop":
                g_pool.service_should_run = False
            elif subject.startswith("meta.should_doc"):
                ipc_pub.notify(
                    {"subject": "meta.doc", "actor": g_pool.app, "doc": service.__doc__}
                )
                for p in g_pool.plugins:
                    if (
                        p.on_notify.__doc__
                        and p.__class__.on_notify != Plugin.on_notify
                    ):
                        ipc_pub.notify(
                            {
                                "subject": "meta.doc",
                                "actor": p.class_name,
                                "doc": p.on_notify.__doc__,
                            }
                        )

        # initiate ui update loop
        ipc_pub.notify(
            {"subject": "service_process.ui.should_update", "initial_delay": 1 / 40}
        )

        g_pool.active_gaze_mapping_plugin = None

        # Event loop
        while g_pool.service_should_run and not process_was_interrupted:
            socks = dict(poller.poll())
            if pupil_sub.socket in socks:
                topic, pupil_datum = pupil_sub.recv()

                events = {}
                events["pupil"] = [pupil_datum]

                if g_pool.active_gaze_mapping_plugin:
                    new_gaze_data = g_pool.active_gaze_mapping_plugin.map_pupil_to_gaze(
                        [pupil_datum]
                    )
                    events["gaze"] = []

                    for gaze_datum in new_gaze_data:
                        gaze_pub.send(gaze_datum)
                        events["gaze"].append(gaze_datum)

                for plugin in g_pool.plugins:
                    plugin.recent_events(events=events)

            if notify_sub.socket in socks:
                topic, n = notify_sub.recv()
                handle_notifications(n)
                for plugin in g_pool.plugins:
                    plugin.on_notify(n)

            # check if a plugin need to be destroyed
            g_pool.plugins.clean()

        session_settings["loaded_plugins"] = g_pool.plugins.get_initializers()
        session_settings["version"] = str(g_pool.version)
        session_settings["eye0_process_alive"] = eye_procs_alive[0].value
        session_settings["eye1_process_alive"] = eye_procs_alive[1].value
        session_settings[
            "min_calibration_confidence"
        ] = g_pool.min_calibration_confidence
        session_settings["audio_mode"] = audio.get_audio_mode()
        session_settings.close()

        # de-init all running plugins
        for pupil_datum in g_pool.plugins:
            pupil_datum.alive = False
        g_pool.plugins.clean()

    except Exception:
        import traceback

        trace = traceback.format_exc()
        logger.error(f"Process Service crashed with trace:\n{trace}")

    finally:
        # shut down eye processes:
        stop_eye_process(0)
        stop_eye_process(1)

        logger.info("Process shutting down.")
        ipc_pub.notify({"subject": "service_process.stopped"})

        # shut down launcher
        n = {"subject": "launcher_process.should_stop"}
        ipc_pub.notify(n)
        sleep(1.0)


def service_profiled(
    timebase,
    eye_procs_alive,
    ipc_pub_url,
    ipc_sub_url,
    ipc_push_url,
    user_dir,
    version,
    preferred_remote_port,
    hide_ui,
    debug,
):
    import cProfile
    import os
    import subprocess

    from .service import service

    cProfile.runctx(
        "service(timebase,eye_procs_alive,ipc_pub_url,ipc_sub_url,ipc_push_url,user_dir,version,preferred_remote_port,hide_ui,debug)",
        {
            "timebase": timebase,
            "eye_procs_alive": eye_procs_alive,
            "ipc_pub_url": ipc_pub_url,
            "ipc_sub_url": ipc_sub_url,
            "ipc_push_url": ipc_push_url,
            "user_dir": user_dir,
            "version": version,
            "preferred_remote_port": preferred_remote_port,
            "hide_ui": hide_ui,
            "debug": debug,
        },
        locals(),
        "service.pstats",
    )
    loc = os.path.abspath(__file__).rsplit("pupil_src", 1)
    gprof2dot_loc = os.path.join(loc[0], "pupil_src", "shared_modules", "gprof2dot.py")
    subprocess.call(
        "python "
        + gprof2dot_loc
        + " -f pstats service.pstats | dot -Tpng -o service_cpu_time.png",
        shell=True,
    )
    print(
        "created cpu time graph for service process. Please check out the png next to the service.py file"
    )
