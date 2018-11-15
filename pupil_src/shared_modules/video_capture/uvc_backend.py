"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import time
import logging
import uvc
from version_utils import VersionFormat
from .base_backend import InitialisationError, Base_Source, Base_Manager
from camera_models import load_intrinsics
from .utils import Check_Frame_Stripes, Exposure_Time

# check versions for our own depedencies as they are fast-changing
assert VersionFormat(uvc.__version__) >= VersionFormat("0.13")

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UVC_Source(Base_Source):
    """
    Camera Capture is a class that encapsualtes uvc.Capture:
    """

    def __init__(
        self,
        g_pool,
        frame_size,
        frame_rate,
        name=None,
        preferred_names=(),
        uid=None,
        uvc_controls={},
        check_stripes=True,
        exposure_mode="manual",
    ):
        import platform

        super().__init__(g_pool)
        self.uvc_capture = None
        self._restart_in = 3
        assert name or preferred_names or uid

        if platform.system() == "Windows":
            self.verify_drivers()

        self.devices = uvc.Device_List()

        devices_by_name = {dev["name"]: dev for dev in self.devices}

        # if uid is supplied we init with that
        if uid:
            try:
                self.uvc_capture = uvc.Capture(uid)
            except uvc.OpenError:
                logger.warning(
                    "No avalilable camera found that matched {}".format(preferred_names)
                )
            except uvc.InitError:
                logger.error("Camera failed to initialize.")
            except uvc.DeviceNotFoundError:
                logger.warning(
                    "No camera found that matched {}".format(preferred_names)
                )

        # otherwise we use name or preffered_names
        else:
            if name:
                preferred_names = (name,)
            else:
                pass
            assert preferred_names

            # try to init by name
            for name in preferred_names:
                for d_name in devices_by_name.keys():
                    if name in d_name:
                        uid_for_name = devices_by_name[d_name]["uid"]
                        try:
                            self.uvc_capture = uvc.Capture(uid_for_name)
                        except uvc.OpenError:
                            logger.info(
                                "{} matches {} but is already in use or blocked.".format(
                                    uid_for_name, name
                                )
                            )
                        except uvc.InitError:
                            logger.error("Camera failed to initialize.")
                        else:
                            break

        # checkframestripes will be initialized accordingly in configure_capture()
        self.check_stripes = check_stripes
        self.exposure_mode = exposure_mode
        self.checkframestripes = None
        self.preferred_exposure_time = None

        # check if we were sucessfull
        if not self.uvc_capture:
            logger.error(
                "Init failed. Capture is started in ghost mode. No images will be supplied."
            )
            self.name_backup = preferred_names
            self.frame_size_backup = frame_size
            self.frame_rate_backup = frame_rate
            self.exposure_time_backup = None
            self._intrinsics = load_intrinsics(
                self.g_pool.user_dir, self.name, self.frame_size
            )
        else:
            self.configure_capture(frame_size, frame_rate, uvc_controls)
            self.name_backup = (self.name,)
            self.frame_size_backup = frame_size
            self.frame_rate_backup = frame_rate
            controls_dict = dict(
                [(c.display_name, c) for c in self.uvc_capture.controls]
            )
            try:
                self.exposure_time_backup = controls_dict[
                    "Absolute Exposure Time"
                ].value
            except KeyError:
                self.exposure_time_backup = None

        self.backup_uvc_controls = {}

    def verify_drivers(self):
        import os
        import subprocess

        DEV_HW_IDS = [
            (0x05A3, 0x9230, "Pupil Cam1 ID0"),
            (0x05A3, 0x9231, "Pupil Cam1 ID1"),
            (0x05A3, 0x9232, "Pupil Cam1 ID2"),
            (0x046D, 0x0843, "Logitech Webcam C930e"),
            (0x17EF, 0x480F, "Lenovo Integrated Camera"),
            (0x0C45, 0x64AB, "Pupil Cam2 ID0"),
        ]
        ids_present = 0
        ids_to_install = []
        for id in DEV_HW_IDS:
            cmd_str_query = "PupilDrvInst.exe --vid {} --pid {}".format(id[0], id[1])
            print("Running ", cmd_str_query)
            proc = subprocess.Popen(cmd_str_query)
            proc.wait()
            if proc.returncode == 2:
                ids_present += 1
                ids_to_install.append(id)
        cmd_str_inst = 'Start-Process PupilDrvInst.exe -Wait -WorkingDirectory \\"{}\\"  -ArgumentList \'--vid {} --pid {} --desc \\"{}\\" --vendor \\"Pupil Labs\\" --inst\' -Verb runas;'
        work_dir = os.getcwd()
        # print('work_dir = ', work_dir)
        if ids_present > 0:
            try:
                os.mkdir(os.path.join(work_dir, "win_drv"))
            except FileExistsError:
                pass
            cmd_str = ""
            rmdir_str = "Remove-Item {}\\win_drv -recurse -Force;".format(work_dir)
            for id in ids_to_install:
                cmd_str += rmdir_str + cmd_str_inst.format(
                    work_dir, id[0], id[1], id[2]
                )
            logger.warning("Updating drivers, please wait...")
            elevation_cmd = 'powershell.exe -version 5 -Command "{}"'.format(cmd_str)
            print(elevation_cmd)
            proc = subprocess.Popen(elevation_cmd)
            proc.wait()
            logger.warning("Done updating drivers!")

    def configure_capture(self, frame_size, frame_rate, uvc_controls):
        # Set camera defaults. Override with previous settings afterwards
        if "Pupil Cam" in self.uvc_capture.name:
            self.ts_offset = 0.0
        else:
            logger.info(
                "Hardware timestamps not supported for {}. Using software timestamps.".format(
                    self.uvc_capture.name
                )
            )
            self.ts_offset = -0.1

        # UVC setting quirks:
        controls_dict = dict([(c.display_name, c) for c in self.uvc_capture.controls])

        if ("Pupil Cam2" in self.uvc_capture.name) and frame_size == (320, 240):
            frame_size = (192, 192)

        self.frame_size = frame_size
        self.frame_rate = frame_rate

        try:
            controls_dict["Auto Focus"].value = 0
        except KeyError:
            pass

        if "Pupil Cam1" in self.uvc_capture.name:

            if "ID0" in self.uvc_capture.name or "ID1" in self.uvc_capture.name:

                self.uvc_capture.bandwidth_factor = 1.3

                try:
                    controls_dict["Auto Exposure Priority"].value = 0
                except KeyError:
                    pass

                try:
                    controls_dict["Auto Exposure Mode"].value = 1
                except KeyError:
                    pass

                try:
                    controls_dict["Saturation"].value = 0
                except KeyError:
                    pass

                try:
                    controls_dict["Absolute Exposure Time"].value = 63
                except KeyError:
                    pass

                try:
                    controls_dict["Backlight Compensation"].value = 2
                except KeyError:
                    pass

                try:
                    controls_dict["Gamma"].value = 100
                except KeyError:
                    pass

            else:
                self.uvc_capture.bandwidth_factor = 2.0
                try:
                    controls_dict["Auto Exposure Priority"].value = 1
                except KeyError:
                    pass

        elif "Pupil Cam2" in self.uvc_capture.name:
            if self.exposure_mode == "auto":
                special_settings = {200: 28, 180: 31}
                controls_dict = dict(
                    [(c.display_name, c) for c in self.uvc_capture.controls]
                )
                self.preferred_exposure_time = Exposure_Time(
                    max_ET=special_settings.get(self.frame_rate, 32),
                    frame_rate=self.frame_rate,
                    mode=self.exposure_mode,
                )

            if self.check_stripes:
                self.checkframestripes = Check_Frame_Stripes()

            try:
                controls_dict["Auto Exposure Priority"].value = 0
            except KeyError:
                pass

            try:
                controls_dict["Auto Exposure Mode"].value = 1
            except KeyError:
                pass

            try:
                controls_dict["Saturation"].value = 0
            except KeyError:
                pass

            try:
                controls_dict["Gamma"].value = 200
            except KeyError:
                pass

        else:
            self.uvc_capture.bandwidth_factor = 2.0
            try:
                controls_dict["Auto Focus"].value = 0
            except KeyError:
                pass

        # Restore session settings after setting defaults
        for c in self.uvc_capture.controls:
            try:
                c.value = uvc_controls[c.display_name]
            except KeyError:
                logger.debug(
                    'No UVC setting "{}" found from settings.'.format(c.display_name)
                )

    def _re_init_capture(self, uid):
        current_size = self.uvc_capture.frame_size
        current_fps = self.uvc_capture.frame_rate
        current_uvc_controls = self._get_uvc_controls()
        self.uvc_capture.close()
        self.uvc_capture = uvc.Capture(uid)
        self.configure_capture(current_size, current_fps, current_uvc_controls)
        self.update_menu()

    def _init_capture(self, uid, backup_uvc_controls={}):
        self.uvc_capture = uvc.Capture(uid)
        self.configure_capture(
            self.frame_size_backup, self.frame_rate_backup, backup_uvc_controls
        )
        self.update_menu()

    def _re_init_capture_by_names(self, names, backup_uvc_controls={}):
        # burn-in test specific. Do not change text!
        self.devices.update()
        for d in self.devices:
            for name in names:
                if d["name"] == name:
                    logger.info("Found device. {}.".format(name))
                    if self.uvc_capture:
                        self._re_init_capture(d["uid"])
                    else:
                        self._init_capture(d["uid"], backup_uvc_controls)
                    return
        raise InitialisationError(
            "Could not find Camera {} during re initilization.".format(names)
        )

    def _restart_logic(self):
        if self._restart_in <= 0:
            if self.uvc_capture:
                logger.warning(
                    "Capture failed to provide frames. Attempting to reinit."
                )
                self.name_backup = (self.uvc_capture.name,)
                self.backup_uvc_controls = self._get_uvc_controls()
                self.uvc_capture = None
            try:
                self._re_init_capture_by_names(
                    self.name_backup, self.backup_uvc_controls
                )
            except (InitialisationError, uvc.InitError):
                time.sleep(0.02)
                self.update_menu()
            self._restart_in = int(5 / 0.02)
        else:
            self._restart_in -= 1

    def recent_events(self, events):
        try:
            frame = self.uvc_capture.get_frame(0.05)

            if self.preferred_exposure_time:
                target = self.preferred_exposure_time.calculate_based_on_frame(frame)
                if target is not None:
                    self.exposure_time = target

            if self.checkframestripes and self.checkframestripes.require_restart(frame):
                # set the self.frame_rate in order to restart
                self.frame_rate = self.frame_rate
                logger.info("Stripes detected")

        except uvc.StreamError:
            self._recent_frame = None
            self._restart_logic()
        except (AttributeError, uvc.InitError):
            self._recent_frame = None
            time.sleep(0.02)
            self._restart_logic()
        else:
            if (
                self.ts_offset
            ):  # c930 timestamps need to be set here. The camera does not provide valid pts from device
                frame.timestamp = uvc.get_time_monotonic() + self.ts_offset
            frame.timestamp -= self.g_pool.timebase.value
            self._recent_frame = frame
            events["frame"] = frame
            self._restart_in = 3

    def _get_uvc_controls(self):
        d = {}
        if self.uvc_capture:
            for c in self.uvc_capture.controls:
                d[c.display_name] = c.value
        return d

    def get_init_dict(self):
        d = super().get_init_dict()
        d["frame_size"] = self.frame_size
        d["frame_rate"] = self.frame_rate
        d["check_stripes"] = self.check_stripes
        d["exposure_mode"] = self.exposure_mode
        if self.uvc_capture:
            d["name"] = self.name
            d["uvc_controls"] = self._get_uvc_controls()
        else:
            d["preferred_names"] = self.name_backup
        return d

    @property
    def name(self):
        if self.uvc_capture:
            return self.uvc_capture.name
        else:
            return "Ghost capture"

    @property
    def frame_size(self):
        if self.uvc_capture:
            return self.uvc_capture.frame_size
        else:
            return self.frame_size_backup

    @frame_size.setter
    def frame_size(self, new_size):
        # closest match for size
        sizes = [
            abs(r[0] - new_size[0]) + abs(r[1] - new_size[1])
            for r in self.uvc_capture.frame_sizes
        ]
        best_size_idx = sizes.index(min(sizes))
        size = self.uvc_capture.frame_sizes[best_size_idx]
        if tuple(size) != tuple(new_size):
            logger.warning(
                "{} resolution capture mode not available. Selected {}.".format(
                    new_size, size
                )
            )
        self.uvc_capture.frame_size = size
        self.frame_size_backup = size

        self._intrinsics = load_intrinsics(
            self.g_pool.user_dir, self.name, self.frame_size
        )

        if self.check_stripes and ("Pupil Cam2" in self.uvc_capture.name):
            self.checkframestripes = Check_Frame_Stripes()

    @property
    def frame_rate(self):
        if self.uvc_capture:
            return self.uvc_capture.frame_rate
        else:
            return self.frame_rate_backup

    @frame_rate.setter
    def frame_rate(self, new_rate):
        # closest match for rate
        rates = [abs(r - new_rate) for r in self.uvc_capture.frame_rates]
        best_rate_idx = rates.index(min(rates))
        rate = self.uvc_capture.frame_rates[best_rate_idx]
        if rate != new_rate:
            logger.warning(
                "{}fps capture mode not available at ({}) on '{}'. Selected {}fps. ".format(
                    new_rate, self.uvc_capture.frame_size, self.uvc_capture.name, rate
                )
            )
        self.uvc_capture.frame_rate = rate
        self.frame_rate_backup = rate

        if "Pupil Cam2" in self.uvc_capture.name:
            special_settings = {200: 28, 180: 31}
            if self.exposure_mode == "auto":
                self.preferred_exposure_time = Exposure_Time(
                    max_ET=special_settings.get(new_rate, 32),
                    frame_rate=new_rate,
                    mode=self.exposure_mode,
                )
            else:
                if self.exposure_time is not None:
                    self.exposure_time = min(
                        self.exposure_time, special_settings.get(new_rate, 32)
                    )

            if self.check_stripes:
                self.checkframestripes = Check_Frame_Stripes()

    @property
    def exposure_time(self):
        if self.uvc_capture:
            try:
                controls_dict = dict(
                    [(c.display_name, c) for c in self.uvc_capture.controls]
                )
                return controls_dict["Absolute Exposure Time"].value
            except KeyError:
                return None
        else:
            return self.exposure_time_backup

    @exposure_time.setter
    def exposure_time(self, new_et):
        try:
            controls_dict = dict(
                [(c.display_name, c) for c in self.uvc_capture.controls]
            )
            if abs(new_et - controls_dict["Absolute Exposure Time"].value) >= 1:
                controls_dict["Absolute Exposure Time"].value = new_et
        except KeyError:
            pass

    @property
    def jpeg_support(self):
        return True

    @property
    def online(self):
        return bool(self.uvc_capture)

    def deinit_ui(self):
        self.remove_menu()

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Local USB Source: {}".format(self.name)
        self.update_menu()

    def update_menu(self):
        del self.menu[:]
        from pyglui import ui

        ui_elements = []

        # lets define some  helper functions:
        def gui_load_defaults():
            for c in self.uvc_capture.controls:
                try:
                    c.value = c.def_val
                except:
                    pass

        def gui_update_from_device():
            for c in self.uvc_capture.controls:
                c.refresh()

        def set_frame_size(new_size):
            self.frame_size = new_size

        def set_frame_rate(new_rate):
            self.frame_rate = new_rate
            self.update_menu()

        if self.uvc_capture is None:
            ui_elements.append(ui.Info_Text("Capture initialization failed."))
            self.menu.extend(ui_elements)
            return

        ui_elements.append(ui.Info_Text("{} Controls".format(self.name)))
        sensor_control = ui.Growing_Menu(label="Sensor Settings")
        sensor_control.append(
            ui.Info_Text("Do not change these during calibration or recording!")
        )
        sensor_control.collapsed = False
        image_processing = ui.Growing_Menu(label="Image Post Processing")
        image_processing.collapsed = True

        sensor_control.append(
            ui.Selector(
                "frame_size",
                self,
                setter=set_frame_size,
                selection=self.uvc_capture.frame_sizes,
                label="Resolution",
            )
        )

        def frame_rate_getter():
            return (
                self.uvc_capture.frame_rates,
                [str(fr) for fr in self.uvc_capture.frame_rates],
            )

        sensor_control.append(
            ui.Selector(
                "frame_rate",
                self,
                selection_getter=frame_rate_getter,
                setter=set_frame_rate,
                label="Frame rate",
            )
        )

        if "Pupil Cam2" in self.uvc_capture.name:
            special_settings = {200: 28, 180: 31}

            def set_exposure_mode(exposure_mode):
                self.exposure_mode = exposure_mode
                if self.exposure_mode == "auto":
                    self.preferred_exposure_time = Exposure_Time(
                        max_ET=special_settings.get(self.frame_rate, 32),
                        frame_rate=self.frame_rate,
                        mode=self.exposure_mode,
                    )
                else:
                    self.preferred_exposure_time = None

                logger.info(
                    "Exposure mode for camera {0} is now set to {1} mode".format(
                        self.uvc_capture.name, exposure_mode
                    )
                )
                self.update_menu()

            def exposure_mode_getter():
                return ["manual", "auto"], ["manual mode", "auto mode"]

            sensor_control.append(
                ui.Selector(
                    "exposure_mode",
                    self,
                    setter=set_exposure_mode,
                    selection_getter=exposure_mode_getter,
                    selection=self.exposure_mode,
                    label="Exposure Mode",
                )
            )

            sensor_control.append(
                ui.Slider(
                    "exposure_time",
                    self,
                    label="Absolute Exposure Time",
                    min=1,
                    max=special_settings.get(self.frame_rate, 32),
                    step=1,
                )
            )
            if self.exposure_mode == "auto":
                sensor_control[-1].read_only = True

        if "Pupil Cam" in self.uvc_capture.name:
            blacklist = [
                "Auto Focus",
                "Absolute Focus",
                "Absolute Iris ",
                "Scanning Mode ",
                "Zoom absolute control",
                "Pan control",
                "Tilt control",
                "Roll absolute control",
                "Privacy Shutter control",
            ]
        else:
            blacklist = []

        if "Pupil Cam2" in self.uvc_capture.name:
            blacklist += [
                "Auto Exposure Mode",
                "Auto Exposure Priority",
                "Absolute Exposure Time",
            ]

        for control in self.uvc_capture.controls:
            c = None
            ctl_name = control.display_name
            if ctl_name in blacklist:
                continue

            # now we add controls
            if control.d_type == bool:
                c = ui.Switch(
                    "value",
                    control,
                    label=ctl_name,
                    on_val=control.max_val,
                    off_val=control.min_val,
                )
            elif control.d_type == int:
                c = ui.Slider(
                    "value",
                    control,
                    label=ctl_name,
                    min=control.min_val,
                    max=control.max_val,
                    step=control.step,
                )
            elif type(control.d_type) == dict:
                selection = [value for name, value in control.d_type.items()]
                labels = [name for name, value in control.d_type.items()]
                c = ui.Selector(
                    "value", control, label=ctl_name, selection=selection, labels=labels
                )
            else:
                pass
            # if control['disabled']:
            #     c.read_only = True
            # if ctl_name == 'Exposure, Auto Priority':
            #     # the controll should always be off. we set it to 0 on init (see above)
            #     c.read_only = True

            if c is not None:
                if control.unit == "processing_unit":
                    image_processing.append(c)
                else:
                    sensor_control.append(c)

        ui_elements.append(sensor_control)

        if image_processing.elements:
            ui_elements.append(image_processing)
        ui_elements.append(ui.Button("refresh", gui_update_from_device))

        if "Pupil Cam2" in self.uvc_capture.name:

            def set_check_stripes(check_stripes):
                self.check_stripes = check_stripes
                if self.check_stripes:
                    self.checkframestripes = Check_Frame_Stripes()
                    logger.info(
                        "Check Stripes for camera {} is now on".format(
                            self.uvc_capture.name
                        )
                    )
                else:
                    self.checkframestripes = None
                    logger.info(
                        "Check Stripes for camera {} is now off".format(
                            self.uvc_capture.name
                        )
                    )

            ui_elements.append(
                ui.Switch(
                    "check_stripes",
                    self,
                    setter=set_check_stripes,
                    label="Check Stripes",
                )
            )
        self.menu.extend(ui_elements)

    def cleanup(self):
        self.devices.cleanup()
        self.devices = None
        if self.uvc_capture:
            self.uvc_capture.close()
            self.uvc_capture = None
        super().cleanup()


class UVC_Manager(Base_Manager):
    """Manages local USB sources

    Attributes:
        check_intervall (float): Intervall in which to look for new UVC devices
    """

    gui_name = "Local USB"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.devices = uvc.Device_List()

    def get_init_dict(self):
        return {}

    def init_ui(self):
        self.add_menu()

        from pyglui import ui

        ui_elements = []
        ui_elements.append(ui.Info_Text("Local UVC sources"))

        def dev_selection_list():
            default = (None, "Select to activate")
            self.devices.update()
            dev_pairs = [default] + [
                (d["uid"], d["name"])
                for d in self.devices
                if "RealSense" not in d["name"]
            ]
            return zip(*dev_pairs)

        def activate(source_uid):
            if not source_uid:
                return
            if not uvc.is_accessible(source_uid):
                logger.error("The selected camera is already in use or blocked.")
                return
            settings = {
                "frame_size": self.g_pool.capture.frame_size,
                "frame_rate": self.g_pool.capture.frame_rate,
                "uid": source_uid,
            }
            if self.g_pool.process == "world":
                self.notify_all(
                    {"subject": "start_plugin", "name": "UVC_Source", "args": settings}
                )
            else:
                self.notify_all(
                    {
                        "subject": "start_eye_capture",
                        "target": self.g_pool.process,
                        "name": "UVC_Source",
                        "args": settings,
                    }
                )

        ui_elements.append(
            ui.Selector(
                "selected_source",
                selection_getter=dev_selection_list,
                getter=lambda: None,
                setter=activate,
                label="Activate source",
            )
        )
        self.menu.extend(ui_elements)

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        self.devices.cleanup()
        self.devices = None

    def recent_events(self, events):
        pass
