"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import enum
import logging
import platform
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import gl_utils
import numpy as np
import uvc
from camera_models import Camera_Model
from pyglui import cygl, ui
from version_utils import parse_version

from .base_backend import Base_Manager, Base_Source, InitialisationError, SourceInfo
from .neon_backend.definitions import SCENE_CAM_SPEC, CameraSpec
from .utils import Check_Frame_Stripes, Exposure_Time

# check versions for our own depedencies as they are fast-changing
assert parse_version(uvc.__version__) >= parse_version("0.13")

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TJSAMP(enum.IntEnum):
    """Reimplements turbojpeg.h TJSAMP"""

    TJSAMP_444 = 0
    TJSAMP_422 = 1
    TJSAMP_420 = 2
    TJSAMP_GRAY = 3
    TJSAMP_440 = 4
    TJSAMP_411 = 5


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
        exposure_mode="auto",
        *args,
        **kwargs,
    ):

        super().__init__(g_pool, *args, **kwargs)
        self.uvc_capture = None
        self._last_ts = None
        self._restart_in = 3
        assert name or preferred_names or uid

        if platform.system() == "Windows":
            if g_pool.skip_driver_installation:
                logger.debug("Skipping driver installation")
            else:
                self.verify_drivers()

        self.devices = uvc.Device_List()

        devices_by_name = {dev["name"]: dev for dev in self.devices}

        # if uid is supplied we init with that
        if uid:
            try:
                self.uvc_capture = uvc.Capture(uid)
            except uvc.OpenError:
                logger.warning(
                    f"Camera matching {preferred_names} found but not available"
                )
            except uvc.InitError:
                logger.error("Camera failed to initialize.")
            except uvc.DeviceNotFoundError:
                logger.warning(f"No camera found that matched {preferred_names}")

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
                            break
                        except uvc.OpenError:
                            logger.info(
                                f"{uid_for_name} matches {name} but is already in use "
                                "or blocked."
                            )
                        except uvc.InitError:
                            logger.error("Camera failed to initialize.")
                if self.uvc_capture:
                    break

        # checkframestripes will be initialized accordingly in configure_capture()
        self.enable_stripe_checks = check_stripes
        self.exposure_mode = exposure_mode
        self.stripe_detector = None
        self.preferred_exposure_time = None

        # check if we were sucessfull
        if not self.uvc_capture:
            logger.error("Could not connect to device! No images will be supplied.")
            self.name_backup = preferred_names
            self.frame_size_backup = frame_size
            self.frame_rate_backup = frame_rate
            self.exposure_time_backup = None
            self._intrinsics = Camera_Model.from_file(
                self.g_pool.user_dir, self.name, self.frame_size
            )
        else:
            self.configure_capture(frame_size, frame_rate, uvc_controls)
            self.name_backup = (self.name,)
            self.frame_size_backup = frame_size
            self.frame_rate_backup = frame_rate
            controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
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
            (0x04B4, 0x0036, "Neon Sensor Module v1"),
            (0x0BDA, 0x3036, "Neon Scene Camera v1"),
        ]
        ids_present = 0
        ids_to_install = []
        for id in DEV_HW_IDS:
            cmd_str_query = f"PupilDrvInst.exe --vid {id[0]} --pid {id[1]}"
            print(f"Running {cmd_str_query}")
            logger.debug(f"Running {cmd_str_query}")
            proc = subprocess.Popen(cmd_str_query)
            proc.wait()
            if proc.returncode == 2:
                ids_present += 1
                ids_to_install.append(id)

        if ids_present > 0:
            logger.warning("Updating drivers, please wait...")

            # NOTE: libwdi in PupilDrvIns.exe cannot deal with unicode characters in the
            # temporary path where the drivers will be installed. Check for non-ascii in
            # the default tempdir location and use C:\Windows\Temp as fallback.
            temp_path = None  # use default temp_path
            try:
                with tempfile.TemporaryDirectory(dir=temp_path) as work_dir:
                    work_dir.encode("ascii")
            except UnicodeEncodeError:
                temp_path = Path("C:\\Windows\\Temp")
                if not temp_path.exists():
                    raise RuntimeError(
                        "Your path to Pupil contains Unicode characters and the "
                        "default Temp folder location could not be found. Please place "
                        "Pupil on a path which only contains english characters!"
                    )
                logger.debug(
                    "Detected Unicode characters in working directory! "
                    "Switching temporary driver install location to C:\\Windows\\Temp"
                )

            for id in ids_to_install:
                # Create a new temp dir for every driver so even when experiencing
                # PermissionErrors, we can just continue installing all necessary
                # drivers.
                try:
                    with tempfile.TemporaryDirectory(dir=temp_path) as work_dir:
                        # Need to resolve PupilDrvInst.exe location, which is on PATH
                        # only for running from source. For bundle, the most stable
                        # solution is to use sys._MEIPASS. Note that Path.cwd() can e.g.
                        # return wrong results!
                        if getattr(sys, "frozen", False):
                            bundle_dir = sys._MEIPASS
                            driver_exe = Path(bundle_dir) / "PupilDrvInst.exe"
                            logger.debug(
                                f"Detected running from bundle."
                                f" Using full path to PupilDrvInst.exe at: {driver_exe}"
                            )
                        else:
                            driver_exe = "PupilDrvInst.exe"
                            logger.debug(
                                f"Detected running from source."
                                f" Assuming PupilDrvInst.exe is available on PATH!"
                            )

                        # Using """ here to be able to use both " and ' without escaping
                        # Note: ArgumentList needs quotes ordered this way (' outer, "
                        # inner), otherwise it won't work
                        cmd = (
                            f"""Start-Process '{driver_exe}' -Wait -Verb runas"""
                            f""" -WorkingDirectory '{work_dir}'"""
                            f""" -ArgumentList '--vid {id[0]} --pid {id[1]} --desc "{id[2]}" --vendor "Pupil Labs" --inst'"""
                        )

                        # We now have strings with both " and ' used for quoting. For
                        # passing this as command to powershell.exe below, we need to
                        # wrap the whole string in one set of "". In order for this to
                        # work, we need to escape all " again:
                        cmd = cmd.replace('"', '\\"')

                        elevation_cmd = f'powershell.exe -version 5 -Command "{cmd}"'

                        print(elevation_cmd)
                        logger.debug(elevation_cmd)
                        subprocess.Popen(elevation_cmd).wait()
                except PermissionError:
                    # This can be raised when cleaning up the TemporaryDirectory, if the
                    # process was started from a non-admin shell for a non-admin user
                    # and has only been elevated for the powershell commands. The files
                    # then belong to a different user and cannot be deleted. We can
                    # ignore this, as temp dirs will be cleaned up on shutdown anyways.
                    logger.warning(
                        "Pupil was not run as administrator. If the drivers do not "
                        "work, please try running as administrator again!"
                    )
                    logger.debug(traceback.format_exc())
                except Exception:
                    logger.error(
                        "An error was encountered during the automatic driver "
                        "installation. Please consider installing them manually."
                    )
                    logger.debug(traceback.format_exc())

            logger.info("Done updating drivers!")

    def configure_capture(self, frame_size, frame_rate, uvc_controls):
        # Set camera defaults. Override with previous settings afterwards
        if "Pupil Cam" in self.uvc_capture.name:
            if platform.system() == "Windows":
                # NOTE: Hardware timestamps seem to be broken on windows. Needs further
                # investigation! Disabling for now.
                # TODO: Find accurate offsets for different resolutions!
                offsets = {"ID0": -0.015, "ID1": -0.015, "ID2": -0.07}
                match = re.match(r"Pupil Cam\d (?P<cam_id>ID[0-2])", self.name)
                if not match:
                    logger.debug(f"Could not parse camera name: {self.name}")
                    self.ts_offset = -0.01
                else:
                    self.ts_offset = offsets[match.group("cam_id")]

            else:
                # use hardware timestamps
                self.ts_offset = None
        else:
            logger.debug(
                f"Hardware timestamps not supported for {self.uvc_capture.name}. "
                "Using software timestamps."
            )
            self.ts_offset = -0.1

        # Set NEON bandwidth factors
        if self.uvc_capture.name in CameraSpec.spec_by_name():
            cam = self.uvc_capture.name
            spec = CameraSpec.spec_by_name()[cam]
            self.uvc_capture.bandwidth_factor = spec.bandwidth_factor
            logger.debug(f"Set {cam} bandwidth_factor to {spec.bandwidth_factor}")

        # UVC setting quirks:
        controls_dict = {c.display_name: c for c in self.uvc_capture.controls}

        if (
            ("Pupil Cam2" in self.uvc_capture.name)
            or ("Pupil Cam3" in self.uvc_capture.name)
        ) and frame_size == (320, 240):
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

        elif (
            "Pupil Cam2" in self.uvc_capture.name
            or "Pupil Cam3" in self.uvc_capture.name
        ):
            if self.exposure_mode == "auto":
                # special settings apply to both, Pupil Cam2 and Cam3
                special_settings = {200: 28, 180: 31}
                controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
                self.preferred_exposure_time = Exposure_Time(
                    max_ET=special_settings.get(self.frame_rate, 32),
                    frame_rate=self.frame_rate,
                    mode=self.exposure_mode,
                )

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

        elif self.uvc_capture.name == SCENE_CAM_SPEC.name:
            controls = {
                "Backlight Compensation": 2,
                "Brightness": 0,
                "Contrast": 32,
                "Gain": 64,
                "Hue": 0,
                "Saturation": 64,
                "Sharpness": 50,
                "Gamma": 300,
                "White Balance temperature": 4600,
            }
            for key, value in controls.items():
                try:
                    controls_dict[key].value = value
                except KeyError:
                    logger.debug(
                        f"Setting {key} to {value} failed: Unknown control. Known "
                        f"controls: {list(controls_dict)}"
                    )

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
                logger.debug(f'No UVC setting "{c.display_name}" found from settings.')

        if self.should_check_stripes:
            self.stripe_detector = Check_Frame_Stripes()

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
                    logger.info(f"Found device. {name}.")
                    if self.uvc_capture:
                        self._re_init_capture(d["uid"])
                    else:
                        self._init_capture(d["uid"], backup_uvc_controls)
                    return
        raise InitialisationError(
            f"Could not find Camera {names} during re initilization."
        )

    def _restart_logic(self):
        if self._restart_in <= 0:
            if self.uvc_capture:
                logger.warning("Camera disconnected. Reconnecting...")
                self.name_backup = (self.uvc_capture.name,)
                self.backup_uvc_controls = self._get_uvc_controls()
                self.uvc_capture = None
            try:
                self._re_init_capture_by_names(
                    self.name_backup, self.backup_uvc_controls
                )
            except (InitialisationError, uvc.InitError):
                time.sleep(0.02)
            self._restart_in = int(5 / 0.02)
        else:
            self._restart_in -= 1

    def recent_events(self, events):
        was_online = self.online

        try:
            frame = self.uvc_capture.get_frame(0.05)

            if np.isclose(frame.timestamp, 0):
                # sometimes (probably only on windows) after disconnections, the first frame has 0 ts
                logger.debug(
                    "Received frame with invalid timestamp."
                    " This can happen after a disconnect."
                    " Frame will be dropped!"
                )
                return

            if self.preferred_exposure_time:
                target = self.preferred_exposure_time.calculate_based_on_frame(frame)
                if target is not None:
                    self.exposure_time = target

            if self.stripe_detector and self.stripe_detector.require_restart(frame):
                # set the self.frame_rate in order to restart
                self.frame_rate = self.frame_rate
                logger.debug("Stripes detected")

        except (uvc.StreamError, TimeoutError):
            self._recent_frame = None
            self._restart_logic()
        except (AttributeError, uvc.InitError):
            self._recent_frame = None
            time.sleep(0.02)
            self._restart_logic()
        else:
            if self.ts_offset is not None:
                # c930 timestamps need to be set here. The camera does not provide valid pts from device
                frame.timestamp = uvc.get_time_monotonic() + self.ts_offset

            if self._last_ts is not None and frame.timestamp <= self._last_ts:
                logger.debug(
                    "Received non-monotonic timestamps from UVC! Dropping frame."
                    f" Last: {self._last_ts}, current: {frame.timestamp}"
                )
            else:
                self._last_ts = frame.timestamp
                frame.timestamp -= self.g_pool.timebase.value
                self._recent_frame = frame
                events["frame"] = frame
            self._restart_in = 3

        if was_online != self.online:
            self.update_menu()

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
        d["check_stripes"] = self.enable_stripe_checks
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
            return "(disconnected)"

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
                f"{new_size} resolution capture mode not available. Selected {size}."
            )
        self.uvc_capture.frame_size = size
        self.frame_size_backup = size

        self._intrinsics = Camera_Model.from_file(
            self.g_pool.user_dir, self.name, self.frame_size
        )

        if self.should_check_stripes:
            self.stripe_detector = Check_Frame_Stripes()

    @property
    def bandwidth_factor(self) -> float:
        return self.uvc_capture.bandwidth_factor if self.uvc_capture else float("nan")

    @bandwidth_factor.setter
    def bandwidth_factor(self, value: float):
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as err:
                logger.debug("err")
                return
        if self.uvc_capture and not np.isnan(value):
            self.uvc_capture.bandwidth_factor = value

    @property
    def should_check_stripes(self):
        return self.enable_stripe_checks and ("Pupil Cam2" in self.uvc_capture.name)

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
                f"{new_rate} fps capture mode not available at "
                f"{self.uvc_capture.frame_size} on {self.uvc_capture.name!r}. "
                f"Selected {rate} fps."
            )
        self.uvc_capture.frame_rate = rate
        self.frame_rate_backup = rate

        if (
            "Pupil Cam2" in self.uvc_capture.name
            or "Pupil Cam3" in self.uvc_capture.name
        ):
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

        if self.should_check_stripes:
            self.stripe_detector = Check_Frame_Stripes()

    @property
    def exposure_time(self):
        if self.uvc_capture:
            try:
                controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
                return controls_dict["Absolute Exposure Time"].value
            except KeyError:
                return None
        else:
            return self.exposure_time_backup

    @exposure_time.setter
    def exposure_time(self, new_et):
        try:
            controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
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

    def ui_elements(self):
        ui_elements = []

        if self.uvc_capture is None:
            ui_elements.append(ui.Info_Text("Local USB: camera disconnected!"))
            return ui_elements

        ui_elements.append(ui.Info_Text(f"Camera: {self.name} @ Local USB"))

        # lets define some  helper functions:
        def gui_load_defaults():
            for c in self.uvc_capture.controls:
                try:
                    c.value = c.def_val
                except Exception:
                    pass

        def gui_update_from_device():
            for c in self.uvc_capture.controls:
                c.refresh()

        def set_frame_size(new_size):
            self.frame_size = new_size

        def set_frame_rate(new_rate):
            self.frame_rate = new_rate
            self.update_menu()

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

        # TODO: potential race condition through selection_getter. Should ensure that
        # current selection will always be present in the list returned by the
        # selection_getter. Highly unlikely though as this needs to happen between
        # having clicked the Selector and the next redraw.
        # See https://github.com/pupil-labs/pyglui/pull/112/commits/587818e9556f14bfedd8ff8d093107358745c29b
        sensor_control.append(
            ui.Selector(
                "frame_rate",
                self,
                selection_getter=frame_rate_getter,
                setter=set_frame_rate,
                label="Frame rate",
            )
        )

        if (
            "Pupil Cam2" in self.uvc_capture.name
            or "Pupil Cam3" in self.uvc_capture.name
        ):
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
                    f"Exposure mode for camera {self.uvc_capture.name} is now set to "
                    f"{exposure_mode} mode"
                )
                self.update_menu()

            sensor_control.append(
                ui.Selector(
                    "exposure_mode",
                    self,
                    setter=set_exposure_mode,
                    selection=["manual", "auto"],
                    labels=["manual mode", "auto mode"],
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

        if (
            "Pupil Cam2" in self.uvc_capture.name
            or "Pupil Cam3" in self.uvc_capture.name
        ):
            blacklist += [
                "Auto Exposure Mode",
                "Auto Exposure Priority",
                "Absolute Exposure Time",
            ]

        sensor_control.append(
            ui.Text_Input(
                "bandwidth_factor",
                self,
                label="Bandwidth factor",
                getter=lambda: f"{self.bandwidth_factor:.2f}",
            )
        )

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

            def set_check_stripes(enable_stripe_checks):
                self.enable_stripe_checks = enable_stripe_checks
                if self.enable_stripe_checks:
                    self.stripe_detector = Check_Frame_Stripes()
                    logger.debug(
                        f"Check Stripes for camera {self.uvc_capture.name} is now on"
                    )
                else:
                    self.stripe_detector = None
                    logger.debug(
                        f"Check Stripes for camera {self.uvc_capture.name} is now off"
                    )

            ui_elements.append(
                ui.Switch(
                    "enable_stripe_checks",
                    self,
                    setter=set_check_stripes,
                    label="Check Stripes",
                )
            )

        return ui_elements

    def cleanup(self):
        self.devices.cleanup()
        self.devices = None
        if self.uvc_capture:
            self.uvc_capture.close()
            self.uvc_capture = None
        super().cleanup()

    def gl_display(self):
        # Temporary copy of Base_Source.gl_display until proper frame class hierarchy
        # is implemented
        if self._recent_frame is not None:
            frame = self._recent_frame
            if (
                # `frame.yuv_subsampling` is `None` without calling `frame.yuv_buffer`
                frame.yuv_buffer is not None
                and TJSAMP(frame.yuv_subsampling) == TJSAMP.TJSAMP_422
                # TODO: Find a better solution than this:
                and getattr(self.g_pool, "display_mode", "") != "algorithm"
            ):
                self.g_pool.image_tex.update_from_yuv_buffer(
                    frame.yuv_buffer, frame.width, frame.height
                )
            else:
                self.g_pool.image_tex.update_from_ndarray(frame.bgr)
            gl_utils.glFlush()
        should_flip = getattr(self.g_pool, "flip", False)
        gl_utils.make_coord_system_norm_based(flip=should_flip)
        self.g_pool.image_tex.draw()
        if not self.online:
            cygl.utils.draw_gl_texture(np.zeros((1, 1, 3), dtype=np.uint8), alpha=0.4)
        gl_utils.make_coord_system_pixel_based(
            (self.frame_size[1], self.frame_size[0], 3), flip=should_flip
        )


class UVC_Manager(Base_Manager):
    """Manages local USB sources."""

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.devices = uvc.Device_List()
        self.cam_selection_lut = {
            "eye0": ["ID0"],
            "eye1": ["ID1"],
            "world": ["ID2", "Logitech"],
        }
        # Do not show RealSense cameras in selection, since they are not supported
        # anymore in Pupil Capture since v1.22 and won't work.
        self.ignore_name_patterns = ["RealSense"]

    def get_devices(self):
        self.devices.update()
        uvc_auto_selection_devices = [
            device
            for device in self.devices
            if not any(
                pattern in device["name"] for pattern in self.ignore_name_patterns
            )
            and "Neon" not in device["name"]  # Only ignore Neon in get_devices()
        ]
        if len(uvc_auto_selection_devices) == 0:
            return []
        else:
            return [SourceInfo(label="Local USB", manager=self, key="usb")]

    def get_cameras(self):
        self.devices.update()
        return [
            SourceInfo(
                label=f"{device['name']} @ Local USB",
                manager=self,
                key=f"cam.{device['uid']}",
            )
            for device in self.devices
            if not any(
                pattern in device["name"] for pattern in self.ignore_name_patterns
            )
        ]

    def activate(self, key):
        if key == "usb":
            self.notify_all({"subject": "backend.uvc.auto_activate_source"})
            return

        source_uid = key[4:]
        self.activate_source(source_uid)

    def activate_source(self, source_uid):
        if not source_uid:
            return

        try:
            if not uvc.is_accessible(source_uid):
                logger.error("The selected camera is already in use or blocked.")
                return
        except ValueError as ve:
            logger.error(str(ve))
            logger.debug(traceback.format_exc())
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
                    "subject": "start_eye_plugin",
                    "target": self.g_pool.process,
                    "name": "UVC_Source",
                    "args": settings,
                }
            )

    def on_notify(self, notification):
        """Starts appropriate UVC sources.

        Emits notifications:
            ``backend.uvc.auto_activate_source``: All UVC managers should auto activate a source
            ``start_(eye_)plugin``: Starts UVC sources

        Reacts to notifications:
            ``backend.uvc.auto_activate_source``: Auto activate best source for process
        """
        if notification["subject"] == "backend.uvc.auto_activate_source":
            self.auto_activate_source()

    def auto_activate_source(self):
        logger.debug("Auto activating USB source.")
        self.devices.update()
        if not self.devices or len(self.devices) == 0:
            logger.warning("No default device is available.")
            return

        name_patterns = self.cam_selection_lut[self.g_pool.process]
        matching_cams = [
            device
            for device in self.devices
            if any(pattern in device["name"] for pattern in name_patterns)
        ]

        if not matching_cams:
            logger.warning("Could not find default device.")
            return

        # Select device with highest priority (first device when sorting by priority)
        cam = min(matching_cams, key=self._auto_activate_priority(name_patterns))
        self.activate_source(cam["uid"])

    @staticmethod
    def _auto_activate_priority(pattern_priority):
        def __auto_activate_priority(device):
            least_priority = float("inf")
            # Sorting cameras primarily by name pattern order avoids selecting the wrong
            # camera if more than one pattern is matched.
            matched_pattern_index = next(
                (
                    idx
                    for idx, pat in enumerate(pattern_priority)
                    if pat in device["name"]
                ),
                least_priority,  # default, in case no pattern matched
            )
            # Sorting cams secondarily by bus_number increases chances of selecting only cams from the
            # same headset when having multiple headsets connected. Note that two headsets
            # might have the same bus_number when they share an internal USB bus.
            bus_number = device.get("bus_number", least_priority)
            return matched_pattern_index, bus_number

        return __auto_activate_priority

    def cleanup(self):
        self.devices.cleanup()
        self.devices = None
