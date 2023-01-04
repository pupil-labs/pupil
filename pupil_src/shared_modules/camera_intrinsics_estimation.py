"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import cv2
import gl_utils
import glfw
import numpy as np
import OpenGL.GL as gl
from camera_models import Fisheye_Dist_Camera, Radial_Dist_Camera
from gl_utils import (
    GLFWErrorReporting,
    adjust_gl_view,
    basic_gl_setup,
    clear_gl_screen,
    draw_circle_filled_func_builder,
    make_coord_system_norm_based,
)
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_gl_texture, draw_polyline
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

GLFWErrorReporting.set_default()

# logging
import logging

from hotkey import Hotkey
from plugin import Plugin

logger = logging.getLogger(__name__)


# window calbacks
def on_resize(window, w, h):
    active_window = glfw.get_current_context()
    glfw.make_context_current(window)
    adjust_gl_view(w, h)
    glfw.make_context_current(active_window)


class Camera_Intrinsics_Estimation(Plugin):
    """Camera_Intrinsics_Calibration
    This method is not a gaze calibration.
    This method is used to calculate camera intrinsics.
    """

    icon_chr = chr(0xEC06)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, fullscreen=False, monitor_idx=0):
        super().__init__(g_pool)
        self.collect_new = False
        self.calculated = False
        self.obj_grid = _gen_pattern_grid((4, 11))
        self.img_points = []
        self.obj_points = []
        self.count = 10
        self.display_grid = _make_grid()

        self._window = None

        self.menu = None
        self.button = None
        self.clicks_to_close = 5
        self.window_should_close = False
        self.monitor_idx = monitor_idx
        self.fullscreen = fullscreen
        self.dist_mode = "Fisheye"

        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", get_opensans_font_path())
        self.glfont.set_size(32)
        self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))
        self.glfont.set_align_string(v_align="center")

        self.undist_img = None
        self.show_undistortion = False
        self.show_undistortion_switch = None

        if (
            hasattr(self.g_pool.capture, "intrinsics")
            and self.g_pool.capture.intrinsics
        ):
            logger.info(
                "Click show undistortion to verify camera intrinsics calibration."
            )
            logger.info(
                "Hint: Straight lines in the real world should be straigt in the image."
            )
        else:
            logger.info(
                "No camera intrinsics calibration is currently set for this camera!"
            )

        self._draw_circle_filled = draw_circle_filled_func_builder()

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Camera Intrinsics Estimation"

        def get_monitors_idx_list():
            monitors = [glfw.get_monitor_name(m) for m in glfw.get_monitors()]
            return range(len(monitors)), monitors

        if self.monitor_idx not in get_monitors_idx_list()[0]:
            logger.warning(
                f"Monitor at index {self.monitor_idx} no longer availalbe. "
                "Using default instead."
            )
            self.monitor_idx = 0

        self.menu.append(
            ui.Info_Text(
                "Estimate Camera intrinsics of the world camera. Using an 11x9 asymmetrical circle grid. Click 'i' to capture a pattern."
            )
        )

        self.menu.append(ui.Button("show Pattern", self.open_window))
        self.menu.append(
            # TODO: potential race condition through selection_getter. Should ensure
            # that current selection will always be present in the list returned by the
            # selection_getter. Highly unlikely though as this needs to happen between
            # having clicked the Selector and the next redraw.
            # See https://github.com/pupil-labs/pyglui/pull/112/commits/587818e9556f14bfedd8ff8d093107358745c29b
            ui.Selector(
                "monitor_idx",
                self,
                selection_getter=get_monitors_idx_list,
                label="Monitor",
            )
        )
        dist_modes = ["Fisheye", "Radial"]
        self.menu.append(
            ui.Selector(
                "dist_mode", self, selection=dist_modes, label="Distortion Model"
            )
        )
        self.menu.append(ui.Switch("fullscreen", self, label="Use Fullscreen"))
        self.show_undistortion_switch = ui.Switch(
            "show_undistortion", self, label="show undistorted image"
        )
        self.menu.append(self.show_undistortion_switch)
        self.show_undistortion_switch.read_only = not (
            hasattr(self.g_pool.capture, "intrinsics")
            and self.g_pool.capture.intrinsics
        )

        self.button = ui.Thumb(
            "collect_new",
            self,
            setter=self.advance,
            label="I",
            hotkey=Hotkey.CAMERA_INTRINSIC_ESTIMATOR_COLLECT_NEW_CAPTURE_HOTKEY(),
        )
        self.button.on_color[:] = (0.3, 0.2, 1.0, 0.9)
        self.g_pool.quickbar.insert(0, self.button)

    def deinit_ui(self):
        self.remove_menu()
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None

    def do_open(self):
        if not self._window:
            self.window_should_open = True

    def get_count(self):
        return self.count

    def advance(self, _):
        if self.count == 10:
            logger.info("Capture 10 calibration patterns.")
            self.button.status_text = f"{self.count:d} to go"
            self.calculated = False
            self.img_points = []
            self.obj_points = []

        self.collect_new = True

    def open_window(self):
        if not self._window:
            if self.fullscreen:
                try:
                    monitor = glfw.get_monitors()[self.monitor_idx]
                except Exception:
                    logger.warning(
                        "Monitor at index %s no longer availalbe using default" % idx
                    )
                    self.monitor_idx = 0
                    monitor = glfw.get_monitors()[self.monitor_idx]
                mode = glfw.get_video_mode(monitor)
                height, width = mode.size.height, mode.size.width
            else:
                monitor = None
                height, width = 640, 480

            self._window = glfw.create_window(
                height,
                width,
                "Calibration",
                monitor,
                glfw.get_current_context(),
            )
            if not self.fullscreen:
                # move to y = 31 for windows os
                glfw.set_window_pos(self._window, 200, 31)

            # Register callbacks
            glfw.set_framebuffer_size_callback(self._window, on_resize)
            glfw.set_key_callback(self._window, self.on_window_key)
            glfw.set_window_close_callback(self._window, self.on_close)
            glfw.set_mouse_button_callback(self._window, self.on_window_mouse_button)

            on_resize(self._window, *glfw.get_framebuffer_size(self._window))

            # gl_state settings
            active_window = glfw.get_current_context()
            glfw.make_context_current(self._window)
            basic_gl_setup()
            glfw.make_context_current(active_window)

            self.clicks_to_close = 5

    def on_window_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                self.on_close()

    def on_window_mouse_button(self, window, button, action, mods):
        if action == glfw.PRESS:
            self.clicks_to_close -= 1
        if self.clicks_to_close == 0:
            self.on_close()

    def on_close(self, window=None):
        self.window_should_close = True

    def close_window(self):
        self.window_should_close = False
        if self._window:
            glfw.destroy_window(self._window)
            self._window = None

    def calculate(self):
        self.calculated = True
        self.count = 10
        img_shape = self.g_pool.capture.frame_size

        # Compute calibration
        try:
            if self.dist_mode == "Fisheye":
                calibration_flags = (
                    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                    + cv2.fisheye.CALIB_CHECK_COND
                    + cv2.fisheye.CALIB_FIX_SKEW
                )
                max_iter = 30
                eps = 1e-6
                camera_matrix = np.zeros((3, 3))
                dist_coefs = np.zeros((4, 1))
                rvecs = [
                    np.zeros((1, 1, 3), dtype=np.float64) for i in range(self.count)
                ]
                tvecs = [
                    np.zeros((1, 1, 3), dtype=np.float64) for i in range(self.count)
                ]
                objPoints = [x.reshape(1, -1, 3) for x in self.obj_points]
                imgPoints = self.img_points
                rms, _, _, _, _ = cv2.fisheye.calibrate(
                    objPoints,
                    imgPoints,
                    img_shape,
                    camera_matrix,
                    dist_coefs,
                    rvecs,
                    tvecs,
                    calibration_flags,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps),
                )
                camera_model = Fisheye_Dist_Camera(
                    self.g_pool.capture.name, img_shape, camera_matrix, dist_coefs
                )
            elif self.dist_mode == "Radial":
                rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
                    np.array(self.obj_points),
                    np.array(self.img_points),
                    self.g_pool.capture.frame_size,
                    None,
                    None,
                )
                camera_model = Radial_Dist_Camera(
                    self.g_pool.capture.name, img_shape, camera_matrix, dist_coefs
                )
            else:
                raise ValueError(f"Unkown distortion model: {self.dist_mode}")
        except ValueError as e:
            raise e
        except Exception as e:
            logger.warning("Camera calibration failed to converge!")
            logger.warning(
                "Please try again with a better coverage of the cameras FOV!"
            )
            return

        logger.info(f"Calibrated Camera, RMS:{rms}")

        camera_model.save(self.g_pool.user_dir)
        self.g_pool.capture.intrinsics = camera_model

        self.show_undistortion_switch.read_only = False

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return
        if self.collect_new:
            img = frame.img
            try:
                status, grid_points = cv2.findCirclesGrid(
                    img, (4, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID
                )
            except cv2.error:
                logger.exception(
                    f"Exception in cv2.findCirclesGrid() using shape={img.shape!r} "
                    f"dtype={img.dtype!r}"
                )
                return
            if status:
                self.img_points.append(grid_points)
                self.obj_points.append(self.obj_grid)
                self.collect_new = False
                self.count -= 1
                self.button.status_text = f"{self.count:d} to go"

        if self.count <= 0 and not self.calculated:
            self.calculate()
            self.button.status_text = ""

        if self.window_should_close:
            self.close_window()

        if self.show_undistortion:
            assert self.g_pool.capture.intrinsics
            # This function is not yet compatible with the fisheye camera model and would have to be manually implemented.
            # adjusted_k,roi = cv2.getOptimalNewCameraMatrix(cameraMatrix= np.array(self.camera_intrinsics[0]), distCoeffs=np.array(self.camera_intrinsics[1]), imageSize=self.camera_intrinsics[2], alpha=0.5,newImgSize=self.camera_intrinsics[2],centerPrincipalPoint=1)
            self.undist_img = self.g_pool.capture.intrinsics.undistort(frame.img)

    def gl_display(self):

        for grid_points in self.img_points:
            # we dont need that extra encapsulation that opencv likes so much
            calib_bounds = cv2.convexHull(grid_points)[:, 0]
            draw_polyline(
                calib_bounds, 1, RGBA(0.0, 0.0, 1.0, 0.5), line_type=gl.GL_LINE_LOOP
            )

        if self._window:
            self.gl_display_in_window()

        if self.show_undistortion and self.undist_img is not None:
            gl.glPushMatrix()
            make_coord_system_norm_based()
            draw_gl_texture(self.undist_img)
            gl.glPopMatrix()

    def gl_display_in_window(self):
        active_window = glfw.get_current_context()
        glfw.make_context_current(self._window)

        clear_gl_screen()

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        p_window_size = glfw.get_window_size(self._window)
        r = p_window_size[0] / 15.0
        # compensate for radius of marker
        gl.glOrtho(-r, p_window_size[0] + r, p_window_size[1] + r, -r, -1, 1)
        # Switch back to Model View Matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        # hacky way of scaling and fitting in different window rations/sizes
        grid = _make_grid() * min((p_window_size[0], p_window_size[1] * 5.5 / 4.0))
        # center the pattern
        grid -= np.mean(grid)
        grid += (p_window_size[0] / 2 - r, p_window_size[1] / 2 + r)

        for pt in grid:
            self._draw_circle_filled(
                tuple(pt),
                size=r / 2,
                color=RGBA(0.0, 0.0, 0.0, 1),
            )

        if self.clicks_to_close < 5:
            self.glfont.set_size(int(p_window_size[0] / 30.0))
            self.glfont.draw_text(
                p_window_size[0] / 2.0,
                p_window_size[1] / 4.0,
                f"Touch {self.clicks_to_close} more times to close window.",
            )

        glfw.swap_buffers(self._window)
        glfw.make_context_current(active_window)

    def get_init_dict(self):
        return {"monitor_idx": self.monitor_idx}

    def cleanup(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        if you have a gui or glfw window destroy it here.
        """
        if self._window:
            self.close_window()


def _gen_pattern_grid(size=(4, 11)):
    pattern_grid = []
    for i in range(size[1]):
        for j in range(size[0]):
            pattern_grid.append([(2 * j) + i % 2, i, 0])
    return np.asarray(pattern_grid, dtype="f4")


def _make_grid(dim=(11, 4)):
    """
    this function generates the structure for an asymmetrical circle grid
    domain (0-1)
    """
    x, y = range(dim[0]), range(dim[1])
    p = np.array([[[s, i] for s in x] for i in y], dtype=np.float32)
    p[:, 1::2, 1] += 0.5
    p = np.reshape(p, (-1, 2), "F")

    # scale height = 1
    x_scale = 1.0 / (np.amax(p[:, 0]) - np.amin(p[:, 0]))
    y_scale = 1.0 / (np.amax(p[:, 1]) - np.amin(p[:, 1]))

    p *= x_scale, x_scale / 0.5

    return p
