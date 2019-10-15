from pupil_detectors import PupilDetector, Detector2D
from .detector_base_plugin import PupilDetectorPlugin

from pyglui import ui
from pyglui.cygl.utils import draw_gl_texture
import glfw
from gl_utils import (
    adjust_gl_view,
    clear_gl_screen,
    basic_gl_setup,
    make_coord_system_norm_based,
    make_coord_system_pixel_based,
)
from plugin import Plugin


class Detector2DPlugin(PupilDetectorPlugin):
    uniqueness = 'unique'
    icon_font = 'pupil_icons'
    icon_chr = chr(0xec18)

    def __init__(self, g_pool = None, namespaced_properties = None, detector_2d: Detector2D = None):
        super().__init__(g_pool=g_pool)
        self.detector_2d = detector_2d or Detector2D(namespaced_properties or {})
        #debug window
        self._window = None
        self.windowShouldOpen = False
        self.windowShouldClose = False

    ##### Core API

    @property
    def detector_properties_2d(self) -> dict:
        return self.detector_2d.detector_properties_2d

    def detect(self, frame, user_roi, visualize, pause_video = False):
        if self.windowShouldOpen:
            self.open_window((frame.width,frame.height))
        if self.windowShouldClose:
            self.close_window()

        return self.detector_2d.detect(
            frame,
            user_roi,
            visualize,
            pause_video,
            use_debug_image=self._window != None
        )

    ##### Legacy API

    def set_2d_detector_property(self, name, value):
        self.detector_2d.set_2d_detector_property(name, value)

    ##### Plugin API

    @property
    def pupil_detector(self) -> PupilDetector:
        return self.detector_2d

    @property
    def pretty_class_name(self):
        return 'Pupil Detector 2D'

    def init_ui(self):
        Plugin.add_menu(self)
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = 'pupil_icons'
        info = ui.Info_Text("Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video. "\
                                +"Adjust the pupil intensity range so that the pupil is fully overlaid with blue. "\
                                +"Adjust the pupil min and pupil max ranges (red circles) so that the detected pupil size (green circle) is within the bounds.")
        self.menu.append(info)
        #self.menu.append(ui.Switch('coarse_detection',self.detector_properties_2d,label='Use coarse detection'))
        self.menu.append(ui.Slider('intensity_range',self.detector_properties_2d,label='Pupil intensity range',min=0,max=60,step=1))
        self.menu.append(ui.Slider('pupil_size_min',self.detector_properties_2d,label='Pupil min',min=1,max=250,step=1))
        self.menu.append(ui.Slider('pupil_size_max',self.detector_properties_2d,label='Pupil max',min=50,max=400,step=1))
        self.menu.append(ui.Button('Open debug window',self.toggle_window))
        #advanced_controls_menu = ui.Growing_Menu('Advanced Controls')
        #advanced_controls_menu.append(ui.Slider('contour_size_min',self.detector_properties_2d,label='Contour min length',min=1,max=200,step=1))
        #advanced_controls_menu.append(ui.Slider('ellipse_true_support_min_dist',self.detector_properties_2d,label='ellipse_true_support_min_dist',min=0.1,max=7,step=0.1))
        #self.menu.append(advanced_controls_menu)

    def deinit_ui(self):
        Plugin.remove_menu(self)

    def toggle_window(self):
        if self._window:
            self.windowShouldClose = True
        else:
            self.windowShouldOpen = True

    def open_window(self,size):
        if not self._window:
            if 0: #we are not fullscreening
                monitor = glfw.glfwGetMonitors()[self.monitor_idx]
                mode = glfw.glfwGetVideoMode(monitor)
                width, height= mode[0],mode[1]
            else:
                monitor = None
                width, height = size

            active_window = glfw.glfwGetCurrentContext()
            self._window = glfw.glfwCreateWindow(width, height, "Pupil Detector Debug Window", monitor=monitor, share=active_window)
            if not 0:
                glfw.glfwSetWindowPos(self._window,200,0)

            self.on_resize(self._window,width, height)

            #Register callbacks
            glfw.glfwSetWindowSizeCallback(self._window,self.on_resize)
            # glfwSetKeyCallback(self._window,self.on_window_key)
            glfw.glfwSetWindowCloseCallback(self._window,self.on_close)

            # gl_state settings
            glfw.glfwMakeContextCurrent(self._window)
            basic_gl_setup()

            # refresh speed settings
            glfw.glfwSwapInterval(0)

            glfw.glfwMakeContextCurrent(active_window)

            self.windowShouldOpen = False

    # window calbacks
    def on_resize(self,window,w,h):
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        adjust_gl_view(w,h)
        glfw.glfwMakeContextCurrent(active_window)

    def on_close(self,window):
        self.windowShouldClose = True

    def close_window(self):
        if self._window:
            active_window = glfw.glfwGetCurrentContext()
            glfw.glfwDestroyWindow(self._window)
            self._window = None
            self.windowShouldClose = False
            glfw.glfwMakeContextCurrent(active_window)

    def gl_display_in_window(self,img):
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(self._window)
        clear_gl_screen()
        # gl stuff that will show on your plugin window goes here
        make_coord_system_norm_based()
        draw_gl_texture(img,interpolation=False)
        glfw.glfwSwapBuffers(self._window)
        glfw.glfwMakeContextCurrent(active_window)

    def cleanup(self):
        self.close_window() # if we change detectors, be sure debug window is also closed

    def gl_display(self):
        #display the debug image in the window
        if self._window:
            self.gl_display_in_window(self.detector_2d.debug_image)
