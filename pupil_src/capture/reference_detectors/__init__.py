'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

# make shared modules available across pupil_src
if __name__ == '__main__':
    from sys import path as syspath
    from os import path as ospath
    loc = ospath.abspath(__file__).rsplit('pupil_src', 1)
    syspath.append(ospath.join(loc[0], 'pupil_src', 'shared_modules'))
    del syspath, ospath


# import detector classes from sibling files
from screen_marker_calibration import Screen_Marker_Calibration
from manual_marker_calibration import Manual_Marker_Calibration
from natural_features_calibration import Natural_Features_Calibration
from camera_intrinsics_estimation import Camera_Intrinsics_Estimation 


name_by_index = [   'Screen Marker Calibration',
                    'Manual Marker Calibration',
                    'Natural Features Calibration',
                    'Camera Intrinsics Estimation']

detector_by_index = [   Screen_Marker_Calibration,
                        Manual_Marker_Calibration,
                        Natural_Features_Calibration,
                        Camera_Intrinsics_Estimation]

index_by_name = dict(zip(name_by_index,range(len(name_by_index))))
detector_by_name = dict(zip(name_by_index,detector_by_index))


if __name__ == '__main__':

    active_detector_class = Animated_Nine_Point_Detector

    from glfw import *
    import atb
    from methods import normalize, denormalize, chessboard, circle_grid, gen_pattern_grid, calibrate_camera,Temp
    from uvc_capture import autoCreateCapture
    from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen,draw_gl_point,draw_gl_point_norm,draw_gl_polyline_norm
    from time import time
    from ctypes import c_bool, c_float
    from multiprocessing import Array

    # Callback functions
    def on_resize(w, h):
        atb.TwWindowSize(w, h);
        adjust_gl_view(w,h)

    def on_key(key, pressed):
        if not atb.TwEventKeyboardGLFW(key,pressed):
            if pressed:
                if key == GLFW_KEY_ESC:
                    on_close()

    def on_char(char, pressed):
        if not atb.TwEventCharGLFW(char,pressed):
            pass

    def on_button(button, pressed):
        if not atb.TwEventMouseButtonGLFW(button,pressed):
            if pressed:
                pos = glfwGetMousePos()
                pos = normalize(pos,glfwGetWindowSize())
                pos = denormalize(pos,(img.shape[1],img.shape[0]) ) # Position in img pixels
                ref.detector.new_ref(pos)

    def on_pos(x, y):
        if atb.TwMouseMotion(x,y):
            pass

    def on_scroll(pos):
        if not atb.TwMouseWheel(pos):
            pass

    def on_close():
        running.value=False

    running = c_bool(1)


    # Initialize capture, check if it works
    cap = autoCreateCapture(["Logitech Camera", "C525","C615","C920","C930e"],(1280,720))
    if cap is None:
        print "WORLD: Error could not create Capture"

    s, img = cap.read()
    if not s:
        print "WORLD: Error could not get image"

    height,width = img.shape[:2]

    # helpers called by the main atb bar
    def update_fps():
        old_time, bar.timestamp = bar.timestamp, time()
        dt = bar.timestamp - old_time
        if dt:
            bar.fps.value += .05 * (1 / dt - bar.fps.value)

    def set_window_size(mode,data):
        height,width = img.shape[:2]
        ratio = (1,.75,.5,.25)[mode]
        w,h = int(width*ratio),int(height*ratio)
        glfwSetWindowSize(w,h)
        data.value=mode # update the bar.value

    def get_from_data(data):
        """
        helper for atb getter and setter use
        """
        return data.value


    # Initialize ant tweak bar - inherits from atb.Bar
    atb.init()
    bar = atb.Bar(name = "World", label="Controls",
            help="Scene controls", color=(50, 50, 50), alpha=100,valueswidth=150,
            text='light', position=(10, 10),refresh=.3, size=(300, 200))
    bar.fps = c_float(0.0)
    bar.timestamp = time()
    bar.window_size = c_int(0)
    window_size_enum = atb.enum("Display Size",{"Full":0, "Medium":1,"Half":2,"Mini":3})

    # play and record can be tied together via pointers to the objects
    # bar.play = bar.record_video
    bar.add_var("FPS", bar.fps, step=1., readonly=True)
    bar.add_var("Display_Size", vtype=window_size_enum,setter=set_window_size,getter=get_from_data,data=bar.window_size)

    # add v4l2 camera controls to a seperate ATB bar
    if cap.controls is not None:
        c_bar = atb.Bar(name="Camera_Controls", label=cap.name,
            help="UVC Camera Controls", color=(50,50,50), alpha=100,
            text='light',position=(320, 10),refresh=2., size=(200, 200))

        sorted_controls = [c for c in cap.controls.itervalues()]
        sorted_controls.sort(key=lambda c: c.order)

        for control in sorted_controls:
            name = control.atb_name
            if control.type=="bool":
                c_bar.add_var(name,vtype=atb.TW_TYPE_BOOL8,getter=control.get_val,setter=control.set_val)
            elif control.type=='int':
                c_bar.add_var(name,vtype=atb.TW_TYPE_INT32,getter=control.get_val,setter=control.set_val)
                c_bar.define(definition='min='+str(control.min),   varname=name)
                c_bar.define(definition='max='+str(control.max),   varname=name)
                c_bar.define(definition='step='+str(control.step), varname=name)
            elif control.type=="menu":
                if control.menu is None:
                    vtype = None
                else:
                    vtype= atb.enum(name,control.menu)
                c_bar.add_var(name,vtype=vtype,getter=control.get_val,setter=control.set_val)
                if control.menu is None:
                    c_bar.define(definition='min='+str(control.min),   varname=name)
                    c_bar.define(definition='max='+str(control.max),   varname=name)
                    c_bar.define(definition='step='+str(control.step), varname=name)
            else:
                pass
            if control.flags == "inactive":
                pass
                # c_bar.define(definition='readonly=1',varname=control.name)

        c_bar.add_button("refresh",cap.update_from_device)
        c_bar.add_button("load defaults",cap.load_defaults)

    else:
        c_bar = None


    ref = Temp()
    g_calibrate, g_ref,s_ref,marker = c_bool(0), Array('d',(0,0)), Array('d',(0,0)), c_int(0)
    ref.detector = active_detector_class(g_calibrate,g_ref, s_ref,marker,(10,230))
    # Objects as variable containers

    # Initialize glfw
    glfwInit()
    height,width = img.shape[:2]
    glfwOpenWindow(width, height, 0, 0, 0, 8, 0, 0, GLFW_WINDOW)
    glfwSetWindowTitle("Ref Detector Test")
    glfwSetWindowPos(0,0)

    # Register callbacks
    glfwSetWindowSizeCallback(on_resize)
    glfwSetWindowCloseCallback(on_close)
    glfwSetKeyCallback(on_key)
    glfwSetCharCallback(on_char)
    glfwSetMouseButtonCallback(on_button)
    glfwSetMousePosCallback(on_pos)
    glfwSetMouseWheelCallback(on_scroll)

    # gl_state settings
    import OpenGL.GL as gl
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_BLEND)
    del gl

    # Event loop
    while glfwGetWindowParam(GLFW_OPENED) and running.value:
        update_fps()

        # Get an image from the grabber
        s, img = cap.read()
        ref.detector.update(img)

        # render the screen
        clear_gl_screen()
        draw_gl_texture(img)

        ref.detector.gl_display()

        atb.draw()
        glfwSwapBuffers()

    # end while running and clean-up
    print "Process closed"
    glfwCloseWindow()
    glfwTerminate()
