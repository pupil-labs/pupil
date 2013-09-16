'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import os
from ctypes import c_int,c_bool,c_float
import numpy as np
import atb
from glfw import *
from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen, draw_gl_point_norm, draw_gl_polyline
from time import time, sleep
from methods import *
from c_methods import eye_filter
from uvc_capture import autoCreateCapture
from calibrate import get_map_from_cloud
from pupil_detectors import Canny_Detector
import shelve

def eye(g_pool):
    """
    this needs a docstring
    """

 # Callback functions World
    def on_resize(window,w, h):
        adjust_gl_view(w,h)
        atb.TwWindowSize(w, h)


    def on_key(window, key, scancode, action, mods):
        if not atb.TwEventKeyboardGLFW(key,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS:
                if key == GLFW_KEY_ESCAPE:
                    on_close(window)

    def on_char(window,char):
        if not atb.TwEventCharGLFW(char,1):
            pass

    def on_button(window,button, action, mods):
        if not atb.TwEventMouseButtonGLFW(button,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS:
                pos = glfwGetCursorPos(window)
                pos = normalize(pos,glfwGetWindowSize(window))
                pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # pos in frame.img pixels
                u_r.setStart(pos)
                bar.draw_roi.value = 1
            else:
                bar.draw_roi.value = 0

    def on_pos(window,x, y):
        if atb.TwMouseMotion(int(x),int(y)):
            pass
        if bar.draw_roi.value == 1:
            pos = x,y
            pos = normalize(pos,glfwGetWindowSize(window))
            pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # pos in frame.img pixels
            u_r.setEnd(pos)

    def on_scroll(window,x,y):
        if not atb.TwMouseWheel(int(x)):
            pass

    def on_close(window):
        g_pool.quit.value = True
        print "WORLD Process closing from window"





    # Helper functions called by the main atb bar
    def start_roi():
        bar.display.value = 1
        bar.draw_roi.value = 2

    def update_fps():
        old_time, bar.timestamp = bar.timestamp, time()
        dt = bar.timestamp - old_time
        if dt:
            bar.fps.value += .05 * (1 / dt - bar.fps.value)
            bar.dt.value = dt

    def get_from_data(data):
        """
        helper for atb getter and setter use
        """
        return data.value


    # load session persistent settings
    session_settings = shelve.open('user_settings_eye',protocol=2)
    def load(var_name,default):
        try:
            return session_settings[var_name]
        except:
            return default
    def save(var_name,var):
        session_settings[var_name] = var

    # Initialize capture
    cap = autoCreateCapture(g_pool.eye_src, g_pool.eye_size)
    if cap is None:
        print "EYE: Error could not create Capture"
        return
    # check if it works
    frame = cap.get_frame()
    if frame.img is None:
        print "EYE: Error could not get image"
        return
    height,width = frame.img.shape[:2]


    u_r = Roi(frame.img.shape)
    u_r.set(load('roi',default=None))
    p_r = Roi(frame.img.shape)

    writer = None

    pupil_detector = Canny_Detector()

    atb.init()
    # Create main ATB Controls
    bar = atb.Bar(name = "Eye", label="Display",
            help="Scene controls", color=(50, 50, 50), alpha=100,
            text='light', position=(10, 10),refresh=.3, size=(200, 100))
    bar.fps = c_float(0.0)
    bar.timestamp = time()
    bar.dt = c_float(0.0)
    bar.sleep = c_float(0.0)
    bar.display = c_int(load('bar.display',0))
    bar.draw_pupil = c_bool(load('bar.draw_pupil',True))
    bar.draw_roi = c_int(0)
    bar.record_eye = c_bool(load('bar.record_eye',0))

    dispay_mode_enum = atb.enum("Mode",{"Camera Image":0,
                                        "Region of Interest":1,
                                        "Algorithm":2,
                                        "Corse Pupil Region":3})

    bar.add_var("FPS",bar.fps, step=1.,readonly=True)
    bar.add_var("Mode", bar.display,vtype=dispay_mode_enum, help="select the view-mode")
    bar.add_var("Show_Pupil_Point", bar.draw_pupil)
    bar.add_button("Draw_ROI", start_roi, help="drag on screen to select a region of interest")

    bar.add_var("record_eye_video", bar.record_eye, help="when recording also save the eye video stream")
    bar.add_var("SlowDown",bar.sleep, step=0.01,min=0.0)
    bar.add_var("SaveSettings&Exit", g_pool.quit)

    cap.create_atb_bar(pos=(220,10))

    # create a bar for the detector
    pupil_detector.create_atb_bar(pos=(10,120))


    glfwInit()
    window = glfwCreateWindow(width, height, "Eye", None, None)
    glfwSetWindowPos(window,800,0)
    on_resize(window,width,height)
    #set the last saved window size



    # Register callbacks window
    glfwSetWindowSizeCallback(window,on_resize)
    glfwSetWindowCloseCallback(window,on_close)
    glfwSetKeyCallback(window,on_key)
    glfwSetCharCallback(window,on_char)
    glfwSetMouseButtonCallback(window,on_button)
    glfwSetCursorPosCallback(window,on_pos)
    glfwSetScrollCallback(window,on_scroll)

    glfwMakeContextCurrent(window)

    # gl_state settings
    import OpenGL.GL as gl
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_BLEND)
    del gl

    # event loop
    while not glfwWindowShouldClose(window) and not g_pool.quit.value:
        frame = cap.get_frame()
        update_fps()
        sleep(bar.sleep.value) # for debugging only

        # IMAGE PROCESSING and clipping to user defined eye-region
        eye_img = frame.img[u_r.lY:u_r.uY,u_r.lX:u_r.uX]
        gray_img = grayscale(eye_img)


        # coarse pupil detection
        integral = cv2.integral(gray_img)
        integral =  np.array(integral,dtype=c_float)
        x,y,w = eye_filter(integral)
        if w>0:
            p_r.set((y,x,y+w,x+w))
        else:
            p_r.set((0,0,-1,-1))

        # fine pupil ellipse detection
        result = pupil_detector.detect(frame,u_roi=u_r,p_roi=p_r,visualize=bar.display.value == 2)

        g_pool.pupil_queue.put(result)
        # Work with detected ellipses


        ### RECORDING of Eye Video ###
        # Setup variables and lists for recording
        if g_pool.eye_rx.poll():
            command = g_pool.eye_rx.recv()
            if command is not None:
                record_path = command
                print "INFO: Will save eye video to: ", record_path
                video_path = os.path.join(record_path, "eye.avi")
                timestamps_path = os.path.join(record_path, "eye_timestamps.npy")
                writer = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), bar.fps.value, (frame.img.shape[1], frame.img.shape[0]))
                timestamps = []
            else:
                print "INFO: Done recording eye."
                writer = None
                np.save(timestamps_path,np.asarray(timestamps))
                del timestamps

        if writer:
            writer.write(frame.img)
            timestamps.append(frame.timestamp)


        # direct visualizations on the frame.img data
        if bar.display.value == 1:
            # and a solid (white) frame around the user defined ROI
            gray_img[:,0] = 255
            gray_img[:,-1]= 255
            gray_img[0,:] = 255
            gray_img[-1,:]= 255
            frame.img[u_r.lY:u_r.uY,u_r.lX:u_r.uX] = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

            pupil_img =frame.img[u_r.lY:u_r.uY,u_r.lX:u_r.uX][p_r.lY:p_r.uY,p_r.lX:p_r.uX] # create an RGB view onto the gray pupil ROI
            # draw a frame around the automatic pupil ROI in overlay...
            pupil_img[::2,0] = 255,0,0
            pupil_img[::2,-1]= 255,0,0
            pupil_img[0,::2] = 255,0,0
            pupil_img[-1,::2]= 255,0,0

            frame.img[u_r.lY:u_r.uY,u_r.lX:u_r.uX][p_r.lY:p_r.uY,p_r.lX:p_r.uX] = pupil_img

        elif bar.display.value == 3:
            frame.img = frame.img[u_r.lY:u_r.uY,u_r.lX:u_r.uX][p_r.lY:p_r.uY,p_r.lX:p_r.uX]

        # GL-drawing
        clear_gl_screen()
        draw_gl_texture(frame.img)

        if result['norm_pupil'] is not None:
            pts = cv2.ellipse2Poly( (int(result['center'][0]),int(result['center'][1])),
                                    (int(result["axes"][0]/2),int(result["axes"][1]/2)),
                                    int(result["angle"]),0,360,15)
            draw_gl_polyline(pts,(1.,0,0,.5))
            draw_gl_point_norm(result['norm_pupil'],color=(1.,0.,0.,0.5))

        atb.draw()
        glfwSwapBuffers(window)
        glfwPollEvents()

    # END while running

    # Quite while Recording: Save values
    if writer:
        print "INFO: Done recording eye"
        writer = None
        np.save(timestamps_path,np.asarray(timestamps))
        del timestamps



    # save session persistent settings
    save('roi',u_r.get())
    save('bar.display',bar.display.value)
    save('bar.draw_pupil',bar.draw_pupil.value)
    save('bar.record_eye',bar.record_eye.value)
    session_settings.close()
    cap.close()
    atb.terminate()
    glfwDestroyWindow(window)
    glfwTerminate()

    print "EYE Process closed"

def eye_profiled(g_pool):
    import cProfile,subprocess,os
    from eye import eye
    cProfile.runctx("eye(g_pool,)",{"g_pool":g_pool},locals(),"eye.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats eye.pstats | dot -Tpng -o eye_cpu_time.png", shell=True)
    print "created cpu time graph for eye process. Please check out the png next to the eye.py file"

