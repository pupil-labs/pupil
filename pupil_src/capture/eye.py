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
    # glfw callback functions
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
            if bar.draw_roi.value:
                if pressed:
                    pos = glfwGetMousePos()
                    pos = normalize(pos,glfwGetWindowSize())
                    pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # pos in frame.img pixels
                    u_r.setStart(pos)
                    bar.draw_roi.value = 1
                else:
                    bar.draw_roi.value = 0

    def on_pos(x, y):
        if atb.TwMouseMotion(x,y):
            pass
        if bar.draw_roi.value == 1:
            pos = glfwGetMousePos()
            pos = normalize(pos,glfwGetWindowSize())
            pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # pos in frame.img pixels
            u_r.setEnd(pos)

    def on_scroll(pos):
        if not atb.TwMouseWheel(pos):
            pass

    def on_close():
        g_pool.quit.value = True
        print "EYE Process closing from window"


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
    session_settings = shelve.open('user_settings',protocol=2)
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


    # pupil object
    pupil = Temp()
    pupil.norm_coords = (0.,0.)
    pupil.image_coords = (0.,0.)
    pupil.ellipse = None
    pupil.gaze_coords = (0.,0.)

    try:
        pupil.pt_cloud = np.load('cal_pt_cloud.npy')
        map_pupil = get_map_from_cloud(pupil.pt_cloud,g_pool.world_size)
    except:
        pupil.pt_cloud = None
        def map_pupil(vector):
            """ 1 to 1 mapping
            """
            return vector

    u_r = Roi(frame.img.shape)
    u_r.set(load('roi',default=None))
    p_r = Roi(frame.img.shape)

    # local object
    l_pool = Temp()
    l_pool.calib_running = False
    l_pool.record_running = False
    l_pool.record_positions = []
    l_pool.record_path = None
    l_pool.writer = None

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


    # Initialize glfw
    glfwInit()
    glfwOpenWindow(width, height, 0, 0, 0, 8, 0, 0, GLFW_WINDOW)
    glfwSetWindowTitle("Eye")
    glfwSetWindowPos(800,0)
    if isinstance(g_pool.eye_src, str):
        glfwSwapInterval(0) # turn off v-sync when using video as src for benchmarking


    #register callbacks
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

    # event loop
    while glfwGetWindowParam(GLFW_OPENED) and not g_pool.quit.value:
        update_fps()
        frame = cap.get_frame()
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
        result = pupil_detector.detect(frame.img,u_roi=u_r,p_roi=p_r,visualize=bar.display.value == 2)

        # Work with detected ellipses
        if result:
            pupil.ellipse = result[0]
            pupil.image_coords = pupil.ellipse['center']
            # normalize
            pupil.norm_coords = normalize(pupil.image_coords, (frame.img.shape[1], frame.img.shape[0]),flip_y=True )
            # from pupil to gaze
            pupil.gaze_coords = map_pupil(pupil.norm_coords)
            # publish to globals
            g_pool.gaze[:] = pupil.gaze_coords
        else:
            pupil.ellipse = None
            g_pool.gaze[:] = 0.,0.
            pupil.gaze_coords = None # without this line the last known pupil position is recorded if none is found


        ### CALIBRATION ###
        # Initialize Calibration (setup variables and lists)
        if g_pool.calibrate.value and not l_pool.calib_running:
            l_pool.calib_running = True
            pupil.pt_cloud = []

        # While Calibrating... collect data
        if l_pool.calib_running and (not(g_pool.ref[:]==[0.,0.])) and pupil.ellipse:
            pupil.pt_cloud.append([pupil.norm_coords[0],pupil.norm_coords[1],
                                    g_pool.ref[0], g_pool.ref[1]])

        # Calculate mapping coefs if data has been collected
        if not g_pool.calibrate.value and l_pool.calib_running:
            l_pool.calib_running = 0
            if pupil.pt_cloud: # some data was actually collected
                print "Calibrating with", len(pupil.pt_cloud), "collected data points."
                pupil.pt_cloud = np.array(pupil.pt_cloud)
                map_pupil = get_map_from_cloud(pupil.pt_cloud,g_pool.world_size,verbose=True)
                np.save('cal_pt_cloud.npy',pupil.pt_cloud)


        ### RECORDING ###
        # Setup variables and lists for recording
        if g_pool.pos_record.value and not l_pool.record_running:
            l_pool.record_path = g_pool.eye_rx.recv()
            print "l_pool.record_path: ", l_pool.record_path

            video_path = os.path.join(l_pool.record_path, "eye.avi")
            if bar.record_eye.value:
                l_pool.writer = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), bar.fps.value, (frame.img.shape[1], frame.img.shape[0]))
            l_pool.record_positions = []
            l_pool.record_running = True

        # While recording...
        if l_pool.record_running:
            if pupil.gaze_coords is not None:
                l_pool.record_positions.append([pupil.gaze_coords[0], pupil.gaze_coords[1],pupil.norm_coords[0],pupil.norm_coords[1], bar.dt.value, g_pool.frame_count_record.value])
            if l_pool.writer is not None:
                l_pool.writer.write(frame.img)

        # Done Recording: Save values and flip switch to OFF for recording
        if not g_pool.pos_record.value and l_pool.record_running:
            positions_path = os.path.join(l_pool.record_path, "gaze_positions.npy")
            cal_pt_cloud_path = os.path.join(l_pool.record_path, "cal_pt_cloud.npy")
            np.save(positions_path, np.asarray(l_pool.record_positions))
            try:
                np.save(cal_pt_cloud_path, np.asarray(pupil.pt_cloud))
            except:
                print "Warning: No calibration data associated with this recording."
            l_pool.writer = None
            l_pool.record_running = False


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

        if bar.draw_pupil and pupil.ellipse:
            pts = cv2.ellipse2Poly( (int(pupil.image_coords[0]),int(pupil.image_coords[1])),
                                    (int(pupil.ellipse["axes"][0]/2),int(pupil.ellipse["axes"][1]/2)),
                                    int(pupil.ellipse["angle"]),0,360,15)
            draw_gl_polyline(pts,(1.,0,0,.5))
            draw_gl_point_norm(pupil.norm_coords,color=(1.,0.,0.,0.5))

        atb.draw()
        glfwSwapBuffers()

    # END while running

    # save session persistent settings
    save('roi',u_r.get())
    save('bar.display',bar.display.value)
    save('bar.draw_pupil',bar.draw_pupil.value)
    save('bar.record_eye',bar.record_eye.value)
    session_settings.close()

    atb.terminate()
    glfwCloseWindow()
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

