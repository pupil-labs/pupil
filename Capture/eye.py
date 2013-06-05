'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from ctypes import c_int,c_bool,c_float
import numpy as np
import atb
from glfw import *
from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen,draw_gl_point,draw_gl_point_norm,draw_gl_polyline
from time import time, sleep
from methods import *
from c_methods import eye_filter
from uvc_capture import autoCreateCapture
from calibrate import get_map_from_cloud
from pupil_detectors import Canny_Detector
from os import path
import shelve

def eye(g_pool):
    """
    this needs a docstring
    """
    # # glfw callback functions
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
                    pos = denormalize(pos,(img.shape[1],img.shape[0]) ) #pos in img pixels
                    r.setStart(pos)
                    bar.draw_roi.value = 1
                else:
                    bar.draw_roi.value = 0

    def on_pos(x, y):
        if atb.TwMouseMotion(x,y):
            pass
        if bar.draw_roi.value == 1:
            pos = glfwGetMousePos()
            pos = normalize(pos,glfwGetWindowSize())
            pos = denormalize(pos,(img.shape[1],img.shape[0]) ) #pos in img pixels
            r.setEnd(pos)

    def on_scroll(pos):
        if not atb.TwMouseWheel(pos):
            pass

    def on_close():
        g_pool.quit.value = True
        print "EYE Process closing from window"


    ###helpers called by the main atb bar
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


    # load session persisten settings
    session_settings = shelve.open('user_settings',protocol=2)
    def load(var_name,default):
        try:
            return session_settings[var_name]
        except:
            return default
    def save(var_name,var):
        session_settings[var_name] = var

    # initialize capture
    cap = autoCreateCapture(g_pool.eye_src, g_pool.eye_size)
    if cap is None:
        print "EYE: Error could not create Capture"
        return
    #check if it works
    s, img = cap.read_RGB()
    if not s:
        print "EYE: Error could not get image"
        return
    height,width = img.shape[:2]


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

    r = Roi(img.shape)
    r.set(load('roi',default=None))
    p_r = Roi(img.shape)

    # local object
    l_pool = Temp()
    l_pool.calib_running = False
    l_pool.record_running = False
    l_pool.record_positions = []
    l_pool.record_path = None
    l_pool.writer = None

    pupil_detector = Canny_Detector()

    atb.init()
    ###Create Main ATB Controls
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
    bar.bin_thresh = c_int(60)
    bar.blur = c_int(load('bar.blur',1))
    bar.pupil_ratio = c_float(1.0)
    bar.pupil_angle = c_float(0.0)
    bar.pupil_size = c_float(80.)
    bar.pupil_size_tolerance = c_float(load('bar.pupil_size_tolerance',40))
    bar.canny_aperture = c_int(load('bar.canny_aperture',5))
    bar.canny_thresh = c_int(load('bar.canny_thresh',200))
    bar.canny_ratio = c_int(2)
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

    #add 4vl2 camera controls to a seperate ATB bar
    if cap.controls is not None:
        c_bar = atb.Bar(name="Camera_Controls", label=cap.name,
            help="UVC Camera Controls", color=(50,50,50), alpha=100,
            text='light',position=(220, 10),refresh=2., size=(200, 200))

        # c_bar.add_var("auto_refresher",vtype=atb.TW_TYPE_BOOL8,getter=cap.uvc_refresh_all,setter=None,readonly=True)
        # c_bar.define(definition='visible=0', varname="auto_refresher")

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

    ###create a bar for the detector
    pupil_detector.create_atb_bar(pos=(10,120))


    # Initialize glfw
    glfwInit()
    glfwOpenWindow(width, height, 0, 0, 0, 8, 0, 0, GLFW_WINDOW)
    glfwSetWindowTitle("Eye")
    glfwSetWindowPos(800,0)
    if isinstance(g_pool.eye_src, str):
        glfwSwapInterval(0) # turn of v-sync when using video as src for benchmarking


    #register callbacks
    glfwSetWindowSizeCallback(on_resize)
    glfwSetWindowCloseCallback(on_close)
    glfwSetKeyCallback(on_key)
    glfwSetCharCallback(on_char)
    glfwSetMouseButtonCallback(on_button)
    glfwSetMousePosCallback(on_pos)
    glfwSetMouseWheelCallback(on_scroll)

    #gl_state settings
    import OpenGL.GL as gl
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_BLEND)
    del gl

    #event loop
    while glfwGetWindowParam(GLFW_OPENED) and not g_pool.quit.value:
        update_fps()
        s,img = cap.read_RGB()
        sleep(bar.sleep.value) # for debugging only

        ###IMAGE PROCESSING and clipping to user defined eye-region
        eye_img = img[r.lY:r.uY,r.lX:r.uX]
        gray_img = grayscale(eye_img)


        ### coarse pupil-detection
        integral = cv2.integral(gray_img)
        integral =  np.array(integral,dtype=c_float)
        x,y,w = eye_filter(integral)
        if w>0:
            p_r.set((y,x,y+w,x+w))
        else:
            p_r.set((0,0,-1,-1))

        ###fine pupil ellipse detection
        result = pupil_detector.detect(img,roi=r,p_roi=p_r,visualize=bar.display.value == 2)

        ### Work with detected ellipses
        if result:
            pupil.ellipse = result[0]
            pupil.image_coords = r.add_vector(p_r.add_vector(pupil.ellipse['center']))
            # normalize
            pupil.norm_coords = normalize(pupil.image_coords, (img.shape[1], img.shape[0]),flip_y=True )
            # from pupil to gaze
            pupil.gaze_coords = map_pupil(pupil.norm_coords)
            # puplish to globals
            g_pool.gaze_x.value, g_pool.gaze_y.value = pupil.gaze_coords
        else:
            pupil.ellipse = None
            g_pool.gaze_x.value, g_pool.gaze_y.value = 0.,0.
            pupil.gaze_coords = None #whithout this line the last know pupil position is recorded if none is found


        ###CALIBRATION###
        # Initialize Calibration (setup variables and lists)
        if g_pool.calibrate.value and not l_pool.calib_running:
            l_pool.calib_running = True
            pupil.pt_cloud = []

        # While Calibrating... collect data
        if l_pool.calib_running and ((g_pool.ref_x.value != 0) or (g_pool.ref_y.value != 0)) and pupil.ellipse:
            pupil.pt_cloud.append([pupil.norm_coords[0],pupil.norm_coords[1],
                                g_pool.ref_x.value, g_pool.ref_y.value])

        # Calculate mapping coefs if data has been collected
        if not g_pool.calibrate.value and l_pool.calib_running:
            l_pool.calib_running = 0
            if pupil.pt_cloud: # some data was actually collected
                print "Calibrating with", len(pupil.pt_cloud), "collected data points."
                pupil.pt_cloud = np.array(pupil.pt_cloud)
                map_pupil = get_map_from_cloud(pupil.pt_cloud,g_pool.world_size,verbose=True)
                np.save('cal_pt_cloud.npy',pupil.pt_cloud)


        ###RECORDING###
        # Setup variables and lists for recording
        if g_pool.pos_record.value and not l_pool.record_running:
            l_pool.record_path = g_pool.eye_rx.recv()
            print "l_pool.record_path: ", l_pool.record_path

            video_path = path.join(l_pool.record_path, "eye.avi")
            if bar.record_eye.value:
                l_pool.writer = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), bar.fps.value, (img.shape[1], img.shape[0]))
            l_pool.record_positions = []
            l_pool.record_running = True

        # While recording...
        if l_pool.record_running:
            if pupil.gaze_coords is not None:
                l_pool.record_positions.append([pupil.gaze_coords[0], pupil.gaze_coords[1],pupil.norm_coords[0],pupil.norm_coords[1], bar.dt.value, g_pool.frame_count_record.value])
            if l_pool.writer is not None:
                l_pool.writer.write(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        # Done Recording: Save values and flip switch to off for recording
        if not g_pool.pos_record.value and l_pool.record_running:
            positions_path = path.join(l_pool.record_path, "gaze_positions.npy")
            cal_pt_cloud_path = path.join(l_pool.record_path, "cal_pt_cloud.npy")
            np.save(positions_path, np.asarray(l_pool.record_positions))
            try:
                np.save(cal_pt_cloud_path, np.asarray(pupil.pt_cloud))
            except:
                print "Warning: No calibration data associated with this recording."
            l_pool.writer = None
            l_pool.record_running = False


        ###direct visualzations on the img data
        if bar.display.value == 1:
            # and a solid (white) frame around the user defined ROI
            gray_img[:,0] = 255
            gray_img[:,-1]= 255
            gray_img[0,:] = 255
            gray_img[-1,:]= 255
            img[r.lY:r.uY,r.lX:r.uX] = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

            pupil_img =img[r.lY:r.uY,r.lX:r.uX][p_r.lY:p_r.uY,p_r.lX:p_r.uX] #create an RGB view onto the gray pupil ROI
            #draw a blue dotted frame around the automatic pupil ROI in overlay...
            pupil_img[::2,0] = 0,0,255
            pupil_img[::2,-1]= 0,0,255
            pupil_img[0,::2] = 0,0,255
            pupil_img[-1,::2]= 0,0,255

            img[r.lY:r.uY,r.lX:r.uX][p_r.lY:p_r.uY,p_r.lX:p_r.uX] = pupil_img

        elif bar.display.value == 3:
            img = img[r.lY:r.uY,r.lX:r.uX][p_r.lY:p_r.uY,p_r.lX:p_r.uX]
        ### GL-drawing
        clear_gl_screen()
        draw_gl_texture(img)

        if bar.draw_pupil and pupil.ellipse:
            pts = cv2.ellipse2Poly( (int(pupil.image_coords[0]),int(pupil.image_coords[1])),
                                    (int(pupil.ellipse["axes"][0]/2),int(pupil.ellipse["axes"][1]/2)),
                                    int(pupil.ellipse["angle"]),0,360,15)
            draw_gl_polyline(pts,(1.,0,0,.5))
            draw_gl_point_norm(pupil.norm_coords,color=(1.,0.,0.,0.5))

        atb.draw()
        glfwSwapBuffers()

    ###end while running

    ###save session persisten settings
    save('roi',r.get())
    save('bar.display',bar.display.value)
    save('bar.draw_pupil',bar.draw_pupil.value)
    save('bar.record_eye',bar.record_eye.value)
    # save('bar.blur',bar.blur.value)
    # save('bar.pupil_size_tolerance',bar.pupil_size_tolerance.value)
    # save('bar.canny_aperture',bar.canny_aperture.value)
    # save('bar.canny_thresh',bar.canny_thresh.value)
    session_settings.close()

    atb.terminate()
    glfwCloseWindow()
    glfwTerminate()
    print "EYE Process closed"

def eye_profiled(g_pool):
    import cProfile
    from eye import eye
    cProfile.runctx("eye(g_pool,)",{"g_pool":g_pool},locals(),"eye.pstats")
