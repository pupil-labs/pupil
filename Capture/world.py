import os, sys

from ctypes import  c_int,c_bool,c_float
import numpy as np
from glob import glob
import cv2
from glfw import *
import atb
from methods import normalize, denormalize, chessboard, circle_grid, gen_pattern_grid, calibrate_camera,Temp
from uvc_capture import Capture
# from gl_shapes import Point
from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen,draw_gl_point,draw_gl_point_norm
from time import time


def world_profiled(src,size,g_pool):
    import cProfile
    from world import world
    cProfile.runctx("world(src,size,g_pool)",{'src':src,"size":size,"g_pool":g_pool},locals(),"world.pstats")

def world(src, size, g_pool):
    """world
    """
    ###Callback funtions
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
                pos = denormalize(pos,(img.shape[1],img.shape[0]) ) #pos in img pixels

                if bar.optical_flow.value:
                    flow.point = np.array([pos,],dtype=np.float32)
                    flow.new_ref = True
                    flow.count = 30


    def on_pos(x, y):
        if atb.TwMouseMotion(x,y):
            pass

    def on_scroll(pos):
        if not atb.TwMouseWheel(pos):
            pass

    def on_close():
        g_pool.quit.value = True
        print "WORLD Process closing from window"



    ###objects as variable containers
    # pattern object
    pattern = Temp()
    pattern.centers = None
    pattern.norm_coords = (0., 0.)
    pattern.image_coords = (0., 0.)
    pattern.obj_grid = gen_pattern_grid((4, 11))  # calib grid
    pattern.obj_points = []
    pattern.img_points = []
    pattern.map = (0, 2, 7, 16, 21, 23, 39, 40, 42)
    pattern.board_centers = None
    #opticalflow object
    flow = Temp()
    flow.first =  None
    flow.point =  None
    flow.new_ref = False
    flow.count = 0
    # gaze object
    gaze = Temp()
    gaze.map_coords = (0., 0.)
    gaze.image_coords = (0., 0.)
    # record object
    record = Temp()
    record.writer = None
    record.path_parent = os.path.dirname(os.path.abspath(sys.argv[0]))
    record.path = None
    record.counter = 0

    ### initialize capture, check if it works
    cap = Capture(src, size)
    s, img = cap.read_RGB()

    if not s:
        print "World: Error could not get image"
        return


    ###helpers called by the main atb bar
    def update_fps():
        old_time, bar.timestamp = bar.timestamp, time()
        dt = bar.timestamp - old_time
        if dt:
            bar.fps.value += .05 * (1 / dt - bar.fps.value)

    def screen_cap():
        bar.find_pattern.value = True
        bar.screen_shot = True

    def set_window_size(mode,data):
        height,width = img.shape[:2]
        ratio = (1,.75,.5,.25)[mode]
        w,h = int(width*ratio),int(height*ratio)
        glfwSetWindowSize(w,h)
        data.value=mode #update the bar.value

    def get_from_data(data):
        """
        helper for atb getter and setter use
        """
        return data.value

    ### Initialize ant tweak bar inherits from atb.Bar
    atb.init()
    bar = atb.Bar(name = "World", label="Controls",
            help="Scene controls", color=(50, 50, 50), alpha=100,
            text='light', position=(10, 10),refresh=.3, size=(200, 200))
    bar.fps = c_float(0.0)
    bar.timestamp = time()
    bar.calibrate = g_pool.calibrate
    bar.find_pattern = c_bool(0)
    bar.optical_flow = c_bool(0)
    bar.screen_shot = False
    bar.calibration_images = False
    bar.calibrate_nine = g_pool.cal9
    bar.calibrate_nine_step = g_pool.cal9_step
    bar.calibrate_nine_stage = g_pool.cal9_stage
    bar.calibrate_auto_advance = c_bool(0)
    bar.calibrate_next = c_bool(0)
    bar.calib_running = g_pool.calibrate
    bar.record_video = c_bool(0)
    bar.record_running = c_bool(0)
    bar.play = g_pool.play
    bar.window_size = c_int(0)
    window_size_enum = atb.enum("Display Size",{"Full":0, "Medium":1,"Half":2,"Mini":3})
    # play and record can be tied together via pointers to the objects
    # bar.play = bar.record_video
    bar.add_var("FPS", bar.fps, step=1., readonly=True)
    bar.add_var("Display_Size", vtype=window_size_enum,setter=set_window_size,getter=get_from_data,data=bar.window_size)
    bar.add_var("Cal/Find_Calibration_Pattern", bar.find_pattern, key="p", help="Find Calibration Pattern")
    bar.add_var("Cal/Optical_Flow", bar.optical_flow, key="o", help="Activate Optical Flow")
    bar.add_button("Cal/Screen_Shot", screen_cap, key="s", help="Capture A Frame")
    bar.add_var("Cal/Calibrate", bar.calibrate, key="c", help="Start/Stop Calibration Process")
    bar.add_var("Cal9/Start", bar.calibrate_nine, key="9", help="Start/Stop 9 Point Calibration Process (Tip: hit 9 in the player window)")
    bar.add_var("Cal9/Nine_Pt_stage", bar.calibrate_nine, key="9", help="Start/Stop 9 Point Calibration Process (Tip: hit 9 in the player window)")
    bar.add_var("Cal9/Stage",bar.calibrate_nine_stage,help="Please look at dot ... to calibrate")
    bar.add_var("Cal9/Auto_Advance", bar.calibrate_auto_advance)
    bar.add_var("Cal9/Next_Point",bar.calibrate_next,key="SPACE", help="Hit space to calibrate on next dot")
    bar.add_var("Record Video", bar.record_video, key="r", help="Start/Stop Recording")
    bar.add_var("Play Source Video", bar.play)
    bar.add_var("Exit", g_pool.quit)


    ###add camera controls to a seperate ATB bar
    if cap.uvc_camera is not None:
        c_bar = atb.Bar(name="Camera_Controls", label=cap.uvc_camera.name,
            help="UVC Camera Controls", color=(50,50,50), alpha=100,
            text='light',position=(220, 10),refresh=2., size=(200, 200))
        # c_bar.add_var("auto_refresher",vtype=atb.TW_TYPE_BOOL8,getter=cap.uvc_refresh_all,setter=None,readonly=True)
        # c_bar.define(definition='visible=0', varname="auto_refresher")
        sorted_controls = [c for c in cap.uvc_camera.controls.itervalues()]
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
            elif control.type == "unknown control":
                pass
        c_bar.add_button("refresh",cap.uvc_camera.update_from_device)
        c_bar.add_button("load defaults",cap.uvc_camera.load_defaults)

    else:
        c_bar = None


    ### Initialize glfw
    glfwInit()
    height,width = img.shape[:2]
    glfwOpenWindow(width, height, 0, 0, 0, 8, 0, 0, GLFW_WINDOW)
    glfwSetWindowTitle("World")
    glfwSetWindowPos(0,0)

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
    gl.glPointSize(20)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_BLEND)
    del gl

    ###event loop
    while glfwGetWindowParam(GLFW_OPENED) and not g_pool.quit.value:
        update_fps()
        # get an image from the grabber
        s, img = cap.read()

        # Nine Point calibration state machine timing
        if bar.calibrate_nine.value:
            bar.calibrate.value = True
            if bar.calibrate_nine_step.value > 30:
                bar.calibrate_nine_step.value = 0
                bar.calibrate_nine_stage.value += 1
                bar.calibrate_next.value = False
            if bar.calibrate_nine_stage.value > 8:
                bar.calibrate_nine_stage.value = 0
                bar.calibrate.value = False
                bar.calibrate_nine.value = False
                bar.find_pattern.value = False
            if bar.calibrate_nine_step.value in range(10, 25):
                bar.find_pattern.value = True
            else:
                bar.find_pattern.value = False

            g_pool.cal9_circle_id.value = pattern.map[bar.calibrate_nine_stage.value]
            if bar.calibrate_next.value or bar.calibrate_auto_advance.value:
                bar.calibrate_nine_step.value += 1
            else:
                pass

        g_pool.player_refresh.set()


        #pattern detection and its various uses
        pattern.centers = None
        if bar.find_pattern.value:
            pattern.centers = circle_grid(img)

        if pattern.centers is not None:
            if bar.calibrate_nine.value:
                pattern.image_coords = pattern.centers[g_pool.cal9_circle_id.value][0]
            else:
                mean = pattern.centers.sum(0) / pattern.centers.shape[0]
                pattern.image_coords = mean[0]

            pattern.norm_coords = normalize(pattern.image_coords, (img.shape[1],img.shape[0]),flip_y=True)
        else:
            # If no pattern detected send 0,0 -- check this condition in eye process
            pattern.norm_coords = 0,0

        #optical flow for natural marker calibration method
        if bar.optical_flow.value:
            if flow.new_ref:
                flow.first = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                flow.new_ref = False

            if flow.point is not None and flow.count:
                gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                nextPts, status, err = cv2.calcOpticalFlowPyrLK(flow.first,gray,flow.point,winSize=(100,100))

                if status[0]:
                    flow.point = nextPts
                    flow.first = gray
                    nextPts = nextPts[0]

                    pattern.image_coords = nextPts
                    pattern.norm_coords = normalize(nextPts, (img.shape[1],img.shape[0]),flip_y=True)
                    flow.count -=1
                    print flow.count
                else:
                    # If no pattern detected send 0,0 -- check this condition in eye process
                    pattern.norm_coords = 0,0


        #gather pattern centers and find cam intrisics
        if bar.screen_shot and pattern.centers is not None:
            bar.screen_shot = False
            # calibrate the camera intrinsics if the board is found
            # append list of circle grid center points to pattern.img_points
            # append generic list of circle grid pattern type to  pattern.obj_points
            pattern.img_points.append(pattern.centers)
            pattern.obj_points.append(pattern.obj_grid)
            print "Number of Patterns Captured:", len(pattern.img_points)
            #if pattern.img_points.shape[0] > 10:
            if len(pattern.img_points) > 10:
                camera_matrix, dist_coefs = calibrate_camera(np.asarray(pattern.img_points),
                                                    np.asarray(pattern.obj_points),
                                                    (img.shape[1], img.shape[0]))
                np.save("camera_matrix.npy", camera_matrix)
                np.save("dist_coefs.npy", dist_coefs)
                pattern.img_points = []
                bar.find_pattern.value = False

        # Setup recording process
        if bar.record_video and not bar.record_running:
            record.path = os.path.join(record.path_parent, "data%03d/" % record.counter)
            while True:
                try:
                    os.mkdir(record.path)
                    break
                except:
                    print "We dont want to overwrite data, incrementing counter & trying to make new data folder"
                    record.counter += 1
                    record.path = os.path.join(record.path_parent, "data%03d/" % record.counter)

            #video
            video_path = os.path.join(record.path, "world.avi")
            #FFV1 -- good speed lossless big file
            #DIVX -- good speed good compression medium file
            record.writer = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), bar.fps.value, (img.shape[1], img.shape[0]))

            # audio data to audio process
            audio_path = os.path.join(record.path, "world.wav")
            try:
                g_pool.audio_record.value = 1
                g_pool.audio_tx.send(audio_path)
            except:
                print "no audio initialized"

            # positions data to eye process
            g_pool.pos_record.value = True
            g_pool.eye_tx.send(record.path)

            bar.record_running = 1
            g_pool.frame_count_record.value = 0

        # While Recording...
        if bar.record_video and bar.record_running:
            # Save image frames to video writer
            # increment the frame_count_record value
            # Eye positions can be associated with frames of recording even if different framerates
            g_pool.frame_count_record.value += 1
            record.writer.write(img)

        # Finish all recordings, clean up.
        if not bar.record_video and bar.record_running:
            try:
                g_pool.audio_record.value = 0
            except:
                print "no audio recorded"

            # conviniece function: copy camera intrinsics into each data folder at the end of a recording.
            try:
                camera_matrix = np.load("camera_matrix.npy")
                dist_coefs = np.load("dist_coefs.npy")
                cam_path = os.path.join(record.path, "camera_matrix.npy")
                dist_path = os.path.join(record.path, "dist_coefs.npy")
                np.save(cam_path, camera_matrix)
                np.save(dist_path, dist_coefs)
            except:
                print "no camera intrinsics found, will not copy them into data folder"

            g_pool.pos_record.value = 0
            del record.writer
            bar.record_running = 0

        clear_gl_screen()
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB,img)
        draw_gl_texture(img)


        # render and broadcast pattern point
        if pattern.norm_coords[0] or pattern.norm_coords[1]:
            draw_gl_point_norm(pattern.norm_coords,(0.,1.,0.,0.5))
            # draw_gl_point(pattern.image_coords,(0.,1.,0.,0.5))
            g_pool.pattern_x.value, g_pool.pattern_y.value = pattern.norm_coords
        else:
            # If no pattern detected send 0,0 -- check this condition in eye process
            g_pool.pattern_x.value, g_pool.pattern_y.value = 0,0

        # update gaze point from shared variable pool and draw on screen. If both coords are 0: no pupil pos was detected.
        if g_pool.gaze_x.value !=0 or g_pool.gaze_y.value !=0:
            draw_gl_point_norm((g_pool.gaze_x.value, g_pool.gaze_y.value),(1.,0.,0.,0.5))

        atb.draw()
        glfwSwapBuffers()

    ###end while running clean-up
    print "WORLD Process closed"
    glfwCloseWindow()
    glfwTerminate()
    cap.release()

