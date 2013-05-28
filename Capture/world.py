'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import os, sys
from time import time
from ctypes import  c_int,c_bool,c_float,create_string_buffer
import numpy as np
import cv2
from glob import glob
from glfw import *
import atb
from methods import normalize, denormalize, chessboard, circle_grid, gen_pattern_grid, calibrate_camera,Temp
from uvc_capture import autoCreateCapture
from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen,draw_gl_point,draw_gl_point_norm,draw_gl_polyline_norm
from calibrate import *
from reference_detectors import *

def world_profiled(g_pool):
    import cProfile
    from world import world
    cProfile.runctx("world(g_pool,)",{"g_pool":g_pool},locals(),"world.pstats")

def world(g_pool):
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
                ref.detector.new_ref(pos)


    def on_pos(x, y):
        if atb.TwMouseMotion(x,y):
            pass

    def on_scroll(pos):
        if not atb.TwMouseWheel(pos):
            pass

    def on_close():
        g_pool.quit.value = True
        print "WORLD Process closing from window"

    ref = Temp()
    ref.detector = no_Detector(g_pool.calibrate,g_pool.ref_x,g_pool.ref_y)
    ###objects as variable containers
    # pattern object
    pattern = Temp()
    pattern.centers = None
    pattern.obj_grid = gen_pattern_grid((4, 11))  # calib grid
    pattern.obj_points = []
    pattern.img_points = []
    pattern.map = (0, 2, 7, 16, 21, 23, 39, 40, 42)
    pattern.board_centers = None

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

    # initialize capture, check if it works
    cap = autoCreateCapture(g_pool.world_src, g_pool.world_size)
    if cap is None:
        print "WORLD: Error could not create Capture"
        return
    s, img = cap.read_RGB()
    if not s:
        print "WORLD: Error could not get image"
        return
    height,width = img.shape[:2]


    ###helpers called by the main atb bar
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
        data.value=mode #update the bar.value

    def get_from_data(data):
        """
        helper for atb getter and setter use
        """
        return data.value


    def start_calibration():
        c_type = bar.calibration_type.value
        if  c_type == cal_type["Directed 9-Point"]:
            print 'WORLD: Starting Directed 9-Point calibration.'
            ref.detector = Nine_Point_Detector(global_calibrate=g_pool.calibrate,
                                            shared_x=g_pool.ref_x,
                                            shared_y=g_pool.ref_y,
                                            shared_stage=g_pool.cal9_stage,
                                            shared_step=g_pool.cal9_step,
                                            shared_cal9_active=g_pool.cal9,
                                            shared_circle_id=g_pool.cal9_circle_id,
                                            auto_advance=False)
        elif c_type == cal_type["Automated 9-Point"]:
            print 'WORLD: Starting Automated 9-Point calibration.'
            ref.detector = Nine_Point_Detector(global_calibrate=g_pool.calibrate,
                                            shared_x=g_pool.ref_x,
                                            shared_y=g_pool.ref_y,
                                            shared_stage=g_pool.cal9_stage,
                                            shared_step=g_pool.cal9_step,
                                            shared_cal9_active=g_pool.cal9,
                                            shared_circle_id=g_pool.cal9_circle_id,
                                            auto_advance=True)
        elif c_type == cal_type["Natural Features"]:
            print 'WORLD: Starting Natural Features calibration.'
            ref.detector = Natural_Features_Detector(global_calibrate=g_pool.calibrate,
                                                    shared_x=g_pool.ref_x,
                                                    shared_y=g_pool.ref_y)
        elif c_type == cal_type["Black Dot"]:
            print 'WORLD: Starting Black Dot calibration.'
            ref.detector = Black_Dot_Detector(global_calibrate=g_pool.calibrate,
                                            shared_x=g_pool.ref_x,
                                            shared_y=g_pool.ref_y)
    def advance_calibration():
        ref.detector.advance()

    def stop_calibration():
        ref.detector = no_Detector(global_calibrate=g_pool.calibrate,
                                shared_x=g_pool.ref_x,
                                shared_y=g_pool.ref_y)

    ### Initialize ant tweak bar inherits from atb.Bar
    atb.init()
    bar = atb.Bar(name = "World", label="Controls",
            help="Scene controls", color=(50, 50, 50), alpha=100,
            text='light', position=(10, 10),refresh=.3, size=(200, 200))
    bar.fps = c_float(0.0)
    bar.timestamp = time()
    bar.calibration_type = c_int(1)
    bar.show_calib_result = c_bool(0)
    bar.calibration_images = False
    bar.record_video = c_bool(0)
    bar.record_running = c_bool(0)
    bar.play = g_pool.play
    bar.window_size = c_int(0)
    window_size_enum = atb.enum("Display Size",{"Full":0, "Medium":1,"Half":2,"Mini":3})
    cal_type = {"Directed 9-Point":0,"Automated 9-Point":1,"Natural Features":3,"Black Dot":4}#"Manual 9-Point":2
    calibrate_type_enum = atb.enum("Calibration Method",cal_type)
    bar.rec_name = create_string_buffer(512)

    # play and record can be tied together via pointers to the objects
    # bar.play = bar.record_video
    bar.add_var("FPS", bar.fps, step=1., readonly=True)
    bar.add_var("Display_Size", vtype=window_size_enum,setter=set_window_size,getter=get_from_data,data=bar.window_size)
    bar.add_var("Cal/Calibration_Method",bar.calibration_type, vtype=calibrate_type_enum)
    bar.add_button("Cal/Start_Calibration",start_calibration, key='c')
    bar.add_button("Cal/Next_Point",advance_calibration,key="SPACE", help="Hit space to calibrate on next dot")
    bar.add_button("Cal/Stop_Calibration",stop_calibration, key='d')
    bar.add_var("Cal/show_calibration_result",bar.show_calib_result, help="yellow lines indecate fit error, red outline shows the calibrated area.")
    bar.add_var("Rec/rec_name",bar.rec_name)
    bar.add_var("Rec/Record_Video", bar.record_video, key="r", help="Start/Stop Recording")
    bar.add_separator("Sep1")
    bar.add_var("Play Source Video", bar.play)
    bar.add_var("Exit", g_pool.quit)

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


    ###event loop
    while glfwGetWindowParam(GLFW_OPENED) and not g_pool.quit.value:
        update_fps()

        ###get input characters entered in player
        if g_pool.player_char.value is not '0':
            player_char = g_pool.player_char.value
            g_pool.player_char.value = '0'
            on_char(player_char,False)

        # get an image from the grabber
        s, img = cap.read()
        ref.detector.detect(img)
        if ref.detector.is_done():
            stop_calibration()

        g_pool.player_refresh.set()


        # #gather pattern centers and find cam intrisics
        # if bar.screen_shot and pattern.centers is not None:
        #     bar.screen_shot = False
        #     # calibrate the camera intrinsics if the board is found
        #     # append list of circle grid center points to pattern.img_points
        #     # append generic list of circle grid pattern type to  pattern.obj_points
        #     pattern.centers = circle_grid(img)
        #     pattern.img_points.append(pattern.centers)
        #     pattern.obj_points.append(pattern.obj_grid)
        #     print "Number of Patterns Captured:", len(pattern.img_points)
        #     #if pattern.img_points.shape[0] > 10:
        #     if len(pattern.img_points) > 10:
        #         camera_matrix, dist_coefs = calibrate_camera(np.asarray(pattern.img_points),
        #                                             np.asarray(pattern.obj_points),
        #                                             (img.shape[1], img.shape[0]))
        #         np.save("camera_matrix.npy", camera_matrix)
        #         np.save("dist_coefs.npy", dist_coefs)
        #         pattern.img_points = []
        #         bar.find_pattern.value = False

        ### Setup recording process
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


            # positions data to eye process
            g_pool.pos_record.value = True
            g_pool.eye_tx.send(record.path)

            bar.record_running = 1
            g_pool.frame_count_record.value = 0

        # While Recording...
        if bar.record_video and bar.record_running:
            # Save image frames to video writer
            # increment the frame_count_record value
            # Eye positions can be associated with frames of recording even if different framerates are used
            record.writer.write(img)
            g_pool.frame_count_record.value += 1


        # Finish all recordings, clean up.
        if not bar.record_video and bar.record_running:
            # for conviniece: copy camera intrinsics into each data folder at the end of a recording.
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



        ###render the screen
        clear_gl_screen()
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB,img)
        draw_gl_texture(img)

        ###render calibration results:
        if bar.show_calib_result.value:
            cal_pt_cloud = np.load("cal_pt_cloud.npy")
            pX,pY,wX,wY = cal_pt_cloud.transpose()
            map_fn = get_map_from_cloud(cal_pt_cloud,(width,height))
            modelled_world_pts = map_fn((pX,pY))
            pts = np.array(modelled_world_pts,dtype=np.float32).transpose()
            calib_bounds =  cv2.convexHull(pts)[:,0]
            for observed,modelled in zip(zip(wX,wY),np.array(modelled_world_pts).transpose()):
                draw_gl_polyline_norm((modelled,observed),(1.,0.5,0.,.5))
            draw_gl_polyline_norm(calib_bounds,(1.0,0,0,.5))

        #render visual feedback from detector
        ref.detector.display(img)
        # render detector point
        if ref.detector.pos[0] or ref.detector.pos[1]:
            draw_gl_point_norm(ref.detector.pos,color=(0.,1.,0.,0.5))

        # update gaze point from shared variable pool and draw on screen. If both coords are 0: no pupil pos was detected.
        if g_pool.gaze_x.value !=0 or g_pool.gaze_y.value !=0:
            draw_gl_point_norm((g_pool.gaze_x.value, g_pool.gaze_y.value),color=(1.,0.,0.,0.5))

        atb.draw()
        glfwSwapBuffers()

    ###end while running clean-up
    print "WORLD Process closed"
    glfwCloseWindow()
    glfwTerminate()

