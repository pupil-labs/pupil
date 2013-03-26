import os, sys

from ctypes import  c_int,c_int64,c_bool,c_float
import numpy as np
from glob import glob
import cv2
from glfw import *
import atb
from methods import normalize, denormalize, chessboard, circle_grid, gen_pattern_grid, calibrate_camera
from methods import Temp,capture
# from gl_shapes import Point
from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen,draw_gl_point,draw_gl_point_norm
from time import time

class Bar(atb.Bar):
    """docstring for Bar"""
    def __init__(self, name,g_pool, defs):
        super(Bar, self).__init__(name, **defs)
        self.fps = c_float(0.0)
        self.timestamp = time()
        self.calibrate = g_pool.calibrate
        self.find_pattern = c_bool(0)
        self.optical_flow = c_bool(0)
        self.screen_shot = False
        self.calibration_images = False
        self.calibrate_nine = g_pool.cal9
        self.calibrate_nine_step = g_pool.cal9_step
        self.calibrate_nine_stage = g_pool.cal9_stage
        self.calib_running = g_pool.calibrate
        self.record_video = c_bool(0)
        self.record_running = c_bool(0)
        self.play = g_pool.play
        # play and record can be tied together via pointers to the objects
        # self.play = self.record_video

        self.add_var("FPS", self.fps, step=1., readonly=True)
        self.add_var("Find Calibration Pattern", self.find_pattern, key="p", help="Find Calibration Pattern")
        self.add_var("Optical Flow", self.optical_flow, key="o", help="Activate Optical Flow")
        self.add_button("Screen Shot", self.screen_cap, key="SPACE", help="Capture A Frame")
        self.add_var("Calibrate", self.calibrate, key="c", help="Start/Stop Calibration Process")
        self.add_var("Nine_Pt", self.calibrate_nine, key="9", help="Start/Stop 9 Point Calibration Process (Tip: hit 9 in the player window)")
        self.add_var("Record Video", self.record_video, key="r", help="Start/Stop Recording")
        self.add_var("Play Source Video", self.play)
        self.add_var("Exit", g_pool.quit)

    def update_fps(self):
        old_time, self.timestamp = self.timestamp, time()
        dt = self.timestamp - old_time
        if dt:
            self.fps.value += .05 * (1 / dt - self.fps.value)

    def screen_cap(self):
        self.find_pattern.value = True
        self.screen_shot = True

def world_profiled(src,size,g_pool):
    import cProfile
    from world import world
    cProfile.runctx("world(src,size,g_pool)",{'src':src,"size":size,"g_pool":g_pool},locals(),"world.pstats")


def world(src, size, g_pool):
    """world
    """

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
            bar.update()


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

    # initialize capture, check if it works
    cap = capture(src, size)
    s, img = cap.read_RGB()
    if not s:
        print "World: Error could not get image"
        return
    height,width = img.shape[:2]

    # Initialize ant tweak bar inherits from atb.Bar
    atb.init()
    bar = Bar("World", g_pool, dict(label="Controls",
            help="Scene controls", color=(50, 50, 50), alpha=50,
            text='light', refresh=.2, position=(10, 10), size=(200, 200)))

    #add 4vl2 camera controls to ATB bar
    import v4l2_ctl
    controls = v4l2_ctl.extract_controls(src)
    bar.camera_ctl = dict()
    bar.camera_state = dict()
    for control in controls:
        if control["type"]=="(bool)":
            bar.camera_ctl[control["name"]]=c_bool(control["value"])
            bar.camera_state[control["name"]]=c_bool(control["value"])
            bar.add_var("Camera/"+control["name"],bar.camera_ctl[control["name"]])
        elif control["type"]=="(int)":
            bar.camera_ctl[control["name"]]=c_int64(control["value"])
            bar.camera_state[control["name"]]=c_int64(control["value"])
            bar.add_var("Camera/"+control["name"],bar.camera_ctl[control["name"]],max=control["max"],min=control["min"],step=control["step"])
        elif control["type"]=="(menu)":
            bar.camera_ctl[control["name"]]=c_int64(control["value"])
            bar.camera_state[control["name"]]=c_int64(control["value"])
            bar.add_var("Camera/"+control["name"],bar.camera_ctl[control["name"]],max=control["max"],min=control["min"],step=1)
        else:
            pass

    # Initialize glfw
    glfwInit()
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

    #event loop
    while glfwGetWindowParam(GLFW_OPENED) and not g_pool.quit.value:
        bar.update_fps()
        # get an image from the grabber
        s, img = cap.read()

        #update camera control if needed
        for k in bar.camera_ctl.viewkeys():
            if bar.camera_state[k].value != bar.camera_ctl[k].value:
                #print src,k,bar.camera_ctl[k].value
                v4l2_ctl.set(src,k,bar.camera_ctl[k].value)
                bar.camera_state[k].value= bar.camera_ctl[k].value


        # Nine Point calibration state machine timing
        if bar.calibrate_nine.value:
            bar.calibrate.value = True
            bar.find_pattern.value = False
            if bar.calibrate_nine_step.value > 30:
                bar.calibrate_nine_step.value = 0
                bar.calibrate_nine_stage.value += 1
            if bar.calibrate_nine_stage.value > 8:
                bar.calibrate_nine_stage.value = 0
                bar.calibrate.value = False
                bar.calibrate_nine.value = False
                bar.find_pattern.value = False
            if bar.calibrate_nine_step.value in range(10, 25):
                bar.find_pattern.value = True

            g_pool.cal9_circle_id.value = pattern.map[bar.calibrate_nine_stage.value]
            bar.calibrate_nine_step.value += 1
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
        if g_pool.gaze_x.value or g_pool.gaze_y.value:
            draw_gl_point_norm((g_pool.gaze_x.value, g_pool.gaze_y.value),(1.,0.,0.,0.5))

        bar.update()
        bar.draw()
        glfwSwapBuffers()

    #end while running
    print "WORLD Process closed"
    glfwCloseWindow()
    glfwTerminate()
