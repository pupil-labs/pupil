'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from ctypes import c_int,c_bool,c_float,pointer
import cPickle as pickle
import numpy as np
import atb
from glfw import *
from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen,draw_gl_point,draw_gl_point_norm,draw_gl_polyline
from time import time, sleep
from methods import *
from c_methods import eye_filter
from uvc_capture import autoCreateCapture
from calibrate import get_map_from_cloud
from os import path


class Bar(atb.Bar):
    """docstring for Bar"""
    def __init__(self, name,g_pool, bar_defs):
        super(Bar, self).__init__(name,**bar_defs)
        self.fps = c_float(0.0)
        self.timestamp = time()
        self.dt = c_float(0.0)
        self.sleep = c_float(0.0)
        self.display = c_int(1)
        self.draw_pupil = c_bool(1)
        self.draw_roi = c_int(0)
        self.bin_thresh = c_int(60)
        self.blur = c_int(3)
        self.pupil_ratio = c_float(1.0)
        self.pupil_angle = c_float(0.0)
        self.pupil_size = c_float(80.)
        self.pupil_size_tolerance = c_float(40.)
        self.canny_aperture = c_int(5)
        self.canny_thresh = c_int(200)
        self.canny_ratio = c_int(2)
        self.record_eye = c_bool(0)

        #add class field here and it will become session persistant
        self.session_save = {'display':self.display,
                            'draw_pupil':self.draw_pupil,
                            'bin_thresh':self.bin_thresh,
                            'pupil_ratio':self.pupil_ratio,
                            'pupil_size':self.pupil_size,
                            'mean_blur':self.blur,
                            'canny_aperture':self.canny_aperture,
                            'canny_thresh':self.canny_thresh,
                            'canny_ratio':self.canny_ratio}

        self.load()
        dispay_mode_enum = atb.enum("Mode",{"Camera Image":0,
                                            "Region of Interest":1,
                                            "Egdes":2,
                                            "Corse Pupil Region":3})
        self.add_var("Display/FPS",self.fps, step=1.,readonly=True)
        self.add_var("Display/SlowDown",self.sleep, step=0.01,min=0.0)
        self.add_var("Display/Mode", self.display,vtype=dispay_mode_enum, help="select the view-mode")
        self.add_var("Display/Show_Pupil_Point", self.draw_pupil)
        self.add_button("Draw_ROI", self.roi, help="drag on screen to select a region of interest", Group="Display")
        self.add_var("Pupil/Shade", self.bin_thresh,readonly=True)
        self.add_var("Pupil/Ratio", self.pupil_ratio, readonly=True)
        self.add_var("Pupil/Angle", self.pupil_angle,step=.1,readonly=True)
        self.add_var("Pupil/Size", self.pupil_size, readonly=True)
        self.add_var("Pupil/Size_Tolerance", self.pupil_size_tolerance, step=1, min=0)
        self.add_var("Canny/MeanBlur", self.blur,step=2,max=7,min=1)
        self.add_var("Canny/Aperture",self.canny_aperture, step=2, max=7, min=3)
        self.add_var("Canny/Lower_Threshold", self.canny_thresh, step=1,min=1)
        self.add_var("Canny/LowerUpperRatio", self.canny_ratio, step=1,min=0,help="Canny recommended a ratio between 3/1 and 2/1")
        self.add_var("record_eye_video", self.record_eye, help="when recording also save the eye video stream")
        self.add_var("SaveSettings&Exit", g_pool.quit)

    def update_fps(self):
        old_time, self.timestamp = self.timestamp, time()
        dt = self.timestamp - old_time
        if dt:
            self.fps.value += .05 * (1 / dt - self.fps.value)
        self.dt = dt

    def save(self):
        new_settings = dict([(key,field.value) for key, field in self.session_save.items()])
        settings_file = open('session_settings','wb')
        pickle.dump(new_settings,settings_file)
        settings_file.close

    def roi(self):
        self.draw_roi.value = 2


    def load(self):
        try:
            settings_file = open('session_settings','rb')
            new_settings = pickle.load(settings_file)
            settings_file.close
        except IOError:
            print "No session_settings file found. Using defaults"
            return

        for key,val in new_settings.items():
            try:
                self.session_save[key].value = val
            except KeyError:
                print "Warning the Sessions file is from a different version, not all fields may be updated"


class Roi(object):
    """this is a simple 2D Region of Interest class
    it is applied on numpy arrays for convinient slicing
    like this:

    roi_array_slice = full_array[r.lY:r.uY,r.lX:r.uX]
    #do something with roi_array_slice
    full_array[r.lY:r.uY,r.lX:r.uX] = roi_array_slice

    this creates a view, no data copying done
    """
    def __init__(self, array_shape):
        self.array_shape = array_shape
        self.lX = 0
        self.lY = 0
        self.uX = array_shape[1]-0
        self.uY = array_shape[0]-0
        self.nX = 0
        self.nY = 0
        self.load()

    def setStart(self,(x,y)):
        x,y = int(x),int(y)
        x,y = max(0,x),max(0,y)
        self.nX,self.nY = x,y

    def setEnd(self,(x,y)):
            x,y = int(x),int(y)
            x,y = max(0,x),max(0,y)
            #make sure the ROI actually contains enough pixels
            if abs(self.nX - x) > 25 and abs(self.nY - y)>25:
                self.lX = min(x,self.nX)
                self.lY = min(y,self.nY)
                self.uX = max(x,self.nX)
                self.uY = max(y,self.nY)

    def add_vector(self,(x,y)):
        """
        adds the roi offset to a len2 vector
        """
        return (self.lX+x,self.lY+y)

    def set(self,vals):
        if vals is not None and len(vals) is 4:
            self.lX,self.lY,self.uX,self.uY = vals

    def get(self):
        return self.lX,self.lY,self.uX,self.uY

    def save(self):
        new_settings = Temp()
        new_settings.array_shape = self.array_shape
        new_settings.vals = self.get()
        settings_file = open('session_settings_roi','wb')
        pickle.dump(new_settings,settings_file)
        settings_file.close

    def load(self):
            try:
                settings_file = open('session_settings_roi','rb')
                new_settings = pickle.load(settings_file)
                settings_file.close
            except IOError:
                print "No session_settings_roi file found. Using defaults"
                return

            if new_settings.array_shape == self.array_shape:
                self.set(new_settings.vals)
            else:
                print "Warning: Image Array size changed, disregarding saved Region of Interest"


def eye_profiled(g_pool):
    import cProfile
    from eye import eye
    cProfile.runctx("eye(g_pool,)",{"g_pool":g_pool},locals(),"eye.pstats")

def eye(g_pool):
    """
    this process needs a docstring
    """
    # # callback functions
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

    # initialize capture, check if it works
    cap = autoCreateCapture(g_pool.eye_src, g_pool.eye_size)
    if cap is None:
        print "EYE: Error could not create Capture"
        return
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
        map_pupil = get_map_from_cloud(pupil.pt_cloud,g_pool.world_size) ###world video size here
    except:
        pupil.pt_cloud = None
        def map_pupil(vector):
                return vector

    r = Roi(img.shape)
    p_r = Roi(img.shape)


    # local object
    l_pool = Temp()
    l_pool.calib_running = False
    l_pool.record_running = False
    l_pool.record_positions = []
    l_pool.record_path = None
    l_pool.writer = None
    l_pool.region_r = 20

    atb.init()
    bar = Bar("Eye",g_pool, dict(label="Controls",
            help="eye detection controls", color=(50,50,50), alpha=100,
            text='light',position=(10, 10),refresh=.1, size=(200, 300)) )



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
    gl.glPointSize(20)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_BLEND)
    del gl

    #event loop
    while glfwGetWindowParam(GLFW_OPENED) and not g_pool.quit.value:
        bar.update_fps()
        s,img = cap.read_RGB()
        sleep(bar.sleep.value) # for debugging only

        ###IMAGE PROCESSING
        gray_img = grayscale(img[r.lY:r.uY,r.lX:r.uX])

        integral = cv2.integral(gray_img)
        integral =  np.array(integral,dtype=c_float)
        x,y,w = eye_filter(integral)
        if w>0:
            p_r.set((y,x,y+w,x+w))
        else:
            p_r.set((0,0,-1,-1))



        # create view into the gray_img with the bounds of the rough pupil estimation
        pupil_img = gray_img[p_r.lY:p_r.uY,p_r.lX:p_r.uX]

        # pupil_img = cv2.morphologyEx(pupil_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=2)
        if True:
            hist = cv2.calcHist([pupil_img],[0],None,[256],[0,256]) #(images, channels, mask, histSize, ranges[, hist[, accumulate]])
            bins = np.arange(hist.shape[0])
            spikes = bins[hist[:,0]>40] #every color seen in more than 40 pixels
            if spikes.shape[0] >0:
                lowest_spike = spikes.min()
            offset = 40

            ##display the histogram
            sx,sy = 100,1
            colors = ((255,0,0),(0,0,255),(0,255,255))
            h,w,chan = img.shape
            #normalize
            hist *= 1./hist.max()
            for i,h in zip(bins,hist[:,0]):
                c = colors[1]
                cv2.line(img,(w,int(i*sy)),(w-int(h*sx),int(i*sy)),c)
            cv2.line(img,(w,int(lowest_spike*sy)),(int(w-.5*sx),int(lowest_spike*sy)),colors[0])
            cv2.line(img,(w,int((lowest_spike+offset)*sy)),(int(w-.5*sx),int((lowest_spike+offset)*sy)),colors[2])


        # # k-means on the histogram finds peaks but thats no good for us...
        # term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # compactness, bestLabels, centers = cv2.kmeans(data=hist, K=2, criteria=term_crit, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
        # cv2.line(img,(0,1),(int(compactness),1),(0,0,0))
        # good_cluster = np.argmax(centers)
        # # A = hist[bestLabels.ravel() == good_cluster]
        # # B = hist[bestLabels.ravel() != good_cluster]
        # bins = np.arange(hist.shape[0])
        # good_bins = bins[bestLabels.ravel() == good_cluster]
        # good_bins_mean = good_bins.sum()/good_bins.shape[0]
        # good_bins_min = good_bins.min()

        # h,w,chan = img.shape
        # for h, i, label in zip(hist[:,0],range(hist.shape[0]), bestLabels.ravel()):
        #     c = colors[label]
        #     cv2.line(img,(w,int(i*sy)),(w-int(h*sx),int(i*sy)),c)


        else:
            # direct k-means on the image is best but expensive
            Z = pupil_img[::w/30+1,::w/30+1].reshape((-1,1))
            Z = np.float32(Z)
            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 2.0)
            K = 5
            ret,label,center = cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            offset = 0
            center.sort(axis=0)
            lowest_spike =  int(center[1])
            # # Now convert back into uint8, and make original image
            # center = np.uint8(center)
            # res = center[label.flatten()]
            # binary_img = res.reshape((pupil_img.shape))
            # binary_img = bin_thresholding(binary_img,image_upper=res.min()+1)
            # bar.bin_thresh.value = res.min()+1



        bar.bin_thresh.value = lowest_spike
        binary_img = bin_thresholding(pupil_img,image_upper=lowest_spike+offset)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        cv2.dilate(binary_img, kernel,binary_img, iterations=2)
        spec_mask = bin_thresholding(pupil_img, image_upper=250)
        cv2.erode(spec_mask, kernel,spec_mask, iterations=1)

        if bar.blur.value >1:
            pupil_img = cv2.medianBlur(pupil_img,bar.blur.value)

        # create contours using Canny edge dectetion
        contours = cv2.Canny(pupil_img,
                            bar.canny_thresh.value,
                            bar.canny_thresh.value*bar.canny_ratio.value,
                            apertureSize= bar.canny_aperture.value)

        # remove contours in areas not dark enough and where the glint (spectral refelction from IR leds)
        contours = cv2.min(contours, spec_mask)
        contours = cv2.min(contours,binary_img)

        # Ellipse fitting from countours
        result = fit_ellipse(img[r.lY:r.uY,r.lX:r.uX][p_r.lY:p_r.uY,p_r.lX:p_r.uX],
                            contours,
                            binary_img,
                            target_size=bar.pupil_size.value,
                            size_tolerance=bar.pupil_size_tolerance.value)


        # # Vizualizations
        overlay =cv2.cvtColor(pupil_img, cv2.COLOR_GRAY2RGB) #create an RGB view onto the gray pupil ROI
        overlay[:,:,0] = cv2.max(pupil_img,contours) #green channel
        overlay[:,:,2] = cv2.max(pupil_img,binary_img) #blue channel
        overlay[:,:,1] = cv2.min(pupil_img,spec_mask) #red channel

        #draw a blue dotted frame around the automatic pupil ROI in overlay...
        overlay[::2,0] = 0,0,255
        overlay[::2,-1]= 0,0,255
        overlay[0,::2] = 0,0,255
        overlay[-1,::2]= 0,0,255

        # and a solid (white) frame around the user defined ROI
        gray_img[:,0] = 255
        gray_img[:,-1]= 255
        gray_img[0,:] = 255
        gray_img[-1,:]= 255


        if bar.display.value == 0:
            img = img
        elif bar.display.value == 1:
            img[r.lY:r.uY,r.lX:r.uX] = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        elif bar.display.value == 2:
            img[r.lY:r.uY,r.lX:r.uX] = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            img[r.lY:r.uY,r.lX:r.uX][p_r.lY:p_r.uY,p_r.lX:p_r.uX] = overlay
        elif bar.display.value == 3:
            img = cv2.cvtColor(pupil_img, cv2.COLOR_GRAY2RGB)
        else:
            pass

        if result is not None:
            pupil.ellipse, others = result
            pupil.image_coords = r.add_vector(p_r.add_vector(pupil.ellipse['center']))
            #update pupil size,angle and ratio for the ellipse filter algorithm
            bar.pupil_size.value  = bar.pupil_size.value +  .5*(pupil.ellipse['major']-bar.pupil_size.value)
            bar.pupil_ratio.value = bar.pupil_ratio.value + .7*(pupil.ellipse['ratio']-bar.pupil_ratio.value)
            bar.pupil_angle.value = bar.pupil_angle.value + 1.*(pupil.ellipse['angle']-bar.pupil_angle.value)

            # if pupil found tighten the size tolerance
            bar.pupil_size_tolerance.value -=1
            bar.pupil_size_tolerance.value =max(10,min(50,bar.pupil_size_tolerance.value))

            # clamp pupil size
            bar.pupil_size.value = max(20,min(300,bar.pupil_size.value))

            # normalize
            pupil.norm_coords = normalize(pupil.image_coords, (img.shape[1], img.shape[0]),flip_y=True )

            # from pupil to gaze
            pupil.gaze_coords = map_pupil(pupil.norm_coords)
            g_pool.gaze_x.value, g_pool.gaze_y.value = pupil.gaze_coords

        else:
            pupil.ellipse = None
            g_pool.gaze_x.value, g_pool.gaze_y.value = 0.,0.
            pupil.gaze_coords = None #whithout this line the last know pupil position is recorded if none is found

            bar.pupil_size_tolerance.value +=1


        ###CALIBRATION###
        # Initialize Calibration (setup variables and lists)
        if g_pool.calibrate.value and not l_pool.calib_running:
            l_pool.calib_running = True
            pupil.pt_cloud = []

        # While Calibrating...
        if l_pool.calib_running and ((g_pool.pattern_x.value != 0) or (g_pool.pattern_y.value != 0)) and pupil.ellipse:
            pupil.pt_cloud.append([pupil.norm_coords[0],pupil.norm_coords[1],
                                g_pool.pattern_x.value, g_pool.pattern_y.value])

        # Calculate mapping coefs
        if not g_pool.calibrate.value and l_pool.calib_running:
            l_pool.calib_running = 0
            if pupil.pt_cloud: # some data was actually collected
                print "Calibrating with", len(pupil.pt_cloud), "collected data points."
                map_pupil = get_map_from_cloud(np.array(pupil.pt_cloud),g_pool.world_size,verbose=True)
                np.save('cal_pt_cloud.npy',np.array(pupil.pt_cloud))


        ###RECORDING###
        # Setup variables and lists for recording
        if g_pool.pos_record.value and not l_pool.record_running:
            l_pool.record_path = g_pool.eye_rx.recv()
            print "l_pool.record_path: ", l_pool.record_path

            video_path = path.join(l_pool.record_path, "eye.avi")
            #FFV1 -- good speed lossless big file
            #DIVX -- good speed good compression medium file
            if bar.record_eye.value:
                l_pool.writer = cv2.VideoWriter(video_path, cv2.cv.CV_FOURCC(*'DIVX'), bar.fps.value, (img.shape[1], img.shape[0]))
            l_pool.record_positions = []
            l_pool.record_running = True

        # While recording...
        if l_pool.record_running:
            if pupil.gaze_coords is not None:
                l_pool.record_positions.append([pupil.gaze_coords[0], pupil.gaze_coords[1],pupil.norm_coords[0],pupil.norm_coords[1], bar.dt, g_pool.frame_count_record.value])
            if l_pool.writer is not None:
                l_pool.writer.write(cv2.cvtColor(img,cv2.cv.COLOR_BGR2RGB))

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



        ### GL-drawing
        clear_gl_screen()
        draw_gl_texture(img)

        if bar.draw_pupil and pupil.ellipse:
            pts = cv2.ellipse2Poly( (int(pupil.image_coords[0]),int(pupil.image_coords[1])),
                                    (int(pupil.ellipse["axes"][0]/2),int(pupil.ellipse["axes"][1]/2)),
                                    int(pupil.ellipse["angle"]),
                                    0,
                                    360,
                                    15)
            draw_gl_polyline(pts,(1.,0,0,.5))
            draw_gl_point_norm(pupil.norm_coords,(1.,0.,0.,0.5))

        atb.draw()
        glfwSwapBuffers()


    #end while running
    print "EYE Process closed"
    r.save()
    bar.save()
    atb.terminate()
    glfwCloseWindow()
    glfwTerminate()

