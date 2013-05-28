'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import cv2
import numpy as np
from methods import normalize,denormalize
from gl_utils import draw_gl_point,draw_gl_point_norm,draw_gl_polyline_norm

class Ref_Detector(object):
    """
    base class of reference detectors
    build a detector based on this class. It needs to have interfaces
    defined by these methods:
    you NEED to do at least what is done in these fn-prototypes

    ...
    """
    def __init__(self,global_calibrate,shared_x,shared_y):
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = True
        self.shared_x = shared_x
        self.shared_y = shared_y
        self.pos = 0,0 # 0,0 is used to indicate no point detected

    def detect(self,img):
        """
        get called once every frame.
        reference positon need to be pupiled to shard_x and y
        if no referece was found, pupilish 0,0
        """
        self.shared_x.value = 0
        self.shared_y.value = 0

    def advance(self):
        """
        gets called when the user hit spacebar in the world window
        """
        pass

    def new_ref(self,pos):
        """
        gets called when the user click on the wolrd window screen
        """
        pass

    def display(self,img):
        """
        gets called and the end of each image loop. use ref to img data to draw some results
        or
        use opengl calls.
        """
        pass

    def is_done(self):
        """
        gets called after detect()
        return true if the calibration routine has finished.
        the caller will then reduc the ref-count of this instance to 0 triggering __del__

        if your calibration routine does not end by itself, the use will induce the end using the gui
        in which case the ref-count will get 0 as well.
        """
        return False

    def __del__(self):
        self.global_calibrate.value = False

class no_Detector(Ref_Detector):
    """
    docstring for no_Detector
    this dummy class is instaciated when no calibration is running, it does nothing
    """
    def __init__(self, global_calibrate,shared_x,shared_y):
        global_calibrate.value = False
        shared_x.value = 0
        shared_y.value = 0
        self.pos = 0,0

    def detect(self,img):
        pass

    def __del__(self):
        pass

class Black_Dot_Detector(Ref_Detector):
    """docstring for black_dot_detector"""
    def __init__(self, global_calibrate,shared_x,shared_y):
        super(Black_Dot_Detector, self).__init__(global_calibrate,shared_x,shared_y)
        params = cv2.SimpleBlobDetector_Params()
        # params.minDistBetweenBlobs = 500.0
        params.minArea = 100.0
        params.maxArea = 2000.
        params.filterByColor = 1
        params.blobColor = 0
        params.minThreshold = 90.
        self.detector = cv2.SimpleBlobDetector(params)
        self.counter = 0
        self.canditate_points = None

    def detect(self,img):
        s_img = cv2.cvtColor(img[::2,::2],cv2.COLOR_BGR2GRAY)
        self.canditate_points = self.detector.detect(s_img)
        if self.counter and len(self.canditate_points)>0:
            self.counter -= 1
            kp = self.canditate_points[0]
            pt = (kp.pt[0]*2,kp.pt[1]*2)
            self.pos = normalize(pt,(img.shape[1],img.shape[0]),flip_y=True)
            print self.pos
        else:
            self.pos = 0,0
        self.publish()

    def display(self,img):
        for kp in self.canditate_points:
            draw_gl_point((kp.pt[0]*2,kp.pt[1]*2),size=int(kp.size)*4,color=(0.,1.,1.,.5))


    def publish(self):
        self.shared_x.value, self.shared_y.value = self.pos

    def advance(self):
        self.counter = 30*1


class Nine_Point_Detector(Ref_Detector):
    """docstring for Nine_Point_"""
    def __init__(self, global_calibrate,shared_x,shared_y,shared_stage,shared_step,shared_cal9_active,shared_circle_id,auto_advance=False):
        super(Nine_Point_Detector, self).__init__(global_calibrate,shared_x,shared_y)
        self.shared_cal9_active = shared_cal9_active
        self.shared_cal9_active.value = True
        self.shared_stage = shared_stage
        self.shared_step = shared_step
        self.shared_circle_id = shared_circle_id
        self.stage = 0
        self.step = 0
        self.next = False
        self.auto_advance = auto_advance
        self.map = (0, 2, 7, 16, 21, 23, 39, 40, 42)
        self.grid_points = None


    def detect(self,img):
        # Statemachine
        if self.step > 30:
            self.step = 0
            self.stage += 1
            self.next = False
        # done exit now (is_done() will now return True)
        if self.stage > 8:
            return
        # Detection
        self.pos = 0,0
        if self.step in range(10, 25):
            status, grid_points = cv2.findCirclesGridDefault(img, (4,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            if status:
                img_pos = grid_points[self.map[self.stage]][0]
                self.pos = normalize(img_pos, (img.shape[1],img.shape[0]),flip_y=True)
        # Advance
        if self.next or self.auto_advance:
            self.step += 1

        self.publish()

    def is_done(self):
        return self.stage > 8

    def advance(self):
        self.next=True

    def publish(self):
        self.shared_stage.value = self.stage
        self.shared_step.value = self.step
        self.shared_circle_id.value = self.map[self.stage]
        self.shared_x.value, self.shared_y.value = self.pos

    def reset(self):
        self.step = 0
        self.stage = 0
        self.is_done = False
        self.pos = 0,0

    def __del__(self):
        self.reset()
        self.publish()
        self.global_calibrate.value = False
        self.shared_cal9_active.value = False



class Natural_Features_Detector(Ref_Detector):
    """docstring for Natural_Features_Detector"""
    def __init__(self,global_calibrate,shared_x,shared_y,):
        super(Natural_Features_Detector, self).__init__(global_calibrate,shared_x,shared_y)
        self.first_img = None
        self.point = None
        self.count = 0

    def detect(self,img):

        if self.first_img is None:
            self.first_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        if self.count:
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            nextPts, status, err = cv2.calcOpticalFlowPyrLK(self.first_img,gray,self.point,winSize=(100,100))
            if status[0]:
                self.point = nextPts
                self.first_img = gray
                nextPts = nextPts[0]
                self.pos = normalize(nextPts,(img.shape[1],img.shape[0]),flip_y=True)
                self.count -=1
            else:
                self.pos = 0,0
        else:
            self.pos = 0,0

        self.publish()

    def publish(self):
        self.shared_x.value, self.shared_y.value = self.pos

    def new_ref(self,pos):
        self.first_img = None
        self.point = np.array([pos,],dtype=np.float32)
        self.count = 30


class Camera_Intrinsics_Calibration(no_Detector):
    """docstring for Camera_Intrinsics_Calibration"""
    def __init__(self,global_calibrate,shared_x,shared_y):
        super(Camera_Intrinsics_Calibration, self).__init__(global_calibrate,shared_x,shared_y)
        self.collect_new = False
        self.obj_grid = _gen_pattern_grid((4, 11))
        self.img_points = []
        self.obj_points = []
        self.count = 10
        self.img_shape = None

    def detect(self,img):
        if self.collect_new:
            status, grid_points = cv2.findCirclesGridDefault(img, (4,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            if status:
                self.img_points.append(grid_points)
                self.obj_points.append(self.obj_grid)
                self.collect_new = False
                self.count -=1
                print "%i of Patterns still to capture" %(self.count)
                self.img_shape = img.shape

    def advance(self):
        self.collect_new = True

    def is_done(self):
        return self.count <= 0

    def __del__(self):
        if not self.count:
            camera_matrix, dist_coefs = _calibrate_camera(np.asarray(self.img_points),
                                                        np.asarray(self.obj_points),
                                                        (self.img_shape[1], self.img_shape[0]))
            np.save("camera_matrix.npy", camera_matrix)
            np.save("dist_coefs.npy", dist_coefs)
            print "Camera Instrinsics calculated and saved to file"
        else:
            print "Waring: Not enough images captured, please try again"

###private helper fns...
def _calibrate_camera(img_pts, obj_pts, img_size):
    # generate pattern size
    camera_matrix = np.zeros((3,3))
    dist_coef = np.zeros(4)
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts,
                                                    img_size, camera_matrix, dist_coef)
    return camera_matrix, dist_coefs

def _gen_pattern_grid(size=(4,11)):
    pattern_grid = []
    for i in xrange(size[1]):
        for j in xrange(size[0]):
            pattern_grid.append([(2*j)+i%2,i,0])
    return np.asarray(pattern_grid, dtype='f4')


