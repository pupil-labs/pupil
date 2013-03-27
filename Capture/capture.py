import numpy as np
import cv2

class capture():
    """docstring for capture"""
    def __init__(self, src, size=None):
        self.src = src
        self.auto_rewind = False
        if isinstance(self.src, int) or isinstance(self.src, str):
            #set up as cv2 capture
            self.VideoCapture = cv2.VideoCapture(src)
            self.set_size(size)
            self.get_frame = self.VideoCapture.read
        elif src == None:
            self.VideoCapture = None
            self.get_frame = None
        else:
            #set up as pipe
            self.VideoCapture = src
            self.size = size
            self.np_size = size[::-1]
            self.VideoCapture.send(self.size) #send desired size to the capture function in the main
            self.get_frame = self.VideoCapture.recv #retrieve first frame

    def set_size(self,size):
        if size is not None:
            if isinstance(self.src, int):
                self.size = size
                width,height = size
                self.VideoCapture.set(3, width)
                self.VideoCapture.set(4, height)
            else:
                self.size = self.VideoCapture.get(3),self.VideoCapture.get(4)
            self.np_size = self.size[::-1]

    def read(self):
        s, img =self.get_frame()
        if  self.auto_rewind and not s:
            self.rewind()
            s, img = self.get_frame()
        return s,img

    def read_RGB(self):
        s,img = self.read()
        if s:
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR,img)
        return s,img

    def read_HSV(self):
        s,img = self.read()
        if s:
            cv2.cvtColor(img, cv2.COLOR_RGB2HSV,img)
        return s,img

    def rewind(self):
        self.VideoCapture.set(1,0) #seek to 0



def local_grab_threaded(pipe_world,src_id_world,pipe_eye,src_id_eye,g_pool):
    import threading
    from time import sleep, time

    class capture_thread(threading.Thread):
        """docstring for capture_thread"""
        def __init__(self,pipe,src):
            threading.Thread.__init__(self)
            self.pipe = pipe
            self.src = src
            self.cap_init(src)

        def cap_init(self,src_id):
            self.cap = cv2.VideoCapture(src_id)
            size = self.pipe.recv() #recieve desired size from caputure instance from inside the other process.
            self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, size[1])
        def run(self):
            tick = time()
            while not g_pool.quit.value:
                tick = time()
                self.pipe.send(self.cap.read())
                sleep(max(0,1/31.-(time()-tick)))

    """grab:
        - Initialize a camera feed
        -this is needed for certain cameras that have to run in the main loop.
        - it pushes image frames to the capture class
          that is initialize with one pipeend as the source
    """
    thread1 = capture_thread(pipe_world,src_id_world)
    thread2 = capture_thread(pipe_eye,src_id_eye)
    thread1.start() # This actually causes the thread to run
    thread2.start()
    thread1.join()  # This waits until the thread has completed
    thread2.join()

    print "Local Grab exit"



def local_grab(pipe,src_id,g_pool):
    """grab:
        - Initialize a camera feed
        -this is needed for certain cameras that have to run in the main loop.
        - it pushed image frames to the capture class
            that it initialize with one pipeend as the source
    """

    quit = g_pool.quit
    cap = cv2.VideoCapture(src_id)
    size = pipe.recv() #recieve designed size from caputure instance from inside the other process.
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, size[1])

    while not quit.value:
        try:
            pipe.send(cap.read())
        except:
            pass
    print "Local Grab exit"