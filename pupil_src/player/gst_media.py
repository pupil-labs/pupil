

import os
from gi.repository import Gst
import cv2
import numpy
import logging
import Queue
logger = logging.getLogger(__name__)


Gst.init(None)


def gst_to_opencv(sample):
    buf = sample.get_buffer()
    caps = sample.get_caps()
    arr = numpy.ndarray(
        (caps.get_structure(0).get_value('height'),
         caps.get_structure(0).get_value('width'),
         3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=numpy.uint8)
    return arr




# non os specific defines
class Frame(object):
    """docstring of Frame"""
    def __init__(self, timestamp,img,compressed_img=None, compressed_pix_fmt=None):
        self.timestamp = timestamp
        self.img = img
        self.compressed_img = compressed_img
        self.compressed_pix_fmt = compressed_pix_fmt



class Gst_Player(object):
    """this is our media player implemented using gstreamer"""
    def __init__(self, src_file):

        if not os.path.isfile(src_file):
            logger.error('Could not locate VideoFile %s'%src)
            return

        self.src_file = src_file

        # Create the elements
        self.source = Gst.ElementFactory.make("playbin", "source")
        self.sink = Gst.ElementFactory.make("appsink", "sink")
        # Create the empty pipeline
        self.pipeline = Gst.Pipeline.new("pipeline")


        if not self.source or not self.sink or not self.pipeline:
            logger.error("Not all Gstreamer elements could be created.")
            return

        self.source.set_property("uri", "file://"+self.src_file)
        self.sink.set_property("emit-signals", True)

        caps = Gst.caps_from_string("video/x-raw, format=BGR") #width=1280, height=720
        self.sink.set_property("caps", caps)
        self.source.set_property("video-sink", self.sink)
        self.sink.connect("new-sample", self.on_new_sample, self.sink)
        self.sink.set_property("max-buffers", 1)

        # Build the pipeline
        self.pipeline.add(self.source)

        self.bus = self.pipeline.get_bus()
        self.bus.set_sync_handler(self.on_msg,None)

        # we use a one slot queue to pass img from gst callback to user thread
        self.queue  = Queue.Queue(maxsize=1)
        self.playing = False



    def play(self):
        # Start playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Unable to set the pipeline to the playing state.")
        self.playing = True


    def pause(self):
        # Pause playing
        ret = self.pipeline.set_state(Gst.State.PAUSED)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Unable to set the pipeline to the paused state.")
        self.playing = False


    def on_new_sample(self, sink, data):
        sample = sink.emit("pull-sample")
        img = gst_to_opencv(sample)
        ts =   sample.get_buffer().pts
        img = Frame(ts, img)

        # we really do want the frame index, but this is not (yet) implemented.


        # logger.debug("put img")

        # dropframe mode
        # try:
        #     #if we have a uncollected frame in the queue, drop it
        #     self.queue.get_nowait()
        #     logger.warn("dropping frame")
        # except Queue.Empty:
        #     pass
        # self.queue.put_nowait(img)

        #wait mode, this can stall the gst pipeline.
        # self.queue.put(img)

        #timemout mode, waits for consumer but times out to avaid complete stall
        try:
            self.queue.put(img,timeout=1.)
        except Queue.Full:
            logger.warn("Dropped Frame!")
        return Gst.FlowReturn.OK

    def on_msg(self,bus,message,usr_data):
        """
        parsing messages comming from gst bus.
        """
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("Error received from element %s: %s" % (
                message.src.get_name(), err))
            logger.debug("Debugging information: %s" % debug)
        elif message.type == Gst.MessageType.EOS:
            logger.info("End-Of-Stream reached.")
        elif message.type == Gst.MessageType.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old_state, new_state, pending_state = message.parse_state_changed()
                logger.info("Pipeline state changed from %s to %s." %
                       (old_state.value_nick, new_state.value_nick))
        elif message.type == Gst.MessageType.TAG:
            pass
            # print message.parse_tag().to_string()
        else:
            logger.debug(str(message.type))
            # print("Unexpected message received.")
        return Gst.FlowReturn.OK

    def read(self):
        # logger.debug("get img")
        try:
            img = self.queue.get(timeout = 1)
        except Queue.Empty:
            logger.error("no Video")
            return None
        return img

    def cleanup(self):
        # Free resources
        self.pipeline.set_state(Gst.State.NULL)




def bench():
    import cv2
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    cap = Gst_Player("/Users/mkassner/Pupil/pupil_code/recordings/2013_12_11/000/world.avi")
    cap.play()
    for x in range(30):
        frame = cap.read()
        # print frame.timestamp/(float(Gst.SECOND))*31
        cv2.imshow("Gstreamer test",frame.img)
        cv2.waitKey(1)

    # cap.read() #flush out old frame
    cap.pause()
    cap.cleanup()
    cv2.destroyWindow("Gstreamer test")



if __name__ == '__main__':
    # bench()

    import cProfile,subprocess,os
    cProfile.runctx("bench()",{},locals(),"world.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
    print "created cpu time graph for world process. Please check out the png next to the world.py file"






