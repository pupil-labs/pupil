

import os
from gi.repository import Gst
import cv2
import numpy
import logging
import Queue
from time import sleep
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

def caps_string_to_dict(string):
    items = string.split(", ")
    media_type = items.pop(0)
    items = [pair.split("=") for pair in items]
    d = dict(items)
    d["media_type"] = media_type
    for k,v in d.iteritems():
        if v.startswith("(string)"):
            v = v.replace("(string)","")
            d[k] = v
        elif v.startswith("(int)"):
            v = v.replace("(int)","")
            d[k] = int(v)
        elif v.startswith("(fraction)"):
            v = v.replace("(fraction)","")
            enum,denom = v.split("/")
            d[k] = float(enum)/float(denom)
        else:
            pass
    return d

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


        self.media_info = None
        self.height = None
        self.width = None
        self.fps  = None
        if not os.path.isfile(src_file):
            logger.error('Could not locate VideoFile %s'%src)
            return

        self.src_file = src_file

        # Create the elements
        self.source = Gst.ElementFactory.make("uridecodebin", "source")
        self.convert = Gst.ElementFactory.make("videoconvert", "convert")
        self.sink = Gst.ElementFactory.make("appsink", "sink")
        # Create the empty pipeline
        self.pipeline = Gst.Pipeline.new("pipeline")


        if not self.source or not self.sink or not self.convert or not self.pipeline:
            logger.error("Not all Gstreamer elements could be created.")
            return

        self.source.set_property("uri", "file://"+self.src_file)
        self.source.connect("pad-added", self.on_pad_added, self.convert)

        caps = Gst.caps_from_string("video/x-raw, format=BGR") #width=1280, height=720
        self.sink.set_property("caps", caps)
        self.sink.set_property("emit-signals", True)
        self.sink.set_property("max-buffers", 1)
        # self.sink.set_property("drop", True)
        # self.sink.set_property("sync", False)
        self.sink.connect("new-sample", self.on_new_sample, self.sink)



        # Build the pipeline
        self.pipeline.add(self.source)
        self.pipeline.add(self.convert)
        self.pipeline.add(self.sink)
        self.bus = self.pipeline.get_bus()
        self.bus.set_sync_handler(self.on_msg,None)


        if not Gst.Element.link(self.convert, self.sink):
            logger.error("GST Elements could not be linked.")
            return

        # we use a one slot queue to pass img from gst callback to user thread
        self.queue  = Queue.Queue(maxsize=1)
        self.playing = False

        self.ready()
        self.play()
        self.wait_for_media_info()

    def wait_for_media_info(self,tries=10):
        for x in range(tries):
            if self.media_info:
                return True
            sleep(.1)
        logger.error("Could not get media info in time.")
        return False



    def ready(self):
        # Start playing
        ret = self.pipeline.set_state(Gst.State.READY)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Unable to set the pipeline to the playing state.")
        self.playing = True


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


    def seek_frame(self,frame_idx):
        logger.debug("seeking to Frame: %s" %frame_idx)

        #manual flusing of the pipeline. Should be done smarter
        self.read(0.001)
        if not self.pipeline.seek_simple(Gst.Format.DEFAULT,Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,frame_idx):
            logger.error("Seek failed please report")
        #manual flusing of the frame stuck in callback.
        self.read(0.001)


    def seek_frame_no_manual_flush(self,frame_idx):
        logger.debug("seeking to Frame: %s" %frame_idx)

        if not self.pipeline.seek_simple(Gst.Format.DEFAULT,Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,frame_idx):
            logger.error("Seek failed please report")


    def on_pad_added(self,src, new_pad, _):
        logger.debug("Received new pad '%s' from '%s':" % (new_pad.get_name(),
              src.get_name()))
        # If our converter is already linked, we have nothing to do here
        if new_pad.is_linked():
            logger.debug("We are already linked. Ignoring.")
            return

        # Check the new pad's type
        new_pad_type = new_pad.query_caps(None).to_string()
        if not new_pad_type.startswith("video/x-raw"):
            logger.debug("  It has type '%s' which is not raw video. Ignoring." %
                  new_pad_type)
            return


        #adding media info:
        media_info =  caps_string_to_dict(new_pad_type)
        self.media_info = media_info
        self.height = media_info['height']
        self.width =  media_info['width']
        self.fps = media_info['framerate']

        # Attempt the link
        ret = new_pad.link(self.convert.get_static_pad("sink"))
        return


    def on_new_sample(self, sink, data):
        sample = sink.emit("pull-sample")
        img = gst_to_opencv(sample)
        ts =   sample.get_buffer().pts
        img = Frame(ts,img)
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

        # wait mode, this can stall the gst pipeline. Need to flush queue before anything else.
        self.queue.put(img)

        #timemout mode, waits for consumer but times out to avaid complete stall
        # try:
        #     self.queue.put(img,timeout=10.)
        # except Queue.Full:
        #     logger.warn("Dropped Frame!")

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
            pass
            # logger.debug(str(message.type))
            # print("Unexpected message received.")
        return Gst.FlowReturn.OK

    def read(self, timeout = 1):
        # logger.debug("get img")
        try:
            img = self.queue.get(timeout)
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
    for x in range(300):
        frame = cap.read()
        print frame.timestamp/(float(Gst.SECOND))
        # cv2.imshow("Gstreamer test",frame.img)
        # cv2.waitKey(1)
        # if x == 100:
        #     cap.seek_frame(0)

    # cap.read() #flush out old frame
    cap.pause()
    cap.cleanup()
    cv2.destroyWindow("Gstreamer test")



def bench_cv():
    import cv2
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    cap = cv2.VideoCapture("/Users/mkassner/Pupil/pupil_code/recordings/2013_12_11/000/world.avi")
    for x in range(300):
        s,frame = cap.read()
        print frame.shape
        # cv2.imshow("Gstreamer test",frame.img)
        # cv2.waitKey(1)
        # if x == 100:
            # cap.seek_frame(0)

    # cap.read() #flush out old frame
    cv2.destroyWindow("Gstreamer test")

def bench_compare():
    bench_cv()
    bench()

if __name__ == '__main__':

    import cProfile,subprocess,os
    cProfile.runctx("bench_compare()",{},locals(),"world.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
    print "created cpu time graph for world process. Please check out the png next to the world.py file"

