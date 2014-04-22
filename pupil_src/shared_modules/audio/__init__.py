'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import platform,sys,os
os_name = platform.system()
del platform
from time import sleep

import subprocess as sp
import signal


#logging
import logging
logger = logging.getLogger(__name__)




# OS specific audio players via terminal
if os_name == "Linux":

    if getattr(sys, 'frozen', False):
        # we are running in a |PyInstaller| bundle
        ffmpeg_bin = os.path.join(sys._MEIPASS,'avconv')
    else:
        # we are running in a normal Python environment
        ffmpeg_bin = "avconv"



    def beep():
        sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/message.ogg"])

    def tink():
        sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/message.ogg"])

    def say(message):
        sp.Popen(["spd-say", message])


    class Audio_Capture(object):
        """docstring for audio_capture"""
        def __init__(self, out_file):
            super(Audio_Capture, self).__init__()
            command = [ ffmpeg_bin,
                    '-f', 'alsa',
                    '-i', 'hw:0,0',
                    '-v', 'error',
                    out_file]
            try:
                self.process =  sp.Popen(command,stdout=sp.PIPE,stderr=sp.PIPE)
            except OSError:
                logger.debug("Audio module for recording not found. Not recording audio.")
                self.process = None
                return
            logger.debug("stared recording mic to %s with avconv process, pid: %s"%(out_file,self.process.pid))

        def __del__(self):
            if self.process:
                self.process.send_signal(signal.SIGINT)
                out,err = self.process.communicate()
                if err:
                    logger.warning('Audio recording failed. Error:\n %s'%err)
                if out:
                    logger.debug('Audio recording. INFO:\n %s'%out)
                logger.debug("Finished recording mic with avconv process, pid: %s"%(self.process.pid))


elif os_name == "Darwin":


    if getattr(sys, 'frozen', False):
        # we are running in a |PyInstaller| bundle
        sox_bin = os.path.join(sys._MEIPASS,'sox')
    else:
        # we are running in a normal Python environment
        sox_bin = "sox"


    class Audio_Capture(object):
        """docstring for audio_capture"""
        def __init__(self, out_file):
            super(Audio_Capture, self).__init__()
            command = [ sox_bin,
                    '-d','-q', out_file]
            try:
                self.process =  sp.Popen(command,stdout=sp.PIPE,stderr=sp.PIPE)
            except OSError:
                logger.debug("Audio module for recording not found. Not recording audio.")
                self.process = None
                return
            logger.debug("stared recording mic to %s with SOX process, pid: %s"%(out_file,self.process.pid))

        def __del__(self):
            if self.process:
                self.process.send_signal(signal.SIGINT)
                out,err = self.process.communicate()
                if err:
                    logger.warning('Audio recording failed. Error:\n %s'%err)
                if out:
                    logger.debug('Audio recording. INFO:\n %s'%out)
                logger.debug("Finished recording mic with SOX process, pid: %s"%(self.process.pid))



    def beep():
        sp.Popen(["afplay", "/System/Library/Sounds/Pop.aiff"])

    def tink():
        sp.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])

    def say(message):
        sp.Popen(["say", message, "-v" "Victoria"])

else:
    def beep():
        print '\a'

    def tink():
        print '\a'

    def say(message):
        print '\a'
        print message

    class Audio_Capture(object):
        """docstring for audio_capture"""
        def __init__(self, out_file):
            super(Audio_Capture, self).__init__()
            logger.debug("Audio Capture not implemented on this OS")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    beep()
    sleep(1)
    tink()
    cap = Audio_Capture('test.mp3')
    say("Hello, I am Pupil's audio module.")
    sleep(3)
    cap = None