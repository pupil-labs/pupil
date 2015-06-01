'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import platform,sys,os
os_name = platform.system()
from time import sleep

import subprocess as sp
import signal


#logging
import logging
logger = logging.getLogger(__name__)




# OS specific audio players via terminal
if os_name == "Linux":

    # if getattr(sys, 'frozen', False):
    #     # we are running in a |PyInstaller| bundle
    #     ffmpeg_bin = os.path.join(sys._MEIPASS,'avconv')
    # else:
    #     # we are running in a normal Python environment
    ffmpeg_bin = "avconv"
    arecord_bin = 'arecord'

    if 'Ubuntu' in platform.linux_distribution():
        def beep():
            try:
                sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/message.ogg"])
            except OSError:
                logger.warning("Soundfile not found.")
        def tink():
            try:
                sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/button-pressed.ogg"])
            except OSError:
                logger.warning("Soundfile not found.")

        def say(message):
            try:
                sp.Popen(["spd-say", message])
            except OSError:
                logger.warning("could not say: '%s'. Please install spd-say if you want Pupil capture to speek to you.")
    else:
        def beep():
            print '\a'

        def tink():
            print '\a'

        def say(message):
            print '\a'
            print message


    class Audio_Input_Dict(dict):
        """docstring for Audio_Input_Dict"""
        def __init__(self):
            super(Audio_Input_Dict, self).__init__()
            self['No Audio'] =-1
            try:
                ret = sp.check_output([arecord_bin,"-l"])
            except OSError:
                logger.warning("Could not enumerate audio input devices. Calling arecord failed.")
                return
            '''
            **** List of CAPTURE Hardware Devices ****
            card 0: AudioPCI [Ensoniq AudioPCI], device 0: ES1371/1 [ES1371 DAC2/ADC]
              Subdevices: 1/1
              Subdevice #0: subdevice #0
            card 1: C930e [Logitech Webcam C930e], device 0: USB Audio [USB Audio]
              Subdevices: 1/1
              Subdevice #0: subdevice #0

            '''
            # logger.debug(ret)

            lines = ret.split("\n")
            # logger.debug(lines)
            devices = [l.split(',')[0] for l in lines[1:] if not l.startswith("  ") and l]

            device_names = [w.split(":")[-1] for w in devices]
            device_names = [w[1:] for w in device_names]
            for d,idx in zip(device_names,range(len(device_names))):
                self[d] = idx


    class Audio_Capture(object):
        """docstring for audio_capture"""
        def __init__(self,audio_src_idx=0, out_file='out.wav'):
            super(Audio_Capture, self).__init__()
            # command = [ ffmpeg_bin,
            #         '-f', 'alsa',
            #         '-i', 'hw:0,0',
            #         '-v', 'error',
            #         out_file]
            logger.debug("Recording audio  using 'arecord' device %s to %s"%(audio_src_idx,out_file))
            command = [ arecord_bin,
                        '-D', 'plughw:'+str(audio_src_idx)+',0',
                        '-r', '16000',
                        '-f', 'S16_LE',
                        '-c', '2',
                        '-q', #we use quite because signint will write into stderr and we use this to check for real errors.
                        out_file]
            try:
                self.process =  sp.Popen(command,stdout=sp.PIPE,stderr=sp.PIPE)
            except OSError:
                logger.debug("Audio module for recording not found. Not recording audio. please do 'sudo apt-get install libav-tools'.")
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

    class Audio_Input_Dict(dict):
        """docstring for Audio_Input_Dict"""
        def __init__(self):
            super(Audio_Input_Dict, self).__init__()
            self['No Audio'] = -1
            self['Default Mic'] = 0

    # if getattr(sys, 'frozen', False):
    #     # we are running in a |PyInstaller| bundle
    #     sox_bin = os.path.join(sys._MEIPASS,'sox')
    # else:
    #     # we are running in a normal Python environment
    sox_bin = "sox"


    class Audio_Capture(object):
        """docstring for audio_capture"""
        def __init__(self,audio_src_idx=0, out_file='out.wav'):
            super(Audio_Capture, self).__init__()
            logger.debug("Recording audio  using 'sox' device %s to %s"%(audio_src_idx,out_file))

            command = [ sox_bin,
                    '-d','-q', out_file]
            try:
                self.process =  sp.Popen(command,stdout=sp.PIPE,stderr=sp.PIPE)
            except OSError:
                logger.warning("Audio module for recording not found. Not recording audio. Please do 'brew install sox' ")
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


    class Audio_Input_Dict(dict):
        """docstring for Audio_Input_Dict"""
        def __init__(self):
            super(Audio_Input_Dict, self).__init__()
            self['No Audio'] = -1


    class Audio_Capture(object):
        """docstring for audio_capture"""
        def __init__(self, audio_src_idx=0, out_file='out.wav'):
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
    print Audio_Input_Dict()
