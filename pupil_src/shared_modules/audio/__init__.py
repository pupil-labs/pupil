'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import platform
os_name = platform.system()
del platform
from time import sleep

import subprocess as sp
import signal

# OS specific audio players via terminal
if os_name == "Linux":
    def beep():
        sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/message.ogg"])

    def tink():
        sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/message.ogg"])

    def say(message):
        sp.Popen(["spd-say", message])


    class audio_capture(object):
        """docstring for audio_capture"""
        def __init__(self, out_file):
            super(audio_capture, self).__init__()
            self.out_path = out_path
            ffmpeg_bin = 'ffmpeg'
            command = [ ffmpeg_bin,
                    '-i', 'hw:0,0',
                    '-f', 'alsa',
                    out_file]
            self.process =  sp.Popen(command)

        def __del__(self):
            self.process.send_signal(signal.SIGINT)


elif os_name == "Darwin":
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

if __name__ == '__main__':
    beep()
    sleep(1)
    tink()
    sleep(1)
    say("Hello, I am Pupil's audio module.")