'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import platform,sys,os
os_name = platform.system()
from time import sleep

import subprocess as sp
import signal


#logging
import logging
logger = logging.getLogger(__name__)

import pyaudio as pa



audio_modes = ('voice and sound', 'sound only','voice only','silent')
default_audio_mode = audio_modes[0]
audio_mode = default_audio_mode

def set_audio_mode(new_mode):
    '''a save way to set the audio mode
    '''
    if new_mode in ('voice and sound', 'silent','voice only', 'sound only'):
        global audio_mode
        audio_mode = new_mode

def get_input_devices_by_api(api):
    pyaudio = pa.PyAudio()
    ds_info = pyaudio.get_host_api_info_by_type(api)
    dev_list = [dev_info for dev_info in [pyaudio.get_device_info_by_host_api_device_index(ds_info['index'], dev_idx) for dev_idx in range(ds_info['deviceCount'])] if dev_info['maxInputChannels'] > 0]
    pyaudio.terminate()
    return dev_list



# OS specific audio players via terminal
if os_name == "Linux":

    # if getattr(sys, 'frozen', False):
    #     # we are running in a |PyInstaller| bundle
    #     ffmpeg_bin = os.path.join(sys._MEIPASS,'avconv')
    # else:
    #     # we are running in a normal Python environment
    ffmpeg_bin = "avconv"
    arecord_bin = 'arecord'

    if platform.linux_distribution()[0] in ('Ubuntu', 'debian'):
        def beep():
            if 'sound' in audio_mode:
                try:
                    sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/message.ogg"])
                except OSError:
                    logger.warning("Soundfile not found.")

        def tink():
            if 'sound' in audio_mode:
                try:
                    sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/button-pressed.ogg"])
                except OSError:
                    logger.warning("Soundfile not found.")

        def say(message):
            if 'voice' in audio_mode:
                try:
                    sp.Popen(["spd-say", message])
                except OSError:
                    install_warning = "could not say: '{}'. Please install spd-say if you want Pupil capture to speek to you."
                    logger.warning(install_warning.format(message))
    else:
        def beep():
            if 'sound' in audio_mode:
                print('\a')

        def tink():
            if 'sound' in audio_mode:
                print('\a')

        def say(message):
            if 'sound' in audio_mode:
                print('\a')
                print(message)

    class Audio_Input_Dict(dict):
        """docstring for Audio_Input_Dict"""
        def __init__(self):
            super().__init__()
            self['No Audio'] = -1

            for dev_info in get_input_devices_by_api(pa.paALSA):
                #print(dev_info)
                if "hw:" in dev_info['name']:
                    self[dev_info['name']] = dev_info['name']

elif os_name == "Darwin":

    class Audio_Input_Dict(dict):
        """docstring for Audio_Input_Dict"""
        def __init__(self):
            super().__init__()
            self['No Audio'] = -1
            for idx, dev_info in enumerate(get_input_devices_by_api(pa.paCoreAudio)):
                if 'NoMachine' not in dev_info['name']:
                    self[dev_info['name']] = idx





    def beep():
        if 'sound' in audio_mode:
            sp.Popen(["afplay", "/System/Library/Sounds/Pop.aiff"])

    def tink():
        if 'sound' in audio_mode:
            sp.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])

    def say(message):
        if 'voice' in audio_mode:
            sp.Popen(["say", message, "-v" "Victoria"])

elif os_name == "Windows":
    def beep():
        if 'sound' in audio_mode:
            print('\a')

    def tink():
        if 'sound' in audio_mode:
            print('\a')

    def say(message):
        if 'voice' in audio_mode:
            print('\a')
            print(message)


    class Audio_Input_Dict(dict):
        """docstring for Audio_Input_Dict"""
        def __init__(self):
            super().__init__()
            for dev_info in get_input_devices_by_api(pa.paDirectSound):
                self[dev_info['name']] = dev_info['name']
            self['No Audio'] = None

else:
    def beep():
        if 'sound' in audio_mode:
            print('\a')

    def tink():
        if 'sound' in audio_mode:
            print('\a')

    def say(message):
        if 'voice' in audio_mode:
            print('\a')
            print(message)


    class Audio_Input_Dict(dict):
        """docstring for Audio_Input_Dict"""
        def __init__(self):
            super().__init__()
            self['No Audio'] = -1


    class Audio_Capture(object):
        """docstring for audio_capture"""
        def __init__(self, audio_src_idx=0, out_file='out.wav'):
            super().__init__()
            logger.debug("Audio Capture not implemented on this OS")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print(Audio_Input_Dict())


    # beep()
    # sleep(1)
    # tink()
    # cap = Audio_Capture('test.mp3')
    # say("Hello, I am Pupil's audio module.")
    # sleep(3)
    # cap = None
