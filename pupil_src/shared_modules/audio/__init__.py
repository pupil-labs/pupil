"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import platform, sys, os
import subprocess as sp
from time import sleep


logger = logging.getLogger(__name__)


AUDIO_MODE_VOICE_AND_SOUND = "voice and sound"
AUDIO_MODE_SOUND_ONLY = "sound only"
AUDIO_MODE_VOICE_ONLY = "voice only"
AUDIO_MODE_SILENT = "silent"


def get_audio_mode_list():
    return (
        AUDIO_MODE_VOICE_AND_SOUND,
        AUDIO_MODE_SOUND_ONLY,
        AUDIO_MODE_VOICE_ONLY,
        AUDIO_MODE_SILENT,
    )


def get_default_audio_mode():
    return AUDIO_MODE_VOICE_AND_SOUND


_audio_mode = get_default_audio_mode()


def get_audio_mode():
    global _audio_mode
    return _audio_mode


def set_audio_mode(new_mode):
    """a save way to set the audio mode
    """
    if new_mode in get_audio_mode_list():
        global _audio_mode
        _audio_mode = new_mode
    else:
        logger.warning(f'Unknown audio mode: "{new_mode}"')


def is_voice_enabled() -> bool:
    return "voice" in get_audio_mode()


def is_sound_enabled() -> bool:
    return "sound" in get_audio_mode()


# OS specific audio players via terminal
os_name = platform.system()

if os_name == "Linux":

    if platform.linux_distribution()[0] in ("Ubuntu", "debian"):

        def beep():
            if is_sound_enabled():
                try:
                    sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/message.ogg"])
                except OSError:
                    logger.warning("Soundfile not found.")

        def tink():
            if is_sound_enabled():
                try:
                    sp.Popen(
                        ["paplay", "/usr/share/sounds/ubuntu/stereo/button-pressed.ogg"]
                    )
                except OSError:
                    logger.warning("Soundfile not found.")

        def say(message):
            if is_voice_enabled():
                try:
                    sp.Popen(["spd-say", message])
                except OSError:
                    install_warning = "could not say: '{}'. Please install spd-say if you want Pupil capture to speek to you."
                    logger.warning(install_warning.format(message))

    else:

        def beep():
            if is_sound_enabled():
                print("\a")

        def tink():
            if is_sound_enabled():
                print("\a")

        def say(message):
            if is_voice_enabled():
                print("\a")
                print(message)

elif os_name == "Darwin":

    def beep():
        if is_sound_enabled():
            sp.Popen(["afplay", "/System/Library/Sounds/Pop.aiff"])

    def tink():
        if is_sound_enabled():
            sp.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])

    def say(message):
        if is_voice_enabled():
            sp.Popen(["say", message, "-v" "Victoria"])

elif os_name == "Windows":

    def beep():
        if is_sound_enabled():
            print("\a")

    def tink():
        if is_sound_enabled():
            print("\a")

    def say(message):
        if is_voice_enabled():
            print("\a")
            print(message)

else:

    def beep():
        if is_sound_enabled():
            print("\a")

    def tink():
        if is_sound_enabled():
            print("\a")

    def say(message):
        if is_voice_enabled():
            print("\a")
            print(message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    beep()
    sleep(1)

    tink()
    sleep(1)

    say("Hello, I am Pupil's audio module.")
    sleep(3)
