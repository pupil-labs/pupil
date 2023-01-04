"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import platform
import subprocess as sp
from time import sleep

logger = logging.getLogger(__name__)


AUDIO_MODE_SOUND_ONLY = "sound only"
AUDIO_MODE_SILENT = "silent"


def get_audio_mode_list():
    return (
        AUDIO_MODE_SOUND_ONLY,
        AUDIO_MODE_SILENT,
    )


def get_default_audio_mode():
    return AUDIO_MODE_SOUND_ONLY


_audio_mode = get_default_audio_mode()


def get_audio_mode():
    global _audio_mode
    return _audio_mode


def set_audio_mode(new_mode):
    """a save way to set the audio mode"""
    if new_mode in get_audio_mode_list():
        global _audio_mode
        _audio_mode = new_mode
    else:
        logger.warning(f'Unknown audio mode: "{new_mode}"')


def is_voice_enabled() -> bool:
    return "voice" in get_audio_mode()


def is_sound_enabled() -> bool:
    return "sound" in get_audio_mode()


def beep():
    if is_sound_enabled():
        _platform_specific_switch(
            linux_fn=_linux_beep,
            darwin_fn=_darwin_beep,
            windows_fn=_windows_beep,
            unknown_fn=_unknown_beep,
        )


def tink():
    if is_sound_enabled():
        _platform_specific_switch(
            linux_fn=_linux_tink,
            darwin_fn=_darwin_tink,
            windows_fn=_windows_tink,
            unknown_fn=_unknown_tink,
        )


def _platform_specific_switch(linux_fn, darwin_fn, windows_fn, unknown_fn, **kwargs):
    os_name = platform.system()
    if os_name == "Linux":
        linux_fn(**kwargs)
    elif os_name == "Darwin":
        darwin_fn(**kwargs)
    elif os_name == "Windows":
        windows_fn(**kwargs)
    else:
        unknown_fn(**kwargs)


def _linux_beep():
    try:
        sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/message.ogg"])
    except OSError:
        logger.warning("Soundfile not found.")
        print("\a")


def _linux_tink():
    try:
        sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/button-pressed.ogg"])
    except OSError:
        print("\a")


def _darwin_beep():
    sp.Popen(["afplay", "/System/Library/Sounds/Pop.aiff"])


def _darwin_tink():
    sp.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])


def _windows_beep():
    print("\a")


def _windows_tink():
    print("\a")


def _unknown_beep():
    print("\a")


def _unknown_tink():
    print("\a")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    beep()
    sleep(1)

    tink()
    sleep(1)
