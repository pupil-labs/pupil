import abc
from pyglui import ui


def no_valid_video_elements():
    return (
        ui.Info_Text("No valid overlay video loaded yet."),
        ui.Info_Text("To load a video, drag and drop it onto Player."),
        ui.Info_Text(
            "Valid overlay videos conform to the Pupil data format and "
            "their timestamps are in sync with the opened recording."
        ),
    )


def generic_overlay_elements(video_path, config):
    return (
        ui.Info_Text("Loaded video: {}".format(video_path)),
        ui.Slider(
            "value",
            config.scale,
            label="Scale",
            min=config.scale.constraint.low,
            max=config.scale.constraint.high,
            step=0.05,
        ),
        ui.Slider(
            "value",
            config.alpha,
            label="Transparency",
            min=config.alpha.constraint.low,
            max=config.alpha.constraint.high,
            step=0.05,
        ),
        ui.Switch("value", config.hflip, label="Flip horizontally"),
        ui.Switch("value", config.vflip, label="Flip vertically"),
    )
