import abc
from storage import SingleFileStorage
from video_overlay.models.config import Configuration
from video_overlay.workers.overlay_renderer import OverlayRenderer


class OverlayManager(SingleFileStorage):
    def __init__(self, rec_dir, plugin):
        super().__init__(rec_dir, plugin)
        self._overlays = []
        self._load_from_disk()

    @property
    def _storage_file_name(self):
        return "video_overlays.msgpack"

    @property
    def _item_class(self):
        return Configuration

    def add(self, item):
        overlay = OverlayRenderer(item)
        self._overlays.append(overlay)

    def delete(self, item):
        for overlay in self._overlays.copy():
            if overlay.config is item:
                self._overlays.remove(overlay)

    @property
    def items(self):
        yield from (overlay.config for overlay in self._overlays)

    @property
    def overlays(self):
        return self._overlays

    @property
    def most_recent(self):
        return self._overlays[-1]

    def remove_overlay(self, overlay):
        self._overlays.remove(overlay)
        self.save_to_disk()
