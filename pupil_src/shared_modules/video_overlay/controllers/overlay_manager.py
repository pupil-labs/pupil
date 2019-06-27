import abc
from storage import SingleFileStorage
from video_overlay.models.config import Configuration
from video_overlay.workers.overlay_renderer import OverlayRenderer


class OverlayManager(SingleFileStorage):
    def __init__(self, rec_dir, plugin):
        super().__init__(rec_dir, plugin)
        self._overlays = []
        self._load_from_disk()
        self._patch_on_cleanup(plugin)

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

    def _patch_on_cleanup(self, plugin):
        """Patches cleanup observer to trigger on get_init_dict().

        Save current settings to disk on get_init_dict() instead of cleanup().
        This ensures that the World Video Exporter loads the most recent settings.
        """
        plugin.remove_observer("cleanup", self._on_cleanup)
        plugin.add_observer("get_init_dict", self._on_cleanup)
