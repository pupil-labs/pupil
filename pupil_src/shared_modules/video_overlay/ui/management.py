from pyglui import ui

from video_overlay.ui.interactions import Draggable
from video_overlay.ui.menu import GenericOverlayMenu


class UIManagement:
    def __init__(self, plugin, parent_menu, existing_overlays):
        self._parent_menu = parent_menu
        self._draggables = []

        self._add_menu_with_general_elements()
        self._add_menu_for_existing_overlays(existing_overlays)
        self._add_draggable_for_existing_overlays(existing_overlays)

        plugin.add_observer("on_click", self._on_click)
        plugin.add_observer("on_pos", self._on_pos)
        plugin.add_observer("_add_overlay_to_storage", self._add_overlay_menu)
        plugin.add_observer("_add_overlay_to_storage", self._add_overlay_draggable)

    def _add_menu_with_general_elements(self):
        self._parent_menu.append(
            ui.Info_Text(
                "This plugin is able to overlay videos with synchronized timestamps."
            )
        )
        self._parent_menu.append(
            ui.Info_Text(
                "Drag and drop such videos onto the "
                "main Player window in order to load them"
            )
        )
        self._parent_menu.append(ui.Separator())

    def _add_menu_for_existing_overlays(self, existing_overlays):
        for overlay in existing_overlays:
            self._add_overlay_menu(overlay)

    def _add_overlay_menu(self, overlay):
        self._parent_menu.append(GenericOverlayMenu(overlay))

    def _add_draggable_for_existing_overlays(self, existing_overlays):
        for overlay in existing_overlays:
            self._add_overlay_draggable(overlay)

    def _add_overlay_draggable(self, overlay):
        draggable = Draggable(overlay)
        self._draggables.append(draggable)

    def teardown(self):
        del self._parent_menu[:]
        del self._draggables[:]
        del self._parent_menu

    def _on_click(self, *args, **kwargs):
        pass

    def _on_pos(self, *args, **kwargs):
        pass
