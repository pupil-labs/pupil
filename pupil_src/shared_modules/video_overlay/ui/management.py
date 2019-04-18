from collections import OrderedDict
from pyglui import ui

from observable import Observable

from video_overlay.ui.interactions import Draggable
from video_overlay.ui.menu import GenericOverlayMenuRenderer


class UIManagement(Observable):
    def __init__(self, plugin, parent_menu, existing_overlays):
        self._parent_menu = parent_menu
        self._menu_renderers = {}
        # Insrt order is important for correct dragging behavior
        self._draggables = OrderedDict()

        self._add_menu_with_general_elements()
        self._add_menu_for_existing_overlays(existing_overlays)
        self._add_draggable_for_existing_overlays(existing_overlays)

        plugin.add_observer("on_click", self._on_click)
        plugin.add_observer("on_pos", self._on_pos)
        plugin.add_observer("_overlay_added_to_storage", self._add_overlay_menu)
        plugin.add_observer("_overlay_added_to_storage", self._add_overlay_draggable)

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
        renderer = GenericOverlayMenuRenderer(overlay)
        renderer.add_observer("remove_button_clicked", self.remove_overlay)
        self._menu_renderers[overlay] = renderer
        self._parent_menu.append(renderer.menu)

    def _add_draggable_for_existing_overlays(self, existing_overlays):
        for overlay in existing_overlays:
            self._add_overlay_draggable(overlay)

    def _add_overlay_draggable(self, overlay):
        draggable = Draggable(overlay)
        self._draggables[overlay] = draggable

    def remove_overlay(self, overlay):
        renderer = self._menu_renderers[overlay]
        renderer.remove_all_observers("remove_button_clicked")
        del self._menu_renderers[overlay]
        del self._draggables[overlay]
        self._parent_menu.remove(renderer.menu)

    def teardown(self):
        del self._parent_menu[:]
        del self._parent_menu
        self._draggables.clear()
        self._menu_renderers.clear()

    def _on_click(self, *args, **kwargs):
        # iterate over draggables in reverse order since last element is drawn on top
        any(d.on_click(*args, **kwargs) for d in reversed(self._draggables.values()))

    def _on_pos(self, *args, **kwargs):
        any(d.on_pos(*args, **kwargs) for d in reversed(self._draggables.values()))
