"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc

from pyglui import ui


class SelectAndRefreshMenu(abc.ABC):
    """
    Holds a GrowingMenu that contains an item selector and below item specific
    elements. When the currently selected item changes, the specific elements are
    rendered again, s.t. they always relate to the current item.
    You can also render elements above the selector that do not change when the
    current item changes.
    """

    selector_label = "Current:"

    def __init__(self):
        self.menu = ui.Growing_Menu(self.menu_label)
        self.current_item = None
        # when the current element changes, a few menu elements remain (=the selector
        # and things above) and the rest gets deleted and rendered again (=item
        # specific elements)
        self._number_of_static_menu_elements = 0

    @property
    @abc.abstractmethod
    def menu_label(self):
        pass

    @property
    @abc.abstractmethod
    def items(self):
        pass

    @property
    @abc.abstractmethod
    def item_labels(self):
        pass

    @abc.abstractmethod
    def render_above_selector_elements(self, menu):
        pass

    @abc.abstractmethod
    def render_item(self, item, menu):
        pass

    def render(self):
        if not self.current_item and len(self.items) > 0:
            self.current_item = self.items[0]
        self.menu.elements.clear()
        self.render_above_selector_elements(self.menu)
        if len(self.items) > 0:
            self._render_item_selector_and_current_item()

    def _render_item_selector_and_current_item(self):
        self.menu.append(
            ui.Selector(
                "current_item",
                self,
                setter=self._on_change_current_item,
                selection=self.items,
                labels=self.item_labels,
                label=self.selector_label,
            )
        )
        self.menu.append(ui.Separator())
        self._number_of_static_menu_elements = len(self.menu.elements)
        # apparently, the 'setter' function is only triggered if the selection
        # changes, but not for the initial selection, so we call it manually
        if self.current_item:
            self._on_change_current_item(self.current_item)

    # TODO: implement this with an attribute observer when the feature is available
    def _on_change_current_item(self, item):
        self.current_item = item
        del self.menu.elements[self._number_of_static_menu_elements :]
        temp_menu = ui.Growing_Menu("Temporary")
        self.render_item(item, temp_menu)
        self.menu.elements.extend(temp_menu.elements)
