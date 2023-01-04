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
import logging

from gaze_producer import ui as plugin_ui
from pyglui import ui

logger = logging.getLogger(__name__)


class StorageEditMenu(plugin_ui.SelectAndRefreshMenu, abc.ABC):
    """
    A SelectAndRefreshMenu that shows the items in a storage. It has a button above the
    selector to create a new item and shows for every item two buttons,
    one to duplicate the current item, and one to delete it.
    """

    new_button_label = "New"
    duplicate_button_label = "Duplicate Current Configuration"
    delete_button_label = "Delete"

    def __init__(self, storage):
        super().__init__()
        self._storage = storage

    @abc.abstractmethod
    def _new_item(self):
        pass

    @abc.abstractmethod
    def _duplicate_item(self, item):
        pass

    @abc.abstractmethod
    def _render_custom_ui(self, item, menu):
        pass

    @abc.abstractmethod
    def _item_label(self, item):
        pass

    @property
    def items(self):
        # storages are just iterable, but we need things like len() and
        # access by index, so we return a list
        return [item for item in self._storage]

    @property
    def item_labels(self):
        return [self._item_label(item) for item in self._storage]

    def render_above_selector_elements(self, menu):
        menu.append(self._create_new_button())
        if self.items:
            menu.append(self._create_duplicate_button())

    def _create_new_button(self):
        return ui.Button(
            label=self.new_button_label, function=self._on_click_new_button
        )

    def _create_duplicate_button(self):
        return ui.Button(
            label=self.duplicate_button_label, function=self._on_click_duplicate_button
        )

    def _create_delete_button(self):
        return ui.Button(label=self.delete_button_label, function=self._on_click_delete)

    def render_item(self, item, menu):
        self._render_custom_ui(item, menu)
        menu.append(self._create_delete_button())

    def _on_click_new_button(self):
        new_item = self._new_item()
        self._storage.add(new_item)
        if new_item not in self._storage:
            logger.error("New item could not be added. Aborting.")
            return
        self.current_item = new_item
        self.render()

    def _on_click_duplicate_button(self):
        new_item = self._duplicate_item(self.current_item)
        self._storage.add(new_item)
        if new_item not in self._storage:
            logger.error("New item could not be added. Aborting.")
            return
        self.current_item = new_item
        self.render()

    def _on_click_delete(self):
        if self.current_item is None:
            return
        current_index = self.items.index(self.current_item)
        self._storage.delete(self.current_item)
        current_index = min(current_index, len(self.items) - 1)
        if current_index != -1:
            self.current_item = self.items[current_index]
        else:
            self.current_item = None
        self.render()
