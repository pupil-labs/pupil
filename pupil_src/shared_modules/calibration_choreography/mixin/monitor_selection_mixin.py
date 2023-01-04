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
import typing as T

from calibration_choreography.controller import GUIMonitor

logger = logging.getLogger(__name__)


class MonitorSelectionMixin:
    @staticmethod
    def currently_connected_monitor_names() -> T.List[str]:
        return list(GUIMonitor.currently_connected_monitors_by_name().keys())

    @property
    def selected_monitor_name(self) -> str:
        try:
            monitor_name = self.__selected_monitor_name
        except AttributeError:
            monitor_name = GUIMonitor.primary_monitor().name

        self.selected_monitor_name = monitor_name
        return self.__selected_monitor_name

    @selected_monitor_name.setter
    def selected_monitor_name(self, monitor_name: str):
        if (
            monitor_name is None
            or GUIMonitor.find_monitor_by_name(monitor_name) is None
        ):
            primary_name = GUIMonitor.primary_monitor().name
            if monitor_name is not None:
                logger.info(
                    f'Monitor "{monitor_name}" no longer available. Using "{primary_name}" instead.'
                )
            monitor_name = primary_name
        self.__selected_monitor_name = monitor_name
