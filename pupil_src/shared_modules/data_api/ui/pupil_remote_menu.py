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
import socket

from observable import Observable
from pyglui import ui

from data_api.controller import PupilRemoteController


logger = logging.getLogger(__name__)


class PupilRemoteMenu(Observable):
    menu_label = "Pupil Remote"

    def on_change_custom_address(self, host: str, port: str):
        # self.__start_server(new_address)
        self._update_menu()

    def __init__(self, pupil_remote_controller: PupilRemoteController):
        self.__sub_menu = None
        self.__pupil_remote_controller = pupil_remote_controller

    def append_to_menu(self, menu):
        if self.__sub_menu is None:
            self.__sub_menu = ui.Growing_Menu(self.menu_label)
        menu.append(self.__sub_menu)
        self._update_menu()

    @property
    def _use_primary_interface(self) -> bool:
        return self.__pupil_remote_controller.use_primary_interface

    @_use_primary_interface.setter
    def _use_primary_interface(self, value: bool):
        self.__pupil_remote_controller.use_primary_interface = value
        self._update_menu()

    @property
    def _port(self) -> str:
        return self.__pupil_remote_controller.port

    @_port.setter
    def _port(self, value: int):
        self.__pupil_remote_controller.port = int(value)
        self._update_menu()

    # @property
    # def _local_address(self) -> str:
    #     return f"127.0.0.1:{self._port}"

    # @property
    # def _remote_address(self) -> str:
    #     try:
    #         external_ip = socket.gethostbyname(socket.gethostname())
    #     except Exception:
    #         external_ip = "Your external ip"
    #     return f"{external_ip}:{self._port}"

    # @property
    # def _custom_address(self) -> str:
    #     return f"{self._custom_host}:{self._port}"

    # @_custom_address.setter
    # def _custom_address(self, value: str):
    #     if value.count(":") != 1:
    #         logger.error("address format not correct")
    #         return
    #     self._custom_host, self.__port = value.split(":")
    #     self.on_change_custom_address(host=self._custom_host, port=self._port)

    def _update_menu(self):
        self.__remove_menu_items()
        self.__insert_menu_items()

    def __remove_menu_items(self):
        if self.__sub_menu is None:
            return
        del self.__sub_menu.elements[:]

    def __insert_menu_items(self):
        if self.__sub_menu is None:
            return

        self.__sub_menu.append(
            ui.Info_Text("Pupil Remote using ZeroMQ REQ REP scheme.")
        )
        self.__sub_menu.append(
            ui.Switch("_use_primary_interface", self, label="Use primary network interface")
        )

        if self._use_primary_interface:
            self.__sub_menu.append(
                ui.Text_Input("_port", self, label="Port")
            )
            self.__sub_menu.append(
                ui.Info_Text(f'Connect locally:   "tcp://{self._local_address}"')
            )
            self.__sub_menu.append(
                ui.Info_Text(f'Connect remotely: "tcp://{self._remote_address}"')
            )
        else:
            self.__sub_menu.append(
                ui.Text_Input("_custom_address", self, label="Address")
            )
            self.__sub_menu.append(
                ui.Info_Text(f'Bound to: "tcp://{self._custom_address}"')
            )
