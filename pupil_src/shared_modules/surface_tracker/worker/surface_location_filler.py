import abc
import typing as T
import weakref

import observable

from surface_tracker.surface import Surface, Surface_Location
from surface_tracker import background_tasks
from surface_tracker import offline_utils
from .worker_utils import mp_context


class Surface_Location_Filler(observable.Observable):

    ON_DID_CALCULATE_SURFACE_LOCATION_CALLBACK = T.Callable[[Surface, int, Surface_Location], None]
    ON_DID_CANCEL = T.Callable[[], None]
    ON_DID_COMPLETE = T.Callable[[], None]

    def __init__(
        self,
        on_did_calculate_surface_location: ON_DID_CALCULATE_SURFACE_LOCATION_CALLBACK = None,
        on_did_cancel: ON_DID_CANCEL = None,
        on_did_complete: ON_DID_COMPLETE = None,
    ):
        self.__reset_to_idle_state()

        if on_did_calculate_surface_location is not None:
            self.add_observer(
                "on_did_calculate_surface_location",
                on_did_calculate_surface_location,
            )
        if on_did_cancel is not None:
            self.add_observer(
                "on_did_cancel",
                on_did_cancel,
            )
        if on_did_complete is not None:
            self.add_observer(
                "on_did_complete",
                on_did_complete,
            )

    def on_did_calculate_surface_location(self, surface: Surface, cache_idx: int, location: Surface_Location):
        pass

    def on_did_cancel(self):
        pass

    def on_did_complete(self):
        pass

    def start(self, surfaces, cache_seek_idx, marker_cache, camera_model):
        self.__surfaces = surfaces

        self.__background_task = background_tasks.background_data_processor( #TODO: Use 1 task for all surfaces
            marker_cache,
            offline_utils.surfaces_locator_callable(
                self.__surfaces,
                camera_model,
            ),
            cache_seek_idx,
            mp_context,
        )

    def fetch(self):
        if self.__background_task is None:
            return

        for cache_idx, enumerated_locations in self.__background_task.fetch():
            for surface_idx, location in enumerated_locations:
                self.on_did_calculate_surface_location(
                    surface=self.__surfaces[surface_idx],
                    cache_idx=cache_idx,
                    location=location,
                )

        if self.__background_task.completed:
            self.__reset_to_idle_state()
            self.on_did_complete()

    def cancel(self):
        if self.__background_task is not None:
            self.__background_task.cancel()
        self.__reset_to_idle_state()
        self.on_did_cancel()

    def __reset_to_idle_state(self):
        self.__background_task = None
        self.__surfaces = []
