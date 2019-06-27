"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as t
import eye_movement.utils as utils
import eye_movement.model as model


class Eye_Movement_Seek_Controller:
    def __init__(
        self,
        plugin,
        storage: model.Classified_Segment_Storage,
        seek_to_timestamp: t.Callable[[float], None],
    ):
        self.storage = storage
        self._seek_to_timestamp = seek_to_timestamp
        self._current_segment_index = 0

    def __getitem__(self, index: int) -> t.Optional[model.Classified_Segment]:
        if 0 <= index < self.total_segment_count:
            return self.storage[index]
        else:
            return None

    @property
    def current_segment(self) -> t.Optional[model.Classified_Segment]:
        return self[self.current_segment_index]

    @property
    def prev_segment(self) -> t.Optional[model.Classified_Segment]:
        return self[self.current_segment_index - 1]

    @property
    def next_segment(self) -> t.Optional[model.Classified_Segment]:
        return self[self.current_segment_index + 1]

    @property
    def current_segment_index(self) -> int:
        return self._current_segment_index if self._current_segment_index else 0

    @property
    def total_segment_count(self) -> int:
        return len(self.storage)

    def seek_to_timestamp(self, timestamp: float):
        self._seek_to_timestamp(timestamp)

    def jump_to_next_segment(self):
        return self._jump_to_segment_offset(+1)

    def jump_to_prev_segment(self):
        return self._jump_to_segment_offset(-1)

    def _jump_to_segment_offset(self, offset: int):

        if offset == 0:
            utils.logger.warning("No eye movement seek offset")
            return

        if self.total_segment_count < 1:
            utils.logger.warning("No eye movement segments available")
            return

        offset_index = self.current_segment_index
        offset_index = offset_index if offset_index else 0

        offset_index += offset
        offset_index %= self.total_segment_count

        offset_segment = self.storage[offset_index]
        offset_timestamp = offset_segment.start_frame_timestamp

        self._current_segment_index = offset_index
        self.seek_to_timestamp(offset_timestamp)

    def update_visible_segments(
        self, visible_segments: t.Iterable[model.Classified_Segment]
    ) -> t.Optional[model.Classified_Segment]:

        if not self.total_segment_count:
            self._current_segment_index = None
            return None

        current_segment = None
        visible_segments = visible_segments if visible_segments else []
        current_segment_index = self.current_segment_index

        if current_segment_index:
            current_segment_index = current_segment_index % self.total_segment_count
            current_segment = self.storage[current_segment_index]

        if not visible_segments:
            self._current_segment_index = current_segment_index
            return current_segment

        if (current_segment not in visible_segments) and len(visible_segments) > 0:
            current_segment = visible_segments[0]
            current_segment_index = self.storage.index(current_segment)

        self._current_segment_index = current_segment_index
        return current_segment
