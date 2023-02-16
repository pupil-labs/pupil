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
import functools
import logging
import os
import typing

import file_methods

from .surface import Surface
from .surface_serializer import (
    InvalidSurfaceDefinition,
    _Surface_Serializer_Base,
    _Surface_Serializer_V00,
    _Surface_Serializer_V01,
)

logger = logging.getLogger(__name__)


class _Surface_File_Store_Base(abc.ABC):
    # Abstract members

    @property
    @abc.abstractmethod
    def file_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def serializer(self) -> _Surface_Serializer_Base:
        pass

    # Public API

    def __init__(self, parent_dir, _persistent_dict_class=file_methods.Persistent_Dict):
        if not os.path.exists(parent_dir):
            raise FileNotFoundError(f"No such file or directory: {parent_dir}")
        if not os.path.isdir(parent_dir):
            raise NotADirectoryError(f"Not a directory: '{parent_dir}'")
        self._parent_dir = parent_dir
        self._persistent_dict_class = _persistent_dict_class

    @property
    def parent_dir(self):
        return self._parent_dir

    @property
    def file_path(self) -> str:
        return os.path.join(self.parent_dir, self.file_name)

    def read_surfaces_from_file(self, surface_class) -> typing.Iterator[Surface]:
        return self._read_surfaces_from_file_path(
            file_path=self.file_path,
            serializer=self.serializer,
            surface_class=surface_class,
            should_skip_on_invalid=False,
        )

    def write_surfaces_to_file(self, surfaces: typing.Iterator[Surface]):
        dict_from_surface = self.serializer.dict_from_surface
        serialized_surfaces = [
            dict_from_surface(surface) for surface in surfaces if surface.defined
        ]
        surface_definitions = self._persistent_dict_class(self.file_path)
        surface_definitions["surfaces"] = serialized_surfaces
        surface_definitions.save()

    # Protected API

    def _read_surfaces_from_file_path(
        self,
        file_path: str,
        serializer: _Surface_Serializer_Base,
        surface_class,
        should_skip_on_invalid: bool = False,
    ) -> typing.Iterator[Surface]:
        # TODO: Assert surface_class is a class, and is a Surface subclass

        def surface_from_dict(surface_dict: dict) -> typing.Optional[Surface]:
            try:
                return serializer.surface_from_dict(
                    surface_definition=surface_dict, surface_class=surface_class
                )
            except InvalidSurfaceDefinition:
                if should_skip_on_invalid:
                    return None
                else:
                    raise

        if not os.path.isfile(file_path):
            return []

        surface_definitions = self._persistent_dict_class(file_path)
        surface_definitions = surface_definitions.get("surfaces", [])

        surfaces = map(surface_from_dict, surface_definitions)
        surfaces = filter(lambda s: s is not None, surfaces)
        return surfaces


class _Surface_File_Store_V00(_Surface_File_Store_Base):
    # _Surface_File_Store_Base API

    @property
    def file_name(self) -> str:
        return "surface_definitions"

    @property
    def serializer(self) -> _Surface_Serializer_Base:
        return _Surface_Serializer_V00()


class _Surface_File_Store_V01(_Surface_File_Store_Base):
    # _Surface_File_Store_Base API

    @property
    def file_name(self) -> str:
        return "surface_definitions_v01"

    @property
    def serializer(self) -> _Surface_Serializer_Base:
        return _Surface_Serializer_V01()

    def read_surfaces_from_file(self, surface_class) -> typing.Iterator[Surface]:
        if os.path.isfile(self.file_path):
            # If the updated file exists, then read from it and ignore the legacy steps
            return super().read_surfaces_from_file(surface_class=surface_class)

        # If the updated file doesn't exist,
        # look for definitions with the new format (seriliazer),
        # but saved at the old file path.
        return self._read_surfaces_from_file_path(
            file_path=self.__legacy_file_path,
            serializer=self.serializer,
            surface_class=surface_class,
            should_skip_on_invalid=True,  # Since this file might contain older surface definitions, those can be skipped
        )

    # Private

    @property
    def __legacy_file_path(self) -> str:
        return os.path.join(self.parent_dir, "surface_definitions")


class Surface_File_Store(_Surface_File_Store_Base):
    Version = int
    Migration_Procedure = typing.Callable[[], None]
    Versioned_File_Store_Mapping = typing.Mapping[Version, _Surface_File_Store_Base]

    # _Surface_File_Store_Base API

    def __init__(self, parent_dir, **kwargs):
        super().__init__(parent_dir=parent_dir, **kwargs)
        self.__versioned_file_stores: Surface_File_Store.Versioned_File_Store_Mapping = {
            0: _Surface_File_Store_V00(parent_dir=parent_dir),
            1: _Surface_File_Store_V01(parent_dir=parent_dir),
            # Add any new file store versions here...
        }

        # Pre-computed properties
        self.__supported_versions = tuple(sorted(self.__versioned_file_stores.keys()))
        self.__migration_step_sequence = tuple(
            zip(self.__supported_versions, self.__supported_versions[1:])
        )

    @property
    def file_name(self) -> str:
        return self.__file_store_latest.file_name

    @property
    def file_path(self) -> str:
        return self.__file_store_latest.file_path

    @property
    def serializer(self) -> _Surface_Serializer_Base:
        return self.__file_store_latest.serializer

    def read_surfaces_from_file(self, surface_class) -> typing.Iterator[Surface]:
        # Perform all migrations
        for source_version, target_version in self.__migration_step_sequence:
            migration_proc = self.__migration_procedure(
                surface_class=surface_class,
                source_version=source_version,
                target_version=target_version,
            )
            migration_proc()

        return self.__file_store_latest.read_surfaces_from_file(
            surface_class=surface_class
        )

    def write_surfaces_to_file(self, surfaces: typing.Iterator[Surface]):
        self.__file_store_latest.write_surfaces_to_file(surfaces=surfaces)

    # Private API

    @property
    def __file_store_latest(self) -> _Surface_File_Store_Base:
        latest_version = self.__supported_versions[-1]
        return self.__versioned_file_stores[latest_version]

    def __migration_procedure(
        self, surface_class, source_version: Version, target_version: Version
    ) -> Migration_Procedure:
        # Handle any special-case migrations here
        if (source_version, target_version) == (0, 1):
            return functools.partial(
                self.__migration_v00_v01,
                surface_class=surface_class,
                source_version=source_version,
                target_version=target_version,
            )

        # Otherwise, fallback on the rewrite migration
        return functools.partial(
            self.__simple_rewrite_migration,
            surface_class=surface_class,
            source_version=source_version,
            target_version=target_version,
        )

    def __simple_rewrite_migration(
        self, surface_class, source_version: Version, target_version: Version
    ):
        source_file_store = self.__versioned_file_stores[source_version]
        target_file_store = self.__versioned_file_stores[target_version]

        if os.path.isfile(target_file_store.file_path):
            # If the target file already exists, there is nothing more to do
            return

        # Otherwise, write the surfaces to the new location with the new format
        surfaces = source_file_store.read_surfaces_from_file(
            surface_class=surface_class
        )
        target_file_store.write_surfaces_to_file(surfaces=surfaces)

    def __migration_v00_v01(
        self, surface_class, source_version: Version, target_version: Version
    ):
        assert source_version == 0
        assert target_version == 1

        # Try the simple migration from 0 to 1.
        # If it fails with InvalidSurfaceDefinition, it means the file that v00 was trying to read doesn't contain v00 definitions, but v01 definitions.
        # Try to migrate from 1 to 1; this will trigger reading from the legacy file and writing to the updated file.

        try:
            self.__simple_rewrite_migration(
                surface_class=surface_class,
                source_version=source_version,
                target_version=target_version,
            )
        except InvalidSurfaceDefinition:
            self.__simple_rewrite_migration(
                surface_class=surface_class,
                source_version=target_version,  # NOTE: target_version, not source_version
                target_version=target_version,
            )
