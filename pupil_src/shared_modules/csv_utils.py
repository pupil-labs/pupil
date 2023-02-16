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
import csv
import os
import typing as t

CSV_EXPORT_RAW_TYPE = t.TypeVar("CSV_EXPORT_RAW_TYPE")
CSV_EXPORT_LABEL_TYPE = t.AnyStr
CSV_EXPORT_VALUE_TYPE = t.Any
CSV_EXPORT_VALUE_GETTER_TYPE = t.Callable[[CSV_EXPORT_RAW_TYPE], CSV_EXPORT_VALUE_TYPE]
CSV_EXPORT_SCHEMA_TYPE = t.List[
    t.Tuple[CSV_EXPORT_LABEL_TYPE, CSV_EXPORT_VALUE_GETTER_TYPE]
]


class CSV_Exporter(abc.ABC, t.Generic[CSV_EXPORT_RAW_TYPE]):
    @classmethod
    @abc.abstractmethod
    def csv_export_schema(cls) -> CSV_EXPORT_SCHEMA_TYPE:
        return []

    @classmethod
    def csv_export_labels(cls) -> t.Iterable[CSV_EXPORT_LABEL_TYPE]:
        return tuple(label for label, _ in cls.csv_export_schema())

    @classmethod
    def csv_export_values(
        cls, raw_value: CSV_EXPORT_RAW_TYPE
    ) -> t.Iterable[CSV_EXPORT_VALUE_TYPE]:
        return tuple(getter(raw_value) for _, getter in cls.csv_export_schema())

    def csv_export(
        self,
        raw_values: t.Iterable[CSV_EXPORT_RAW_TYPE],
        export_dir: str,
        export_name: str,
    ) -> str:
        export_path = os.path.abspath(os.path.join(export_dir, export_name))

        with open(export_path, "w", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(type(self).csv_export_labels())
            for raw_value in raw_values:
                csv_writer.writerow(type(self).csv_export_values(raw_value))

        return export_path


def read_key_value_file(csvfile):
    """Reads CSV file, parses content into dict

    Args:
        csvfile (FILE): Readable file

    Returns:
        DICT: Dictionary containing file content
    """
    kvstore = {}  # init key value store
    first_line = csvfile.readline()
    if "key" not in first_line or "value" not in first_line:
        csvfile.seek(0)  # Seek to start if first_line is not an header
    dialect = csv.Sniffer().sniff(first_line, delimiters=",\t")
    reader = csv.reader(
        csvfile, dialect, quoting=csv.QUOTE_NONE, escapechar="\\"
    )  # create reader
    for row in reader:
        kvstore[row[0]] = row[1]
    return kvstore


def write_key_value_file(csvfile, dictionary, append=False):
    """Writes a dictionary to a writable file in a CSV format

    Args:
        csvfile (FILE): Writable file
        dictionary (dict): Dictionary containing key-value pairs
        append (bool, optional): Writes `key,value` as fieldnames if False

    Returns:
        None: No return
    """
    writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_NONE, escapechar="\\")
    if not append:
        writer.writerow(["key", "value"])
    for key, val in dictionary.items():
        writer.writerow([key, val])
