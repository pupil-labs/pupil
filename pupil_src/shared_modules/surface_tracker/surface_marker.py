import typing


__all__ = [
    "Surface_Marker_UID",
    "Surface_Marker_TagID",
]


Surface_Marker_UID = typing.NewType("Surface_Marker_UID", str)


Surface_Marker_TagID = typing.NewType("Surface_Marker_TagID", int)
