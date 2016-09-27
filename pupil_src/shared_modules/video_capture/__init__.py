'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

""" General documentation:

Video capture provides two basic functionalities:
- A source, which provides a stream of image frames
- A manager, which enumerates all available sources

Each source type is usually paired with a matching manager. There are currently
four source types:

- `UVC_Source` (Local USB sources)
- `NDSI_Source` (Remote Pupil Mobile sources)
- `Fake_Source` (Fallback, static random image)
- `File_Source` (For debugging, loads video from file)

See `manager.__init__.py` for more information on managers.
See `source.__init__.py` for more information on sources.
"""

import source, manager

source_classes  = [source.Fake_Source, source.UVC_Source, source.NDSI_Source, source.File_Source]
manager_classes = [manager.Fake_Manager, manager.UVC_Manager, manager.NDSI_Manager, manager.File_Manager]