'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import source, manager

source_classes  = [source.Fake_Source, source.UVC_Source, source.NDSI_Source, source.File_Source]
manager_classes = [manager.Fake_Manager, manager.UVC_Manager, manager.NDSI_Manager, manager.File_Manager]