'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import platform

if platform.system() == 'Darwin':
    import shutil
    import write_version_file
    print "starting version stript:"
    write_version_file.main('dist/Pupil Capture.app/Contents/MacOS')
    print "created version file in dist folder"

    shutil.rmtree('dist/Pupil Capture')
    print 'removed the non-app dist bundle'

elif platform.system() == 'Linux':
    import sys,os
    import write_version_file
    import shutil

    distribtution_dir = 'dist'
    pupil_capture_dir =  os.path.join(distribtution_dir, 'pupil_capture')

    shutil.copy('make_shortcut.sh',os.path.join(distribtution_dir,'make_shortcut.sh'))
    print "Copied a small script that creates a shortcut for the user into distribtution_dir"
    os.chmod(os.path.join(distribtution_dir,'make_shortcut.sh'),0775)
    print "Gave that file excetion rights"

    print "starting version stript:"
    write_version_file.main(pupil_capture_dir)
    print "created version file in dist folder"



