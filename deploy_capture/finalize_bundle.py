import platform

if platform.system() == 'Darwin':
    import shutil
    import write_version_file
    print "starting version stript:"
    write_version_file.main('dist/Pupil Capture.app/Contents/MacOS')
    print "created version file in dist folder"

    print "copy starter app"
    shutil.copytree('run_pupil_capture_from_mac_terminal.app', 'dist/run_pupil_capture_from_mac_terminal.app')

elif platform.system() == 'Linux':
    import sys,os
    import write_version_file
    import shutil

    distribtution_dir = 'dist'
    pupil_capture_dir =  os.path.join(distribtution_dir, 'pupil_capture')


    shutil.copy('patch_uvc_driver.sh',os.path.join(distribtution_dir,'patch_uvc_driver.sh'))
    print "Copied a small script to patch uvc driver into the distribution dir"
    os.chmod(os.path.join(distribtution_dir,'patch_uvc_driver.sh'),0775)
    print "Gave that file excetion rights"

    shutil.copy('make_shortcut.sh',os.path.join(distribtution_dir,'make_shortcut.sh'))
    print "Copied a small script that creates a shortcut for the user into distribtution_dir"
    os.chmod(os.path.join(distribtution_dir,'make_shortcut.sh'),0775)
    print "Gave that file excetion rights"

    print "starting version stript:"
    write_version_file.main(pupil_capture_dir)
    print "created version file in dist folder"



