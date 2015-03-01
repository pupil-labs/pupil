'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import platform
import sys, os
from git_version import write_version_file,dpkg_deb_version
import shutil
from subprocess import call

if platform.system() == 'Darwin':
    print "starting version stript:"
    write_version_file.main('dist/Pupil Capture.app/Contents/MacOS')
    print "created version file in dist folder"

    shutil.rmtree('dist/Pupil Capture')
    print 'removed the non-app dist bundle'

elif platform.system() == 'Linux':

    distribtution_dir = 'dist'
    pupil_capture_dir =  os.path.join(distribtution_dir, 'pupil_capture')

    print "starting version stript:"
    write_version_file(pupil_capture_dir)
    print "created version file in dist folder"

    old_deb_dir = [d for d in os.listdir('.') if  d.startswith('pupil_capture_')]
    for d in old_deb_dir:
        try:
            shutil.rmtree(d)
            print 'removed deb structure dir: "%s"'%d
        except:
            pass

    #lets build the structure for our deb package.
    deb_root = 'pupil_capture_%s'%dpkg_deb_version()
    DEBIAN_dir = os.path.join(deb_root,'DEBIAN')
    opt_dir = os.path.join(deb_root,'opt')
    bin_dir = os.path.join(deb_root,'usr','bin')
    app_dir = os.path.join(deb_root,'usr','share','applications')
    ico_dir = os.path.join(deb_root,'usr','share','icons','hicolor','scalable','apps')
    os.makedirs(DEBIAN_dir)
    os.makedirs(bin_dir)
    os.makedirs(app_dir)
    os.makedirs(ico_dir)

    #DEBAIN Package description
    with open(os.path.join(DEBIAN_dir,'control'),'w') as f:
        dist_size = sum(os.path.getsize(os.path.join(pupil_capture_dir,f)) for f in os.listdir(pupil_capture_dir) if os.path.isfile(os.path.join(pupil_capture_dir,f)))
        content = '''\
Package: pupil-capture
Version: %s
Architecture: amd64
Maintainer: Pupil Labs <info@pupil-labs.com>
Section: applications
Priority: optional
Description: Pupil Capture is part of the Pupil Eye Tracking Platform
Installed-Size: %s
'''%(dpkg_deb_version(),dist_size/1024)
        f.write(content)
    os.chmod(os.path.join(DEBIAN_dir,'control'),0644)

    #bin_starter script
    with open(os.path.join(bin_dir,'pupil_capture'),'w') as f:
        content = '''\
#!/bin/sh
exec /opt/pupil_capture/pupil_capture "$@"'''
        f.write(content) 
    os.chmod(os.path.join(bin_dir,'pupil_capture'),0755)


    #.desktop entry
    with open(os.path.join(app_dir,'pupil_capture.desktop'),'w') as f:
        content = '''\
[Desktop Entry]
Version=1.0
Type=Application
Name=Pupil Capture
Comment=Eye Tracking Capture Program
Exec=/opt/pupil_capture/pupil_capture
Terminal=false
Icon=pupil-capture
Categories=Application;
StartupNotify=true
Name[en_US]=Pupil Capture
Actions=Monocular;Binocular;

[Desktop Action Monocular]
Name= Monocular Mode
Exec=/opt/pupil_capture/pupil_capture

[Desktop Action Binocular]
Name= Binocular Mode
Exec=/opt/pupil_capture/pupil_capture binocular'''
        f.write(content) 
    os.chmod(os.path.join(app_dir,'pupil_capture.desktop'),0644)

    #copy icon:
    shutil.copy('pupil-capture.svg',ico_dir)
    os.chmod(os.path.join(ico_dir,'pupil-capture.svg'),0644)

    #copy the actual application
    shutil.copytree(distribtution_dir,opt_dir)
    

    #run dpkg_deb
    call('dpkg-deb --build %s'%deb_root,shell=True)

    print 'DONE!'