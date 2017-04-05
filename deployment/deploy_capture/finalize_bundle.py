'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import platform
import sys, os
from version import write_version_file,dpkg_deb_version
import shutil
from subprocess import call

if platform.system() == 'Darwin':
    print( "starting version stript:")
    write_version_file('dist/Pupil Capture.app/Contents/MacOS')
    print( "created version file in dist folder")

    shutil.rmtree('dist/Pupil Capture')
    print( 'removed the non-app dist bundle')

    bundle_name = 'pupil_capture_mac_os_x64_v%s'%dpkg_deb_version()
    bundle_dmg_name = 'Install Pupil Capture'
    src_dir = 'dist'
    bundle_app_dir = os.path.join(src_dir,'Pupil Capture.app/' )
    print( "Codesigning now")
    if call("codesign --force --verify --verbose -s 'Developer ID Application: Pupil Labs UG (haftungsbeschrankt) (R55K9ESN6B)' --deep '%s'"%bundle_app_dir,shell=True) != 0:
        print( Exception("Codesinging  failed"))
    # if call("spctl --assess --type execute '%s'"%bundle_app_dir,shell=True) != 0:
        # print Exception("Codesing verification  failed")
    call("ln -s /Applications/ %s/Applications"%src_dir,shell=True)
    call("hdiutil create  -volname '%s' -srcfolder %s -size 240m -format UDZO '%s.dmg'"%(bundle_dmg_name,src_dir,bundle_name),shell=True)

elif platform.system() == 'Windows':
    write_version_file(os.path.join('dist', 'Pupil Capture'))

elif platform.system() == 'Linux':

    distribtution_dir = 'dist'
    pupil_capture_dir =  os.path.join(distribtution_dir, 'pupil_capture')

    print( "starting version stript:")
    write_version_file(pupil_capture_dir)
    print( "created version file in dist folder")

    old_deb_dir = [d for d in os.listdir('.') if  d.startswith('pupil_capture_')]
    for d in old_deb_dir:
        try:
            shutil.rmtree(d)
            print( 'removed deb structure dir: "%s"'%d)
        except:
            pass

    #lets build the structure for our deb package.
    deb_root = 'pupil_capture_linux_os_x64_v%s'%dpkg_deb_version()
    DEBIAN_dir = os.path.join(deb_root,'DEBIAN')
    opt_dir = os.path.join(deb_root,'opt')
    bin_dir = os.path.join(deb_root,'usr','bin')
    app_dir = os.path.join(deb_root,'usr','share','applications')
    ico_dir = os.path.join(deb_root,'usr','share','icons','hicolor','scalable','apps')
    os.makedirs(DEBIAN_dir,0o755)
    os.makedirs(bin_dir,0o755)
    os.makedirs(app_dir,0o755)
    os.makedirs(ico_dir,0o755)

    #DEBAIN Package description
    with open(os.path.join(DEBIAN_dir,'control'),'w') as f:
        dist_size = sum(os.path.getsize(os.path.join(pupil_capture_dir,f)) for f in os.listdir(pupil_capture_dir) if os.path.isfile(os.path.join(pupil_capture_dir,f)))
        content = '''\
Package: pupil-capture
Version: %s
Architecture: amd64
Maintainer: Pupil Labs <info@pupil-labs.com>
Priority: optional
Description: Pupil Capture is part of the Pupil Eye Tracking Platform
Installed-Size: %s
'''%(dpkg_deb_version(),dist_size/1024)
        f.write(content)
    os.chmod(os.path.join(DEBIAN_dir,'control'),0o644)

    #pre install script
    with open(os.path.join(DEBIAN_dir,'preinst'),'w') as f:
        content = '''\
#!/bin/sh
echo 'SUBSYSTEM=="usb",  ENV{DEVTYPE}=="usb_device", GROUP="plugdev", MODE="0664"' > /etc/udev/rules.d/10-libuvc.rules
udevadm trigger'''
        f.write(content)
    os.chmod(os.path.join(DEBIAN_dir,'preinst'),0o755)


    #bin_starter script
    with open(os.path.join(bin_dir,'pupil_capture'),'w') as f:
        content = '''\
#!/bin/sh
exec /opt/pupil_capture/pupil_capture "$@"'''
        f.write(content)
    os.chmod(os.path.join(bin_dir,'pupil_capture'),0o755)


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
Name[en_US]=Pupil Capture
Actions=Terminal;

[Desktop Action Terminal]
Name=Open in Terminal
Exec=x-terminal-emulator -e pupil_capture'''
        f.write(content)
    os.chmod(os.path.join(app_dir,'pupil_capture.desktop'),0o644)

    #copy icon:
    shutil.copy('pupil-capture.svg',ico_dir)
    os.chmod(os.path.join(ico_dir,'pupil-capture.svg'),0o644)

    #copy the actual application
    shutil.copytree(distribtution_dir,opt_dir)
    # set permissions
    for root, dirs, files in os.walk(opt_dir):
        for name in files:
            if name == 'pupil_capture':
                os.chmod(os.path.join(root,name),0o755)
            else:
                os.chmod(os.path.join(root,name),0o644)
        for name in dirs:
            os.chmod(os.path.join(root,name),0o755)
    os.chmod(opt_dir,0o755)


    #run dpkg_deb
    call('fakeroot dpkg-deb --build %s'%deb_root,shell=True)

    print( 'DONE!')
