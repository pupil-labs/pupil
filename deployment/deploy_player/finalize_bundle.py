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


mac_plist_document_type_str = '''
<key>CFBundleDocumentTypes</key>
        <array>
            <dict>
            <key>CFBundleTypeExtensions</key>
            <array>
            <string>*</string>
            </array>
            <key>CFBundleTypeMIMETypes</key>
            <array>
            <string>*/*</string>
            </array>
            <key>CFBundleTypeName</key>
            <string>folder</string>
            <key>CFBundleTypeOSTypes</key>
            <array>
            <string>****</string>
            </array>
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
            </dict>
        </array>
'''

split_str = """
</dict>
</plist>"""

if platform.system() == 'Darwin':
    print( "starting version stript:")
    write_version_file('dist/Pupil Player.app/Contents/MacOS')
    print( "created version file in app dir")

    shutil.rmtree('dist/Pupil Player')
    print( 'removed the non-app dist bundle')

    print( "hack injecting file type info in to info.plist")
    with open("dist/Pupil Player.app/Contents/Info.plist", "r") as f:
        txt = f.read() # read everything in the file
    txt = txt.replace(split_str,mac_plist_document_type_str + split_str)
    with open("dist/Pupil Player.app/Contents/Info.plist", "w") as f:
        f.write(txt)

    bundle_name = 'pupil_player_mac_os_x64_v%s'%dpkg_deb_version()
    bundle_dmg_name = 'Install Pupil Player'
    src_dir = 'dist'
    bundle_app_dir = os.path.join(src_dir,'Pupil Player.app/' )
    print( "Codesigning now")
    if call("codesign --force --verify --verbose -s 'Developer ID Application: Pupil Labs UG (haftungsbeschrankt) (R55K9ESN6B)' --deep '%s'"%bundle_app_dir,shell=True) != 0:
        print( Exception("Codesinging  failed"))
    # if call("spctl --assess --type execute '%s'"%bundle_app_dir,shell=True) != 0:
        # print Exception("Codesing verification  failed")
    call("ln -s /Applications/ %s/Applications"%src_dir,shell=True)
    call("hdiutil create -volname '%s' -srcfolder %s -format UDZO '%s.dmg'"%(bundle_dmg_name,src_dir,bundle_name),shell=True)

elif platform.system() == 'Windows':
    write_version_file(os.path.join('dist', 'Pupil Player'))

elif platform.system() == 'Linux':

    distribtution_dir = 'dist'
    pupil_player_dir =  os.path.join(distribtution_dir, 'pupil_player')

    print( "starting version stript:")
    write_version_file(pupil_player_dir)
    print( "created version file in dist folder")

    old_deb_dir = [d for d in os.listdir('.') if  d.startswith('pupil_player_')]
    for d in old_deb_dir:
        try:
            shutil.rmtree(d)
            print( 'removed deb structure dir: "%s"'%d)
        except:
            pass

    #lets build the structure for our deb package.
    deb_root = 'pupil_player_linux_os_x64_v%s'%dpkg_deb_version()
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
        dist_size = sum(os.path.getsize(os.path.join(pupil_player_dir,f)) for f in os.listdir(pupil_player_dir) if os.path.isfile(os.path.join(pupil_player_dir,f)))
        content = '''\
Package: pupil-player
Version: %s
Architecture: amd64
Maintainer: Pupil Labs <info@pupil-labs.com>
Priority: optional
Description: Pupil Player is part of the Pupil Eye Tracking Platform
Installed-Size: %s
'''%(dpkg_deb_version(),dist_size/1024)
        f.write(content)
    os.chmod(os.path.join(DEBIAN_dir,'control'),0o644)

    #bin_starter script
    with open(os.path.join(bin_dir,'pupil_player'),'w') as f:
        content = '''\
#!/bin/sh
exec /opt/pupil_player/pupil_player "$@"'''
        f.write(content)
    os.chmod(os.path.join(bin_dir,'pupil_player'),0o755)


    #.desktop entry
    with open(os.path.join(app_dir,'pupil_player.desktop'),'w') as f:
        content = '''\
[Desktop Entry]
Version=1.0
Type=Application
Name=Pupil Player
Comment=Eye Tracking Vizualizer Program
Exec=/opt/pupil_player/pupil_player %F
Terminal=false
Icon=pupil-player
Categories=Application;
Name[en_US]=Pupil Player'''
        f.write(content)
    os.chmod(os.path.join(app_dir,'pupil_player.desktop'),0o644)

    #copy icon:
    shutil.copy('pupil-player.svg',ico_dir)
    os.chmod(os.path.join(ico_dir,'pupil-player.svg'),0o644)

    #copy the actual application
    shutil.copytree(distribtution_dir,opt_dir)
    # set permissions
    for root, dirs, files in os.walk(opt_dir):
        for name in files:
            if name == 'pupil_player':
                os.chmod(os.path.join(root,name),0o755)
            else:
                os.chmod(os.path.join(root,name),0o644)
        for name in dirs:
            os.chmod(os.path.join(root,name),0o755)
    os.chmod(opt_dir,0o755)

    #run dpkg_deb
    call('fakeroot dpkg-deb --build %s'%deb_root,shell=True)

    print( 'DONE!')
