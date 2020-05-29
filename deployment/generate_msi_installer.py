import os
import re
import shutil
import textwrap
from pathlib import Path
from typing import List
from uuid import uuid4 as new_guid

# NOTE: you will need to have the WiX Toolset installed in order to run this script!
# 1. Download `wix311.exe` from https://github.com/wixtoolset/wix3/releases/tag/wix3112rtm
# 2. Then install, e.g. to default location.
# 3. Now add the binaries to the PATH. For default installation, they should be in
# C:\Program Files (x86)\WiX Toolset v3.11\bin

for child in Path(".").iterdir():
    if child.is_dir():
        match = re.search(
            r"pupil_v(?P<major>[0-9]+)\.(?P<minor>[0-9]+)-(?P<patch>[0-9]+)-\w+_windows_x64",
            child.name,
        )
        if match is not None:
            # NOTE: MSI only allows versions in the form of x.x.x.x, where all x are
            # integers, so we need to replace the '-' before patch with a '.'. Also we
            # want to prefix 'v' for display.
            raw_version = f"{match['major']}.{match['minor']}.{match['patch']}"
            version = f"v{raw_version}"
            base_dir = child
            print(f"Found release dir for version '{version}': {base_dir.name}")
            break
else:
    raise RuntimeError("Could not find any release dir!")

product_name = f"Pupil {version}"
company_short = "Pupil-Labs"
manufacturer = "Pupil-Labs GmbH"

package_description = f"{company_short} {product_name}"

# NOTE: Generating new GUIDs for product and upgrade code means that different
# installations will not conflict. This is the easiest workflow for enabling customers
# to install different versions alongside. This will however also mean that in the case
# of a patch, the users will always also have to uninstall the old version manually.
product_guid = new_guid()
product_upgrade_code = new_guid()


class SoftwareComponent:
    """Represents capture, player or service and collects all info for WiX XML."""

    def __init__(self, base_dir: Path, name: str, version: str):
        self.name = name
        self.base_dir = base_dir
        for p in self.base_dir.iterdir():
            if p.is_dir() and p.name.startswith(f"pupil_{self.name}_windows_x64"):
                self.dir = p
                break

        self.display_name = f"Pupil {self.name.capitalize()} {version}"

        self.counter = 0
        self.component_ids = []
        self.directory_root = []
        self.crawl_directory(directory=self.dir, tree_root=self.directory_root)

    def crawl_directory(self, directory: Path, tree_root: List) -> None:
        for p in directory.iterdir():
            self.counter += 1

            if p.is_file():
                if re.search(r"pupil_\w*\.exe", p.name):
                    # skip executable
                    continue
                component = {
                    "type": "component",
                    "id": f"FileComponent{self.counter}{self.name}",
                    "guid": new_guid(),
                    "file_id": f"File{self.counter}{self.name}",
                    "file_name": p.name,
                }
                tree_root.append(component)
                self.component_ids.append(component["id"])

            else:
                directory = {
                    "type": "directory",
                    "id": f"Dir{self.counter}{self.name}",
                    "name": p.name,
                    "content": [],
                }
                tree_root.append(directory)
                self.crawl_directory(p, tree_root=directory["content"])

    def parse_dir_data(self, root: List, indent: int = 0) -> str:
        text = ""
        for child in root:
            if child["type"] == "component":
                text += (
                    f"<Component Id='{child['id']}' Guid='{child['guid']}'>\n"
                    f"    <File Id='{child['file_id']}' Name='{child['file_name']}' DiskId='1' KeyPath='yes' />\n"
                    f"</Component>\n"
                )

            elif child["type"] == "directory":
                text += (
                    f"<Directory Id='{child['id']}' Name='{child['name']}'>\n"
                    f"{self.parse_dir_data(root=child['content'], indent=1)}"
                    f"</Directory>\n"
                )
        return textwrap.indent(text, "    " * indent)

    def directory_data(self) -> str:
        cap = self.name.capitalize()
        return f"""
                        <Directory Id='{cap}Dir' Name='{self.display_name}' FileSource="{self.dir.name}">
                            <Component Id='{cap}Executable' Guid='{new_guid()}'>
                                <File Id='{cap}EXE' Name='pupil_{self.name}.exe' DiskId='1' KeyPath='yes'>
                                    <Shortcut Id="startmenu{cap}" Directory="ProgramMenuDir" Name="{self.display_name}" WorkingDirectory='INSTALLDIR' Icon="{cap}Icon.exe" IconIndex="0" Advertise="yes" />
                                    <Shortcut Id="desktop{cap}" Directory="DesktopFolder" Name="{self.display_name}" WorkingDirectory='INSTALLDIR' Icon="{cap}Icon.exe" IconIndex="0" Advertise="yes" />
                                </File>
                            </Component>
                            {self.parse_dir_data(root=self.directory_root, indent=7).strip()}
                        </Directory>
                        """

    def feature_data(self) -> str:
        cap = self.name.capitalize()
        return f"""
            <Feature Id='Pupil{cap}Feature' Title='Pupil {cap}' Description='The Pupil {cap} software component.' Level='1'>
                <ComponentRef Id='{cap}Executable' />
        {
            "".join(f'''
                <ComponentRef Id='{component_id}' />'''
                for component_id in self.component_ids
            )
        }
            </Feature>
            """

    def icon_data(self) -> str:
        cap = self.name.capitalize()
        return f"""
        <Icon Id="{cap}Icon.exe" SourceFile="{self.dir.name}\pupil_{self.name}.exe" />"""


capture_data = SoftwareComponent(base_dir, "capture", version)
player_data = SoftwareComponent(base_dir, "player", version)
service_data = SoftwareComponent(base_dir, "service", version)


template = f"""
<?xml version='1.0' encoding='windows-1252'?>
<Wix xmlns='http://schemas.microsoft.com/wix/2006/wi'>
    <Product Name='{product_name}' Id='{product_guid}' UpgradeCode='{product_upgrade_code}'
        Language='1033' Codepage='1252' Version='{raw_version}' Manufacturer='{manufacturer}'>

        <Package Id='*' Keywords='Installer'
        Description="{package_description}" Manufacturer='{manufacturer}'
        InstallerVersion='100' Languages='1033' Compressed='yes' SummaryCodepage='1252' />

        <Media Id='1' Cabinet='Cabinet.cab' EmbedCab='yes' DiskPrompt="CD-ROM #1" />
        <Property Id='DiskPrompt' Value="{package_description} Installer [1]" />

        <Directory Id='TARGETDIR' Name='SourceDir'>

            <Directory Id='ProgramFilesFolder' Name='PFiles'>
                <Directory Id='PupilLabs' Name='{company_short}'>
                    <Directory Id='INSTALLDIR' Name='{product_name}'>
                        {capture_data.directory_data()}
                        {player_data.directory_data()}
                        {service_data.directory_data()}
                    </Directory>
                </Directory>
            </Directory>

            <Directory Id="ProgramMenuFolder" Name="Programs">
                <Directory Id="ProgramMenuDir" Name="{product_name}">
                    <Component Id="ProgramMenuDir" Guid="{new_guid()}">
                        <RemoveFolder Id='ProgramMenuDir' On='uninstall' />
                        <RegistryValue Root='HKCU' Key='Software\[Manufacturer]\[ProductName]\{version}' Type='string' Value='' KeyPath='yes' />
                    </Component>
                </Directory>
            </Directory>

            <Directory Id="DesktopFolder" Name="Desktop" />
        </Directory>

        <Feature Id='Complete' Title='{product_name}' Description='The full suite of Pupil Capture, Player and Service.'
            Display='collapse' Level='1' ConfigurableDirectory='INSTALLDIR'>
            <ComponentRef Id='ProgramMenuDir' />

            {capture_data.feature_data()}
            {player_data.feature_data()}
            {service_data.feature_data()}

        </Feature>

        {capture_data.icon_data()}
        {player_data.icon_data()}
        {service_data.icon_data()}

        <WixVariable Id="WixUIBannerBmp" Value="..\\msi_graphics\\banner.bmp" />
        <WixVariable Id="WixUIDialogBmp" Value="..\\msi_graphics\\dialog.bmp" />

        <!--
            Copied from https://github.com/wixtoolset/wix3/blob/2d8b37764ec8453dc78dbc91c0fd444feaa6666d/src/ext/UIExtension/wixlib/WixUI_FeatureTree.wxs
            And adjusted to not contain the License Dialog anymore.
        -->
        <UI Id="WixUI_FeatureTree">
            <TextStyle Id="WixUI_Font_Normal" FaceName="Tahoma" Size="8" />
            <TextStyle Id="WixUI_Font_Bigger" FaceName="Tahoma" Size="12" />
            <TextStyle Id="WixUI_Font_Title" FaceName="Tahoma" Size="9" Bold="yes" />

            <Property Id="DefaultUIFont" Value="WixUI_Font_Normal" />
            <Property Id="WixUI_Mode" Value="FeatureTree" />

            <DialogRef Id="ErrorDlg" />
            <DialogRef Id="FatalError" />
            <DialogRef Id="FilesInUse" />
            <DialogRef Id="MsiRMFilesInUse" />
            <DialogRef Id="PrepareDlg" />
            <DialogRef Id="ProgressDlg" />
            <DialogRef Id="ResumeDlg" />
            <DialogRef Id="UserExit" />

            <Publish Dialog="ExitDialog" Control="Finish" Event="EndDialog" Value="Return" Order="999">1</Publish>

            <Publish Dialog="WelcomeDlg" Control="Next" Event="NewDialog" Value="CustomizeDlg">NOT Installed</Publish>
            <Publish Dialog="WelcomeDlg" Control="Next" Event="NewDialog" Value="VerifyReadyDlg">Installed AND PATCH</Publish>

            <Publish Dialog="CustomizeDlg" Control="Back" Event="NewDialog" Value="MaintenanceTypeDlg" Order="1">Installed</Publish>
            <Publish Dialog="CustomizeDlg" Control="Back" Event="NewDialog" Value="WelcomeDlg" Order="2">NOT Installed</Publish>
            <Publish Dialog="CustomizeDlg" Control="Next" Event="NewDialog" Value="VerifyReadyDlg">1</Publish>

            <Publish Dialog="VerifyReadyDlg" Control="Back" Event="NewDialog" Value="CustomizeDlg" Order="1">NOT Installed OR WixUI_InstallMode = "Change"</Publish>
            <Publish Dialog="VerifyReadyDlg" Control="Back" Event="NewDialog" Value="MaintenanceTypeDlg" Order="2">Installed AND NOT PATCH</Publish>
            <Publish Dialog="VerifyReadyDlg" Control="Back" Event="NewDialog" Value="WelcomeDlg" Order="3">Installed AND PATCH</Publish>

            <Publish Dialog="MaintenanceWelcomeDlg" Control="Next" Event="NewDialog" Value="MaintenanceTypeDlg">1</Publish>

            <Publish Dialog="MaintenanceTypeDlg" Control="ChangeButton" Event="NewDialog" Value="CustomizeDlg">1</Publish>
            <Publish Dialog="MaintenanceTypeDlg" Control="RepairButton" Event="NewDialog" Value="VerifyReadyDlg">1</Publish>
            <Publish Dialog="MaintenanceTypeDlg" Control="RemoveButton" Event="NewDialog" Value="VerifyReadyDlg">1</Publish>
            <Publish Dialog="MaintenanceTypeDlg" Control="Back" Event="NewDialog" Value="MaintenanceWelcomeDlg">1</Publish>
        </UI>
        <UIRef Id="WixUI_Common" />
    

    </Product>
</Wix>
"""

wix_file = base_dir / f"{base_dir.name}.wxs"
print(f"Generating WiX file at {wix_file}")
with wix_file.open("w") as f:
    f.write(template.strip())

os.chdir(str(base_dir))
print("Running candle")
os.system(f'candle.exe "{base_dir.name}.wxs"')
print("Running light")
os.system(f'light.exe -ext WixUIExtension "{base_dir.name}.wixobj"')
print("Done.")

print("Copy Installer")
shutil.copyfile(f"{base_dir.name}.msi", f"../{base_dir.name}.msi")
print("Cleanup")
Path(f"{base_dir.name}.wxs").unlink()
Path(f"{base_dir.name}.wixobj").unlink()
Path(f"{base_dir.name}.wixpdb").unlink()
Path(f"{base_dir.name}.msi").unlink()
print("Finished!")
