Set-PSDebug -Trace 1
$Env:PYTHONHASHSEED = 42

Write-Output "Downloading and installing packaging dependencies"
Invoke-WebRequest "https://www.win-rar.com/fileadmin/winrar-versions/winrar/winrar-x64-611.exe" -OutFile winrar.exe
.\winrar.exe /s

$release_dir = "pupil_$(git describe --tags --long)_windows_x64"
Write-Output "Creating bundle at $release_dir"
pyinstaller pupil_core.spec --noconfirm --log-level INFO --distpath $release_dir
python generate_msi_installer.py
Write-Output "Creating archive of $release_dir.msi"
& "C:\Program Files\WinRAR\Rar.exe" a "$release_dir.msi.rar" "$release_dir.msi"
