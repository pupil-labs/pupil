Set-PSDebug -Trace 1
$Env:PYTHONHASHSEED = 42

$release_dir = "pupil_$(git describe --tags --long)_windows_x64"

Write-Output "Downloading and installing packaging dependencies"
Invoke-WebRequest "https://www.win-rar.com/fileadmin/winrar-versions/winrar/winrar-x64-611.exe" -OutFile winrar.exe
.\winrar.exe /s

Write-Output "Creating bundle at $release_dir"
pyinstaller pupil_core.spec --noconfirm --log-level INFO --distpath $release_dir
