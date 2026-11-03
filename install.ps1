# One-time setup: creates .venv, installs the Python packages (~1.5 GB with the NVIDIA libraries)
# and adds an "Audio Censor" shortcut to the Desktop. Run via install.bat.
$dir = $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python 3.10 or newer is required: https://www.python.org/downloads/"
    exit 1
}
if (-not (Test-Path "$dir\.venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv "$dir\.venv"
    if ($LASTEXITCODE -ne 0) { Write-Host "Could not create .venv"; exit 1 }
}
$py = "$dir\.venv\Scripts\python.exe"
Write-Host "Installing packages (this can take a few minutes)..."
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -r "$dir\requirements.txt"
if ($LASTEXITCODE -ne 0) { Write-Host "Package install failed"; exit 1 }

$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut([Environment]::GetFolderPath("Desktop") + "\Audio Censor.lnk")
$lnk.TargetPath = "$dir\.venv\Scripts\pythonw.exe"
$lnk.Arguments = "`"$dir\app.pyw`""
$lnk.WorkingDirectory = $dir
$lnk.IconLocation = "$env:SystemRoot\System32\SndVol.exe,0"
$lnk.Description = "Mutes curse words in PC audio"
$lnk.Save()
Write-Host "Desktop shortcut created: Audio Censor"

if (-not (Get-CimInstance Win32_SoundDevice | Where-Object Name -match "VB-Audio")) {
    Write-Host ""
    Write-Host "One more step: install VB-Cable from https://vb-audio.com/Cable/"
    Write-Host "(unzip it, right-click VBCABLE_Setup_x64.exe > Run as administrator > Install Driver)."
    Write-Host "Windows may switch your sound to 'CABLE Input' afterwards - the app switches it back."
} else {
    Write-Host "VB-Cable is installed. Setup complete."
}
