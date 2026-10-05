<#
Audio Scribe installer for Windows.

Easiest: double-click install.bat. Or run in PowerShell:

    powershell -ExecutionPolicy Bypass -File install.ps1 [options]

Options:
    -WithStems    also install stem separation (Demucs, about 1 GB)
    -NoStems      skip stem separation without asking
    -Cuda         use an NVIDIA GPU build of PyTorch for stems
    -Yes          accept the default answer for every question
    -Uninstall    remove the shortcuts and the .venv and .tools folders

The app stays in this folder. A private Python 3.12 environment is created
in .venv here, so nothing touches any other Python on the machine.
#>
param(
    [switch]$WithStems,
    [switch]$NoStems,
    [switch]$Cuda,
    [switch]$Yes,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
$AppName = 'Audio Scribe'
$PythonVersion = '3.12'
$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $AppDir '.venv'
$Py = Join-Path $Venv 'Scripts\python.exe'
$PyW = Join-Path $Venv 'Scripts\pythonw.exe'
$AssetsDir = Join-Path $AppDir 'audioscribe\assets'
$IconPath = Join-Path $AssetsDir 'audio-scribe.ico'
$ToolsDir = Join-Path $AppDir '.tools'
$TorchVersion = '2.14.0'

# uv (the tool that sets up Python and installs packages) is downloaded at a
# fixed version and checked against these SHA-256 hashes before it is used.
# The hashes match the ones published with the uv 0.12.17 release on GitHub.
$UvVersion = '0.12.17'
$UvHashes = @{
    'x86_64-pc-windows-msvc'  = 'a252121d5b59398fcb137c6ea448176459a44010f33f67e0072305a637119ca7'
    'aarch64-pc-windows-msvc' = '3e1aa6849d77f0e00dc865e4afab5c5b32de053e21fe35bf5ad5cec3734ec976'
}

$StartMenuLink = Join-Path ([Environment]::GetFolderPath('Programs')) "$AppName.lnk"
$DesktopLink = Join-Path ([Environment]::GetFolderPath('Desktop')) "$AppName.lnk"

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Note([string]$Message) {
    Write-Host "    $Message"
}

function Stop-WithError([string]$Message) {
    Write-Host ''
    Write-Host "Error: $Message" -ForegroundColor Red
    exit 1
}

function Confirm-Choice([string]$Question, [bool]$Default) {
    if ($Yes) { return $Default }
    if ($Default) { $hint = '[Y/n]' } else { $hint = '[y/N]' }
    $answer = Read-Host "$Question $hint"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
    return $answer.Trim().ToLower().StartsWith('y')
}

function Invoke-Checked([string]$Exe, [string[]]$ArgList, [string]$What) {
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { Stop-WithError "$What failed (exit code $LASTEXITCODE)." }
}

function Invoke-Uv([string[]]$ArgList, [string]$What) {
    Invoke-Checked $script:Uv (@('pip', 'install', '--python', $Py) + $ArgList) $What
}

if ($Uninstall) {
    Write-Step "Removing $AppName"
    foreach ($link in @($StartMenuLink, $DesktopLink)) {
        if (Test-Path $link) { Remove-Item $link -Force }
    }
    if (Test-Path $Venv) { Remove-Item $Venv -Recurse -Force }
    if (Test-Path $ToolsDir) { Remove-Item $ToolsDir -Recurse -Force }
    Write-Note 'Removed the shortcuts and the .venv and .tools folders.'
    Write-Note "The source folder was left in place: $AppDir"
    Write-Note "Downloaded models stay in $env:USERPROFILE\.cache\huggingface (delete it to free the space)."
    exit 0
}

if (-not (Test-Path (Join-Path $AppDir 'audioscribe\__init__.py'))) {
    Stop-WithError 'Run this script from inside the Audio Scribe folder.'
}

# 1. uv, which manages Python and the packages ------------------------------------------
function Find-Uv {
    # A uv you installed yourself is used as is.
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $local = Join-Path $ToolsDir 'uv.exe'
    if (Test-Path $local) { return $local }
    return $null
}

function Install-PinnedUv {
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { $target = 'aarch64-pc-windows-msvc' }
    else { $target = 'x86_64-pc-windows-msvc' }
    $expected = $UvHashes[$target]
    $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-$target.zip"
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("uv-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    $zip = Join-Path $tmp 'uv.zip'
    try {
        Write-Step "Downloading uv $UvVersion and checking its SHA-256 hash"
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        $actual = (Get-FileHash -Path $zip -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $expected) {
            Stop-WithError "uv download failed the hash check (got $actual). Nothing was installed."
        }
        Write-Note 'Hash matches.'
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        if (-not (Test-Path $ToolsDir)) { New-Item -ItemType Directory -Path $ToolsDir | Out-Null }
        Copy-Item (Join-Path $tmp 'uv.exe') (Join-Path $ToolsDir 'uv.exe') -Force
    } finally {
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
    return (Join-Path $ToolsDir 'uv.exe')
}

$script:Uv = Find-Uv
if (-not $script:Uv) { $script:Uv = Install-PinnedUv }
Write-Note "Using uv at $script:Uv"

# 2. Python environment -------------------------------------------------------------------
$reuse = $false
if (Test-Path $Py) {
    # Windows PowerShell treats redirected stderr as an error, so relax that here.
    $ErrorActionPreference = 'Continue'
    & $Py -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>$null | Out-Null
    $reuse = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = 'Stop'
}
if ($reuse) {
    Write-Step 'Reusing the existing environment in .venv'
} else {
    if (Test-Path $Venv) { Remove-Item $Venv -Recurse -Force }
    Write-Step "Creating a Python $PythonVersion environment in .venv (uv downloads Python if needed)"
    Invoke-Checked $script:Uv @('venv', '--python', $PythonVersion, $Venv) 'Creating the environment'
}

Write-Step "Installing the app's packages (first time is a few hundred MB)"
# --require-hashes: every package must match the SHA-256 in the file or nothing installs.
Invoke-Uv @('--require-hashes', '-r', (Join-Path $AppDir 'requirements.txt')) 'Installing packages'
Invoke-Uv @('--require-hashes', '--no-deps', '-r', (Join-Path $AppDir 'requirements-nodeps.txt')) 'Installing Basic Pitch'

# 3. Optional stem separation ----------------------------------------------------------------
$installStems = $false
if ($WithStems) {
    $installStems = $true
} elseif (-not $NoStems) {
    Write-Step 'Optional: stem separation'
    Write-Note 'Splits a song into vocals, bass, drums, and everything else before analyzing.'
    Write-Note 'Lyrics and notes come out much cleaner on full songs. Adds about 1 GB of downloads.'
    $installStems = Confirm-Choice 'Install stem separation?' $false
}

if ($installStems) {
    $useCuda = [bool]$Cuda
    if (-not $useCuda -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        $useCuda = Confirm-Choice 'An NVIDIA GPU was found. Use it for stem separation? (about 3 GB more)' $false
    }
    $torchCpu = @('--require-hashes', '-r', (Join-Path $AppDir 'requirements-torch-windows.txt'))
    if ($useCuda) {
        Write-Step "Installing PyTorch $TorchVersion with NVIDIA GPU support (official PyTorch index)"
        & $script:Uv pip install --python $Py "torch==$TorchVersion" --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'The GPU build could not be installed, using the CPU build instead.' -ForegroundColor Yellow
            Invoke-Uv $torchCpu 'Installing PyTorch'
        }
    } else {
        Write-Step "Installing PyTorch $TorchVersion (CPU build)"
        Invoke-Uv $torchCpu 'Installing PyTorch'
    }
    Write-Step 'Installing Demucs'
    Invoke-Uv @('--require-hashes', '--no-deps', '-r', (Join-Path $AppDir 'requirements-stems.txt')) 'Installing Demucs'
}

# 4. Check that everything imports -------------------------------------------------------------
Write-Step 'Checking the install'
$env:PYTHONPATH = $AppDir
Push-Location $AppDir
try {
    & $Py -m audioscribe --check
    $checkCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($checkCode -ne 0) { Stop-WithError 'The check failed. See the list above for what is missing.' }

# 5. Icon and shortcuts -------------------------------------------------------------------------
Write-Step 'Adding shortcuts'
$env:QT_QPA_PLATFORM = 'offscreen'
& $Py -m audioscribe --write-icons $AssetsDir | Out-Null
Remove-Item Env:\QT_QPA_PLATFORM -ErrorAction SilentlyContinue

if (-not (Test-Path $PyW)) { $PyW = $Py }

function New-AppShortcut([string]$Path) {
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut($Path)
    $link.TargetPath = $PyW
    $link.Arguments = '-m audioscribe'
    $link.WorkingDirectory = $AppDir
    if (Test-Path $IconPath) { $link.IconLocation = "$IconPath,0" }
    $link.Description = 'Transcribe the words and find the notes in any audio or video file'
    $link.Save()
}

New-AppShortcut $StartMenuLink
Write-Note 'Added to the Start menu.'
if (Confirm-Choice 'Add a desktop shortcut too?' $true) {
    New-AppShortcut $DesktopLink
    Write-Note 'Added a desktop shortcut.'
}

Write-Step 'Done'
Write-Note "Start $AppName from the Start menu."
Write-Note 'Whisper and Demucs models download the first time you use them.'
