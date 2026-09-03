param(
    [Parameter(Mandatory = $true)]
    [string]$OneFileExe,
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-FullPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath(
        [Environment]::ExpandEnvironmentVariables($Path)
    ).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
}

$repoRoot = Resolve-FullPath (Join-Path $PSScriptRoot "..")
$appRoot = Join-Path $repoRoot "bayspec_wavelength_shift_app"
$versionPath = Join-Path $appRoot "release_manifests\beta\VERSION.json"
$versionManifest = Get-Content -LiteralPath $versionPath -Raw |
    ConvertFrom-Json
$version = [string]$versionManifest.version
$numericVersion = $version.Split("-", 2)[0]
$OneFileExe = Resolve-FullPath $OneFileExe
if (-not (Test-Path -LiteralPath $OneFileExe -PathType Leaf)) {
    throw "One-file executable does not exist: $OneFileExe"
}

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $appRoot "release_packages\beta-v$numericVersion"
}
$OutputDirectory = Resolve-FullPath $OutputDirectory
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$iexpress = Join-Path $env:WINDIR "System32\iexpress.exe"
if (-not (Test-Path -LiteralPath $iexpress -PathType Leaf)) {
    throw "Windows IExpress was not found: $iexpress"
}

$stageRoot = Join-Path $env:TEMP (
    "TOUCHBetaInstaller_" + $PID + "_" + [Guid]::NewGuid().ToString("N")
)
$stageRoot = Resolve-FullPath $stageRoot
$safeTempRoot = Resolve-FullPath $env:TEMP
if (-not $stageRoot.StartsWith(
    $safeTempRoot + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Refusing unsafe staging path: $stageRoot"
}

New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null
try {
    $payloadExe = Join-Path $stageRoot "TOUCH-Beta.exe"
    Copy-Item -LiteralPath $OneFileExe -Destination $payloadExe -Force
    Copy-Item -LiteralPath (
        Join-Path $repoRoot "scripts\install_touch_beta.ps1"
    ) -Destination $stageRoot -Force
    Copy-Item -LiteralPath (
        Join-Path $repoRoot "scripts\uninstall_touch_beta.ps1"
    ) -Destination $stageRoot -Force
    Copy-Item -LiteralPath (
        Join-Path $repoRoot "docs\TOUCH_BETA_PORTABLE_README.txt"
    ) -Destination (Join-Path $stageRoot "README-PORTABLE.txt") -Force

    $payloadHash = (Get-FileHash -LiteralPath $payloadExe -Algorithm SHA256).Hash
    $packageManifest = [ordered]@{
        product = "TOUCH"
        edition = "Beta Portable One-File"
        version = $version
        build_id = [string]$versionManifest.build_id
        release_channel = "beta"
        architecture = "windows-x64"
        application_file = "TOUCH-Beta.exe"
        application_sha256 = $payloadHash
        runtime_model_packaging = "embedded_single_current_model"
        bayspec_helper_packaging = "embedded_x86_helper_and_vendor_dll"
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    $packageManifest | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (
            Join-Path $stageRoot "package_manifest.json"
        ) -Encoding UTF8

    @"
@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_touch_beta.ps1"
exit /b %errorlevel%
"@ | Set-Content -LiteralPath (
        Join-Path $stageRoot "install_touch_beta.cmd"
    ) -Encoding ASCII

    $setupStage = Join-Path $stageRoot "TOUCH-Beta-v$numericVersion-Setup.exe"
    $sedPath = Join-Path $stageRoot "touch_beta_installer.sed"
    $sed = @"
[Version]
Class=IEXPRESS
SEDVersion=3

[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=0
HideExtractAnimation=0
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=%InstallPrompt%
DisplayLicense=
FinishMessage=
TargetName=%TargetName%
FriendlyName=%FriendlyName%
AppLaunched=%AppLaunched%
PostInstallCmd=<None>
AdminQuietInstCmd=%AppLaunched%
UserQuietInstCmd=%AppLaunched%
SourceFiles=SourceFiles

[Strings]
InstallPrompt=Install TOUCH Beta $numericVersion for the current Windows user?
TargetName=$setupStage
FriendlyName=TOUCH Beta $numericVersion Installer
AppLaunched=install_touch_beta.cmd
FILE0=TOUCH-Beta.exe
FILE1=install_touch_beta.cmd
FILE2=install_touch_beta.ps1
FILE3=uninstall_touch_beta.ps1
FILE4=package_manifest.json
FILE5=README-PORTABLE.txt

[SourceFiles]
SourceFiles0=$stageRoot\

[SourceFiles0]
%FILE0%=
%FILE1%=
%FILE2%=
%FILE3%=
%FILE4%=
%FILE5%=
"@
    $sed | Set-Content -LiteralPath $sedPath -Encoding ASCII
    $iexpressProcess = Start-Process -FilePath $iexpress -ArgumentList @(
        "/N",
        "/Q",
        $sedPath
    ) -Wait -PassThru
    if (
        $iexpressProcess.ExitCode -ne 0 -or
        -not (Test-Path -LiteralPath $setupStage)
    ) {
        throw (
            "IExpress failed to create the setup executable. Exit code: " +
            $iexpressProcess.ExitCode
        )
    }

    $standaloneName = "TOUCH-Beta-v$numericVersion-Windows-x64.exe"
    $setupName = "TOUCH-Beta-v$numericVersion-Setup-Windows-x64.exe"
    $standaloneOutput = Join-Path $OutputDirectory $standaloneName
    $setupOutput = Join-Path $OutputDirectory $setupName
    Copy-Item -LiteralPath $OneFileExe -Destination $standaloneOutput -Force
    Copy-Item -LiteralPath $setupStage -Destination $setupOutput -Force

    $portableStage = Join-Path $stageRoot "portable"
    New-Item -ItemType Directory -Path $portableStage -Force | Out-Null
    Copy-Item -LiteralPath $standaloneOutput -Destination $portableStage -Force
    Copy-Item -LiteralPath (
        Join-Path $stageRoot "README-PORTABLE.txt"
    ) -Destination $portableStage -Force
    Copy-Item -LiteralPath (
        Join-Path $stageRoot "package_manifest.json"
    ) -Destination $portableStage -Force

    $standaloneHash = (Get-FileHash -LiteralPath $standaloneOutput -Algorithm SHA256).Hash
    $setupHash = (Get-FileHash -LiteralPath $setupOutput -Algorithm SHA256).Hash
    "$standaloneHash  $standaloneName" | Set-Content -LiteralPath (
        Join-Path $portableStage "SHA256SUMS.txt"
    ) -Encoding ASCII

    $zipOutput = Join-Path $OutputDirectory (
        "TOUCH-Beta-v$numericVersion-Portable-Windows-x64.zip"
    )
    Compress-Archive -Path (Join-Path $portableStage "*") -DestinationPath $zipOutput -CompressionLevel Optimal -Force
    $zipHash = (Get-FileHash -LiteralPath $zipOutput -Algorithm SHA256).Hash
    @(
        "$standaloneHash  $standaloneName",
        "$setupHash  $setupName",
        "$zipHash  $(Split-Path -Leaf $zipOutput)"
    ) | Set-Content -LiteralPath (
        Join-Path $OutputDirectory "SHA256SUMS.txt"
    ) -Encoding ASCII

    [ordered]@{
        version = $version
        standalone_exe = $standaloneOutput
        setup_exe = $setupOutput
        portable_zip = $zipOutput
        standalone_sha256 = $standaloneHash
        setup_sha256 = $setupHash
        portable_zip_sha256 = $zipHash
    } | ConvertTo-Json -Depth 3
}
finally {
    if (
        (Test-Path -LiteralPath $stageRoot) -and
        $stageRoot.StartsWith(
            $safeTempRoot + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            try {
                Remove-Item -LiteralPath $stageRoot -Recurse -Force -ErrorAction Stop
                break
            }
            catch {
                if ($attempt -eq 19) {
                    Write-Warning (
                        "Could not remove the verified temporary staging path: " +
                        $stageRoot
                    )
                    break
                }
                Start-Sleep -Milliseconds 250
            }
        }
    }
}
