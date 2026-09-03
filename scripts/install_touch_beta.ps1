param(
    [string]$InstallRoot = "",
    [switch]$Quiet,
    [switch]$NoLaunch,
    [switch]$SkipShortcuts,
    [switch]$SkipRegistry
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Show-InstallerMessage {
    param([string]$Message, [int]$Icon = 64)
    if ($Quiet) {
        return
    }
    $shell = New-Object -ComObject WScript.Shell
    $null = $shell.Popup($Message, 0, "TOUCH Beta", $Icon)
}

function Resolve-FullPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath(
        [Environment]::ExpandEnvironmentVariables($Path)
    ).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
}

try {
    $sourceExe = Join-Path $PSScriptRoot "TOUCH-Beta.exe"
    $sourceManifest = Join-Path $PSScriptRoot "package_manifest.json"
    $sourceReadme = Join-Path $PSScriptRoot "README-PORTABLE.txt"
    $sourceUninstaller = Join-Path $PSScriptRoot "uninstall_touch_beta.ps1"
    foreach ($required in @(
        $sourceExe,
        $sourceManifest,
        $sourceReadme,
        $sourceUninstaller
    )) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required installer payload is missing: $required"
        }
    }

    $manifest = Get-Content -LiteralPath $sourceManifest -Raw |
        ConvertFrom-Json
    $version = [string]$manifest.version
    if (-not $InstallRoot) {
        $InstallRoot = Join-Path $env:LOCALAPPDATA "Programs\TOUCH Beta"
    }
    $InstallRoot = Resolve-FullPath $InstallRoot
    if ($InstallRoot.Length -lt 8) {
        throw "Refusing unsafe installation root: $InstallRoot"
    }

    $targetExe = Join-Path $InstallRoot "TOUCH Beta.exe"
    $runningTarget = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ExecutablePath -and
            [string]::Equals(
                (Resolve-FullPath $_.ExecutablePath),
                $targetExe,
                [System.StringComparison]::OrdinalIgnoreCase
            )
        }
    if ($runningTarget) {
        throw "TOUCH Beta is running. Close it before installing this update."
    }

    New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
    $temporaryExe = "$targetExe.new"
    Copy-Item -LiteralPath $sourceExe -Destination $temporaryExe -Force
    Move-Item -LiteralPath $temporaryExe -Destination $targetExe -Force
    Copy-Item -LiteralPath $sourceManifest -Destination (
        Join-Path $InstallRoot "package_manifest.json"
    ) -Force
    Copy-Item -LiteralPath $sourceReadme -Destination (
        Join-Path $InstallRoot "README-PORTABLE.txt"
    ) -Force
    Copy-Item -LiteralPath $sourceUninstaller -Destination (
        Join-Path $InstallRoot "Uninstall TOUCH Beta.ps1"
    ) -Force

    if (-not $SkipShortcuts) {
        $shell = New-Object -ComObject WScript.Shell
        $shortcutTargets = @(
            (Join-Path ([Environment]::GetFolderPath("Desktop")) "TOUCH Beta.lnk"),
            (Join-Path ([Environment]::GetFolderPath("Programs")) "TOUCH\TOUCH Beta.lnk")
        )
        foreach ($shortcutPath in $shortcutTargets) {
            New-Item -ItemType Directory -Path (
                Split-Path -Parent $shortcutPath
            ) -Force | Out-Null
            $shortcut = $shell.CreateShortcut($shortcutPath)
            $shortcut.TargetPath = $targetExe
            $shortcut.WorkingDirectory = $InstallRoot
            $shortcut.IconLocation = "$targetExe,0"
            $shortcut.Description = "TOUCH Beta $version"
            $shortcut.Save()
        }
    }

    if (-not $SkipRegistry) {
        $uninstallScript = Join-Path $InstallRoot "Uninstall TOUCH Beta.ps1"
        $uninstallKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\TOUCHBeta"
        New-Item -Path $uninstallKey -Force | Out-Null
        $uninstallCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$uninstallScript`""
        $estimatedSizeKb = [int][Math]::Ceiling(
            (Get-Item -LiteralPath $targetExe).Length / 1KB
        )
        New-ItemProperty -Path $uninstallKey -Name DisplayName -Value "TOUCH Beta" -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $uninstallKey -Name DisplayVersion -Value $version -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $uninstallKey -Name Publisher -Value "TOUCH Research System" -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $uninstallKey -Name DisplayIcon -Value "$targetExe,0" -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $uninstallKey -Name InstallLocation -Value $InstallRoot -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $uninstallKey -Name UninstallString -Value $uninstallCommand -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $uninstallKey -Name QuietUninstallString -Value "$uninstallCommand -Quiet" -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $uninstallKey -Name EstimatedSize -Value $estimatedSizeKb -PropertyType DWord -Force | Out-Null
        New-ItemProperty -Path $uninstallKey -Name NoModify -Value 1 -PropertyType DWord -Force | Out-Null
        New-ItemProperty -Path $uninstallKey -Name NoRepair -Value 1 -PropertyType DWord -Force | Out-Null
    }

    if (-not $NoLaunch) {
        Start-Process -FilePath $targetExe
    }
    Show-InstallerMessage "TOUCH Beta $version was installed successfully."
    Write-Output "installed=$targetExe"
    exit 0
}
catch {
    Show-InstallerMessage "Installation failed:`n$($_.Exception.Message)" 16
    Write-Error $_
    exit 1
}
