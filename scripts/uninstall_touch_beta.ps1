param([switch]$Quiet)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-FullPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath($Path).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar
    )
}

try {
    $installRoot = Resolve-FullPath $PSScriptRoot
    $allowedRoot = Resolve-FullPath (
        Join-Path $env:LOCALAPPDATA "Programs\TOUCH Beta"
    )
    if (-not [string]::Equals(
        $installRoot,
        $allowedRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing unsafe uninstall root: $installRoot"
    }

    $targetExe = Join-Path $installRoot "TOUCH Beta.exe"
    $runningTarget = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ExecutablePath -and
            [string]::Equals(
                (Resolve-FullPath $_.ExecutablePath),
                $targetExe,
                [System.StringComparison]::OrdinalIgnoreCase
            )
        }
    foreach ($process in $runningTarget) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }

    $shell = New-Object -ComObject WScript.Shell
    foreach ($shortcutPath in @(
        (Join-Path ([Environment]::GetFolderPath("Desktop")) "TOUCH Beta.lnk"),
        (Join-Path ([Environment]::GetFolderPath("Programs")) "TOUCH\TOUCH Beta.lnk")
    )) {
        if (-not (Test-Path -LiteralPath $shortcutPath -PathType Leaf)) {
            continue
        }
        $shortcut = $shell.CreateShortcut($shortcutPath)
        if ([string]::Equals(
            (Resolve-FullPath $shortcut.TargetPath),
            $targetExe,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            Remove-Item -LiteralPath $shortcutPath -Force
        }
    }

    Remove-Item -LiteralPath (
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\TOUCHBeta"
    ) -Recurse -Force -ErrorAction SilentlyContinue

    $escapedRoot = $installRoot.Replace("'", "''")
    $cleanup = "Start-Sleep -Milliseconds 800; " +
        "Remove-Item -LiteralPath '$escapedRoot' -Recurse -Force"
    $encoded = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($cleanup)
    )
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        $encoded
    ) -WindowStyle Hidden

    if (-not $Quiet) {
        $null = $shell.Popup(
            "TOUCH Beta was removed.",
            0,
            "TOUCH Beta",
            64
        )
    }
    exit 0
}
catch {
    if (-not $Quiet) {
        $shell = New-Object -ComObject WScript.Shell
        $null = $shell.Popup(
            "Uninstall failed:`n$($_.Exception.Message)",
            0,
            "TOUCH Beta",
            16
        )
    }
    Write-Error $_
    exit 1
}
