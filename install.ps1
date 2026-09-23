# Intelligence Network installer for Windows (PowerShell 5.1+).
# Downloads the release zip, verifies the SHA-256 sidecar and installs
# intelligence.exe under %LOCALAPPDATA%\Programs\intelligence.
$ErrorActionPreference = 'Stop'

$version = $env:INTELLIGENCE_VERSION
if (-not $version) { $version = 'v1.0.0' }
if ($version -notmatch '^v') { $version = "v$version" }

if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    Write-Error "Intelligence Network has no Windows artifact for $($env:PROCESSOR_ARCHITECTURE) yet; build from source."
    exit 2
}

$artifactArch = 'windows-x86_64'
$baseUrl = $env:INTELLIGENCE_RELEASE_BASE_URL
if (-not $baseUrl) {
    $baseUrl = "https://github.com/Cowde-Labs/intelligence-network/releases/download/$version"
}

$artifact = "intelligence-network-$version-$artifactArch.zip"
$checksumAsset = "$artifact.sha256"

$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("intelligence-install-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmpDir | Out-Null
try {
    $archivePath = Join-Path $tmpDir $artifact
    $checksumPath = Join-Path $tmpDir $checksumAsset

    Invoke-WebRequest -Uri "$baseUrl/$artifact" -OutFile $archivePath -UseBasicParsing
    Invoke-WebRequest -Uri "$baseUrl/$checksumAsset" -OutFile $checksumPath -UseBasicParsing

    $sidecar = (Get-Content -Raw $checksumPath).Trim()
    $expected = ($sidecar -split '\s+')[0]
    if ($expected -notmatch '^[0-9a-fA-F]{64}$') {
        Write-Error "release checksum is missing or malformed"
        exit 1
    }
    $actual = (Get-FileHash -Algorithm SHA256 -Path $archivePath).Hash
    if ($actual -ine $expected) {
        Write-Error "release checksum mismatch for $artifact"
        exit 1
    }

    Expand-Archive -Path $archivePath -DestinationPath $tmpDir -Force
    $packageDir = Join-Path $tmpDir "intelligence-network-$version-$artifactArch"
    $exe = Join-Path $packageDir 'intelligence.exe'
    if (-not (Test-Path $exe)) {
        Write-Error "release archive does not contain intelligence.exe"
        exit 1
    }

    $installDir = Join-Path $env:LOCALAPPDATA 'Programs\intelligence'
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
    Copy-Item $exe (Join-Path $installDir 'intelligence.exe') -Force

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (($userPath -split ';') -notcontains $installDir) {
        [Environment]::SetEnvironmentVariable('Path', "$userPath;$installDir", 'User')
    }

    Write-Host "Installed Intelligence Network $version to $installDir\intelligence.exe"
    Write-Host "Open a new terminal and run: intelligence up"
}
finally {
    Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
}
