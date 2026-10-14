$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'bin\koharu-remote.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw 'Remote build executable not found.' }
Start-Process -FilePath $exe -WorkingDirectory (Split-Path $exe)
