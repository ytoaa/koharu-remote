param(
    [string]$ConfigPath = (Join-Path $HOME '.koharu\config.toml'),
    [string]$BackupPath = ''
)
$ErrorActionPreference = 'Stop'

function Get-CustomParts([string]$Text) {
    $custom = [Collections.Generic.List[string]]::new()
    $other = [Collections.Generic.List[string]]::new()
    $state = ''
    $selected = $false
    foreach ($line in ($Text -split '\r?\n')) {
        # Header-looking lines inside TOML multiline strings are data, not sections.
        if ($state -eq '' -and $line -match '^\s*\[\[?([^\]]+)\]\]?\s*(?:#.*)?$') {
            $selected = $Matches[1] -match '^custom-models(?:\.|$)'
        }
        if ($selected) { $custom.Add($line) } else { $other.Add($line) }
        for ($i = 0; $i -lt $line.Length; $i++) {
            $c = $line[$i]
            $triple = if ($i + 2 -lt $line.Length) { $line.Substring($i, 3) } else { '' }
            if ($state -eq '') {
                if ($c -eq '#') { break }
                if ($triple -eq "'''" -or $triple -eq '"""') { $state = $triple; $i += 2 }
                elseif ($c -eq "'" -or $c -eq '"') { $state = [string]$c }
            } elseif ($state -eq '"') {
                if ($c -eq '\') { $i++ } elseif ($c -eq '"') { $state = '' }
            } elseif ($state -eq "'") {
                if ($c -eq "'") { $state = '' }
            } elseif ($state -eq '"""') {
                if ($c -eq '\') { $i++ } elseif ($triple -eq '"""') { $state = ''; $i += 2 }
            } elseif ($state -eq "'''" -and $triple -eq "'''") { $state = ''; $i += 2 }
        }
    }
    return @{ Custom = ($custom -join "`n"); Other = ($other -join "`n") }
}

function Restore-CustomSettings([string]$Current, [string]$Previous) {
    if (-not (Test-Path -LiteralPath $Previous -PathType Leaf)) { throw 'Pre-Qwen configuration backup not found.' }
    $old = Get-CustomParts ([IO.File]::ReadAllText($Previous))
    $now = Get-CustomParts ([IO.File]::ReadAllText($Current))
    $suffix = [Guid]::NewGuid().ToString('N')
    $retained = "$Current.before-rollback-$suffix"
    $temporary = "$Current.restore-$suffix"
    try {
        [IO.File]::WriteAllText($temporary, $now.Other.TrimEnd() + "`n" + $old.Custom.TrimEnd() + "`n", [Text.UTF8Encoding]::new($false))
        [IO.File]::Replace($temporary, $Current, $retained)
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
    }
    Write-Output 'CUSTOM_SETTINGS_RESTORED: translation/provider settings retained; reopen the previous Windows app.'
}

if ($MyInvocation.InvocationName -ne '.') {
    if (-not $BackupPath) { $BackupPath = Join-Path (Split-Path $ConfigPath) 'config.before-qwen-0.6.0.toml' }
    Restore-CustomSettings $ConfigPath $BackupPath
}
