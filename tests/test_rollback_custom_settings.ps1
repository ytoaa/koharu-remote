$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\rollback-custom-settings.ps1')
$root = Join-Path ([IO.Path]::GetTempPath()) ('koharu-rollback-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($root) | Out-Null
try {
    $current = Join-Path $root 'config.toml'
    $previous = Join-Path $root 'config.before-qwen-0.6.0.toml'
    $old = @'
[translation]
model = "old-model"
[custom-models]
enabled = true
[custom-models.options.inpainting]
steps = 4
[providers]
name = "old-provider"
'@
    $now = @'
[translation]
model = "new-model"
prompt = '''
[custom-models.fake]
This is prompt data, not a section.
'''
[custom-models]
enabled = false
[custom-models.options.qwen_inpainting]
max_side = 4096
[providers]
name = "new-provider"
'@
    [IO.File]::WriteAllText($current, $now)
    [IO.File]::WriteAllText($previous, $old)
    Restore-CustomSettings $current $previous
    $result = [IO.File]::ReadAllText($current)
    foreach ($required in @('model = "new-model"','name = "new-provider"','[custom-models.fake]','enabled = true','steps = 4')) {
        if (-not $result.Contains($required)) { throw "Missing preserved/restored data: $required" }
    }
    if ($result.Contains('max_side = 4096') -or $result.Contains('old-provider')) { throw 'Restore changed unrelated settings or retained incompatible Qwen options.' }
    $saved = @(Get-ChildItem -LiteralPath $root -Filter 'config.toml.before-rollback-*')
    if ($saved.Count -ne 1 -or [IO.File]::ReadAllText($saved[0].FullName) -ne $now) { throw 'Current configuration was not retained.' }
    Write-Output 'WINDOWS_CUSTOM_ROLLBACK_CONTRACT_PASS'
} finally { Remove-Item -LiteralPath $root -Recurse -Force }
