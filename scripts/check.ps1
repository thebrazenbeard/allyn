# Local-only ALLYN verification. Can be invoked from any working directory.
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$candidates = @(
    @{ Command = 'python'; Prefix = @() },
    @{ Command = 'python3'; Prefix = @() },
    @{ Command = 'py'; Prefix = @('-3') }
)

$selected = $null
foreach ($candidate in $candidates) {
    if (-not (Get-Command $candidate.Command -ErrorAction SilentlyContinue)) {
        continue
    }
    $commandName = $candidate.Command
    $prefix = $candidate.Prefix
    & $commandName @prefix -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>$null
    if ($LASTEXITCODE -eq 0) {
        $selected = $candidate
        break
    }
}

if ($null -eq $selected) {
    Write-Error 'ALLYN verification requires Python 3.11 or newer.'
    exit 2
}

$commandName = $selected.Command
$prefix = $selected.Prefix
$previousPythonPath = $env:PYTHONPATH
$previousCheckRunning = $env:ALLYN_CHECK_RUNNING
try {
    $env:PYTHONPATH = Join-Path $repoRoot 'src'
    $env:ALLYN_CHECK_RUNNING = '1'
    & $commandName @prefix -m unittest discover -s (Join-Path $repoRoot 'tests') -v
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $commandName @prefix -m allyn demo
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $commandName @prefix -m allyn cycle-demo
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:ALLYN_CHECK_RUNNING = $previousCheckRunning
}
