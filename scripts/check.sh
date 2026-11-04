#!/usr/bin/env bash
# Local-only ALLYN verification. Can be invoked from any working directory.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"

python_bin=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
        "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
        python_bin="$candidate"
        break
    fi
done

if [[ -z "$python_bin" ]]; then
    printf '%s\n' 'ALLYN verification requires Python 3.11 or newer.' >&2
    exit 2
fi

ALLYN_CHECK_RUNNING=1 PYTHONPATH="${repo_root}/src" "$python_bin" -m unittest discover -s "${repo_root}/tests" -v
PYTHONPATH="${repo_root}/src" "$python_bin" -m allyn demo
PYTHONPATH="${repo_root}/src" "$python_bin" -m allyn cycle-demo
