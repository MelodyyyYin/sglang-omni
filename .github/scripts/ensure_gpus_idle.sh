#!/usr/bin/env bash
# note (Yue Yin): thin shim — orphan kill is now built into delete_gpu_process.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/delete_gpu_process.sh" --kill-orphans "$@"
