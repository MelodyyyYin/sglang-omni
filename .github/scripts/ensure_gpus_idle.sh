#!/usr/bin/env bash
# Note (Yin): strong-cleanup shim for the post-server / WER path. Run via `bash`
# (delete_gpu_process.sh has no shebang/+x bit, so a bare exec returns 126), and
# pass --kill-orphans so the disaggregated TP=2 spawn workers are reaped here too.
exec bash "$(dirname "${BASH_SOURCE[0]}")/delete_gpu_process.sh" --kill-orphans "$@"
