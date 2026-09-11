#!/bin/bash
# Generate current manuscript assets, sync them, compile, and record hashes.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 build_paper.py
