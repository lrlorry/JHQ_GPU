#!/bin/bash
# cuVS for this machine's CUDA. Two earlier mistakes, both mine:
#
#   The Aliyun mirror pinned in pip.conf does not carry NVIDIA packages, so
#   cuda-bindings resolved to "from versions: none". --extra-index-url only
#   appends; the primary index has to be replaced.
#
#   Then cuvs-cu12 was the wrong build. This box runs CUDA 13.0 and torch
#   2.12.1+cu130, and cuvs-cu12 pulls cuda-bindings back to 12.9.x, which
#   contradicts torch's >=13.0.3. cuvs-cu13 and cupy-cuda13x exist and match
#   what is already here -- no virtualenv needed, and nothing else moves.
set -u
source /etc/network_turbo 2>/dev/null
PIP=/root/miniconda3/bin/pip
PY=/root/miniconda3/bin/python3

echo "=== removing the cu12 build installed by mistake ==="
$PIP uninstall -y cuvs-cu12 libcuvs-cu12 pylibraft-cu12 libraft-cu12 \
                  rmm-cu12 librmm-cu12 2>&1 | tail -3

echo "=== installing the cu13 build ==="
$PIP install --timeout 600 --retries 10 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://pypi.nvidia.com \
    cuvs-cu13 cupy-cuda13x 2>&1 | tail -6

echo "=== restoring cuda-bindings for torch ==="
$PIP install --timeout 600 --retries 10 --index-url https://pypi.org/simple \
    "cuda-bindings>=13.0.3,<14" 2>&1 | tail -3

echo "=== import check ==="
$PY -c "import cuvs; print('cuvs', cuvs.__version__)" 2>&1 | tail -1
$PY -c "import cupy; print('cupy', cupy.__version__)" 2>&1 | tail -1
$PY -c "from cuvs.neighbors import cagra, ivf_pq; print('cagra + ivf_pq importable')" 2>&1 | tail -1
$PY -c "import torch; print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())" 2>&1 | tail -1
echo "=========== CUVS INSTALL DONE ==========="
