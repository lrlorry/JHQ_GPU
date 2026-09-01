#!/bin/bash
# cuVS install. The earlier failures were not bandwidth -- cuvs_cu12 itself is
# 1.5 MB and downloaded fine. This box pins
#   global.index-url = http://mirrors.aliyun.com/pypi/simple
# which does not carry NVIDIA packages, so cuda-bindings>=12.9.2 resolved to
# "from versions: none". --extra-index-url only appends, leaving Aliyun as the
# primary index and the resolution still failing; the primary has to be
# replaced outright.
set -u
source /etc/network_turbo 2>/dev/null
PIP=/root/miniconda3/bin/pip
PY=/root/miniconda3/bin/python3

echo "=== installing cuvs-cu12 (primary index = PyPI, not the Aliyun mirror) ==="
$PIP install --timeout 600 --retries 10 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://pypi.nvidia.com \
    cuvs-cu12 2>&1 | tail -8

echo "=== import check ==="
$PY -c "import cuvs; print('cuvs', cuvs.__version__)" 2>&1 | tail -1
$PY -c "import cupy; print('cupy', cupy.__version__)" 2>&1 | tail -1
$PY -c "from cuvs.neighbors import cagra, ivf_pq; print('cagra + ivf_pq importable')" 2>&1 | tail -1
echo "=========== CUVS INSTALL DONE ==========="
