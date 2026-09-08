#!/bin/bash
# Build the paper's cuVS fork, because the released one does not work here.
#
# cuVS 26.08.01 ships ivf_rabitq's symbols and header but no Python binding,
# and its search returns near-random ids with distances below the true minimum
# -- see results/rabitq/README.md for what that rules out. The artifact the
# paper names is a fork, so build that.
#
# spdlog is told to use its bundled fmt: rapids-cmake turns on
# SPDLOG_USE_STD_FORMAT, which needs libstdc++ 13's <format>, and this box has
# GCC 12.
#
# Only sm_120 and only the C++ library: the wheel carries six architectures
# and a full build of all of them is what makes cuVS take hours. No tests, no
# benchmarks, no Python.
set -u
exec 9>/root/.lock_cuvsbuild; flock -n 9 || { echo "already running"; exit 0; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
ROOT=${ROOT:-/root/autodl-tmp/cuvs_rabitq}
L=${LOG:-/root/cuvsbuild.log}; : > $L
say(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
source /etc/network_turbo 2>/dev/null

say "df before: $(df -h /root/autodl-tmp | tail -1)"
if [ ! -d "$ROOT/.git" ]; then
  say "cloning"
  git clone --branch cuvs_ivf_rabitq --depth 1 \
      https://github.com/Stardust-SJF/cuvs_rabitq.git "$ROOT" >>$L 2>&1
  rc=$?; say "clone_rc=$rc"
  [ $rc -ne 0 ] && { say "=== CUVSBUILD""_DONE clone failed ==="; exit 1; }
fi
cd "$ROOT"
say "HEAD $(git log --oneline -1)"

# rapids-cmake fetches RAFT, CCCL, rmm and friends through CPM. Point its cache
# at the data disk: the system overlay is 30 GB and filled once already today.
export CPM_SOURCE_CACHE=/root/autodl-tmp/cpm_cache
mkdir -p "$CPM_SOURCE_CACHE"

# The fork asks for 3.30.4 and rapids-cmake, which it pulls in, asks for
# 4.0. This box has 3.27.9. pip cannot reach
# PyPI once network_turbo is sourced -- it reports "from versions: none" -- so
# take the binary tarball from GitHub, which is what the proxy is for. Nothing
# system-wide is replaced; everything else here keeps building against 3.27.
CM=$(command -v cmake)
need_new=1
$CM --version | head -1 | awk '{split($3,v,"."); exit !(v[1]>=4)}' && need_new=0
if [ $need_new -eq 1 ]; then
  CMDIR=/root/autodl-tmp/cmake-4.0.3
  if [ ! -x "$CMDIR/bin/cmake" ]; then
    say "cmake $($CM --version | head -1) too old, fetching 4.0.3"
    mkdir -p "$CMDIR"
    curl -fsSL https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-linux-x86_64.tar.gz \
      | tar xz -C "$CMDIR" --strip-components=1 >>$L 2>&1
    rc=$?; say "cmake_fetch_rc=$rc"
    [ $rc -ne 0 ] && { say "=== CUVSBUILD""_DONE cmake fetch failed ==="; exit 1; }
  fi
  CM="$CMDIR/bin/cmake"
  say "using $($CM --version | head -1) at $CM"
fi

# rapids_logger hard-codes SPDLOG_USE_STD_FORMAT ON when it fetches spdlog, so
# passing -DSPDLOG_USE_STD_FORMAT=OFF at the top level does nothing -- the
# define lands on the spdlog target directly. std::format needs libstdc++ 13
# and this box has GCC 12, with no g++-13 in the default repositories. Patch
# the option to OFF in the CPM copy, which puts spdlog back on its bundled fmt
# and changes nothing cuVS relies on.
RL=$(ls -d $CPM_SOURCE_CACHE/rapids_logger/*/ 2>/dev/null | head -1)
if [ -n "${RL:-}" ] && grep -q 'SPDLOG_USE_STD_FORMAT ON' "$RL/CMakeLists.txt"; then
  sed -i 's/SPDLOG_USE_STD_FORMAT ON/SPDLOG_USE_STD_FORMAT OFF/' "$RL/CMakeLists.txt"
  say "patched rapids_logger to leave spdlog on bundled fmt"
fi

say "configuring"
"$CM" -S cpp -B build_sm120 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120-real \
  -DBUILD_TESTS=OFF -DBUILD_C_LIBRARY=OFF -DBUILD_C_TESTS=OFF \
  -DBUILD_CUVS_BENCH=OFF -DBUILD_SHARED_LIBS=ON \
  -DCUVS_COMPILE_LIBRARY=ON \
  -DCMAKE_CUDA_HOST_COMPILER=g++-12 \
  -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 \
  -DSPDLOG_USE_STD_FORMAT=OFF -DSPDLOG_FMT_EXTERNAL=OFF >>$L 2>&1
rc=$?; say "configure_rc=$rc"
[ $rc -ne 0 ] && { grep -iE "error|CMake Error" $L | tail -20 >>$L; say "=== CUVSBUILD""_DONE configure failed ==="; exit 1; }

say "building with $(nproc) cores"
"$CM" --build build_sm120 -j "$(nproc)" >>$L 2>&1
rc=$?; say "build_rc=$rc"
[ $rc -ne 0 ] && { grep -iE " error" $L | tail -25 >>$L; say "=== CUVSBUILD""_DONE build failed ==="; exit 1; }

say "libcuvs.so: $(ls -la build_sm120/libcuvs.so 2>&1 | tail -1)"
say "rabitq symbols: $(nm -D --defined-only build_sm120/libcuvs.so 2>/dev/null | grep -ci rabitq)"
say "df after: $(df -h /root/autodl-tmp | tail -1)"
say "=== CUVSBUILD""_DONE ==="
