# The reference artifact does not build

The CPU baseline in `results/` (238.55 s on stella) comes from `~/JHQ_repro`,
which is not the paper's code, at settings that were never recorded. Replacing
it with the authors' own artifact — `github.com/jiabhan/JHQ`, the repository
the paper's PVLDB availability statement names — was attempted and did not
succeed.

Four defects, each sufficient on its own to stop the build, and then a version
requirement with no solution.

## 1. The FAISS submodule can never be fetched

`.gitmodules` declares `path = external/faiss`, relative to the repository
root. The tree puts the source under `jhq/`, so the path that exists is
`jhq/external/faiss`. `git clone --recurse-submodules`, which the README's
Quick Start prescribes, leaves `jhq/external/faiss/` containing only a
`CMakeLists.txt`. The tree also carries **no gitlink at all** — `git ls-tree -r
HEAD | grep 160000` is empty — so no FAISS commit is pinned anywhere.

## 2. AVX intrinsics without the header that declares them

Four files in `jhqlib/impl` use `__m256` or `__m512`:

```
IndexIVFJHQScanner.cpp   includes <immintrin.h>
IndexJHQIO.cpp           does not
IndexJHQSearch.cpp       does not
IndexJHQTrain.cpp        does not
```

549 errors of the form `'__m256' was not declared in this scope`. No compiler
flag fixes this; `-include immintrin.h` works around it without touching the
sources, which is what was done here.

## 3. No architecture flags anywhere

`grep -E "march|mavx|CMAKE_CXX_FLAGS" jhq/CMakeLists.txt` returns nothing, so
even with the header included the AVX paths compile only if the person building
supplies `-mavx2` or `-march=native` themselves. The README lists prerequisites
and does not mention it.

## 4. The examples do not match the library

`examples/demo_ivfjhq_test.cpp` constructs a `std::unique_ptr<IndexJHQ>` from an
`IndexIVFJHQ*`; the two classes are unrelated in the shipped headers.
`demo_jhq_test.cpp` calls `set_clustering_parameters(bool, int, int)` on an
`IndexJHQ`, whose declared signature takes four arguments. The examples were
written against different headers than the ones published beside them.

## And a FAISS version that does not exist

With 1–3 worked around, the library reaches **2 errors**, both real API drift:

| FAISS | `mapped_io.h`, `index_read_utils.h`, `prefetch.h` | `scan_codes` signature |
|---|---|---|
| v1.7.4 | absent | matches — `(…, float*, idx_t*, size_t)` |
| v1.9.0 | absent | does not match, and two more overrides fail |
| main, 2026-09-04 | present | `ResultHandler&`, does not match |

`jhqlib` needs headers introduced after 1.9.0 and an `InvertedListScanner`
interface retired before them. Nothing pins the version, so it can only be
inferred from the API, and the API points two ways at once.

## What this means for the paper

**There is currently no same-protocol CPU baseline.** The 238.55 s figure
cannot be attributed to the paper's implementation, and the authors' code
cannot be run without editing it. Two options:

1. **Patch their sources** — the include, and whichever `scan_codes` overload a
   chosen FAISS wants — and state exactly what was changed. What is then
   measured is their algorithm through a modified build.
2. **Drop the same-protocol CPU comparison** and say why, citing this. The
   GPU-versus-GPU comparison against cuVS does not depend on it.

Both are defensible; the first is more work and the second is a smaller claim.
The one thing that is not defensible is quoting 238.55 s as the paper's CPU
implementation.

Raw logs: the build attempts are `/root/cpubuild*.log` and `/root/cpu[5-10].log`
on the box; the sequence and their outcomes are recorded above.
