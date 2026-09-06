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

**The authors' artifact cannot be run. That does not leave the paper without a
CPU baseline.**

`JHQ_repro` (`github.com/lrlorry/JHQ_repro`) is a from-scratch C++17
reimplementation, it builds, and every CPU number in `results/` came from it. It
implements equation 4 the way the paper states it:

```cpp
// Analytical 1D Lloyd-Max codewords (paper Eq. 3 / 4):
//   c_i = sigma*sqrt(2) * erfinv((2i-1)/K_1D)
c1d_[i] = sigma_ * float(M_SQRT2) * erfinv_f(2.0f * q_i - 1.0f);
```

which is the same construction the GPU port uses, and `src/codebook.cpp` shares
its lineage with `cpu/codebook.cpp` here — `train_1d_kmeans` has the same
signature in both. As a baseline it is sound, and it is better documented than
the authors' code because its protocol can be stated.

**What is wrong is only that the recorded numbers are at the old protocol.**
`src/jhq_ivf_index.cpp:96` calls

```cpp
res_c1d_ = train_1d_kmeans(residuals.data(), (int)residuals.size(), Kr_);
```

with no `max_iter`, so it takes `codebook.h`'s default of **25** — the setting
this project measured as costing 6-8e-3 of recall against 2000. There is also a
`max_train_n` cap on the residual training set. So 238.55 s and the recall
beside it describe a 25-iteration codebook, and the GPU rows they would be
compared against describe a 2000-iteration one.

The fix is to re-run `JHQ_repro` with `max_iter` passed at that call site and
the residual training set matched, not to drop the comparison. Then both sides
are equation 4 plus 2000 residual Lloyd iterations and the only difference left
is CPU against GPU.

One more thing has to be right in that re-run: `results/` records that the CPU
timings do not reproduce without pinned threads — 32 threads beat 208 by 1.9x
on this host — so the thread count belongs in the protocol alongside the
iteration count.
