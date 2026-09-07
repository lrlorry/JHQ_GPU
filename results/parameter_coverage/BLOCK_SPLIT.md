# The measurements are split across two block sizes

`JHQ_BLOCK` sets threads per block in the scan kernel:

```cpp
static const int BLOCK = [] {
    const char* e = getenv("JHQ_BLOCK");
    const int   v = e ? atoi(e) : 256;
    return (v == 128 || v == 256 || v == 512 || v == 1024) ? v : 256;
}();
```

Legal values 128, 256, 512, 1024; **default 256**; anything else falls back to
256 silently. The frontier harness passes `--block 1024`. Several standalone
scripts did not set it at all and therefore ran at 256.

## Which result came from which

| at **BLOCK=256** (unset) | at **BLOCK=1024** (explicit) |
|---|---|
| refine stage share, 5.2-6.5% | cascade pricing ladder |
| the four LUT variants: `__half` in shared −65 to −89%, L1 carveout −88 to −113% | the prefix optimum, M/2 on the large sets and M/4 on vogue |
| the scan split: compaction 44% on vogue, 9-10% on bge-m3 and stella; lookup 23/52/42% | `JHQ_BITONIC_SELECT` A/B |
| | the exact-path baseline (`ex.log`) |
| | the layer decomposition (JQ vs JHQ) |
| | **all 63 frontier rows** |

**These two groups cannot be put side by side without saying so.** The scan
split and the cascade pricing have been discussed together in this project's
write-ups; they are at different block sizes.

## Why the block size is not a neutral knob here

It changes two things at once.

**`cap`, and therefore the shared-memory floor.** `cap` is the power of two at
or above `ck + BLOCK`, and `scan_base = cap*8 + 8`. At `ck=1000` every block
size from 256 to 1024 lands on `cap=2048`, so scan_base is 16 KB throughout —
but at the paper's `alpha=8` (`ck=80`), BLOCK=256 gives `cap=512` and 4 KB
while BLOCK=1024 still gives 16 KB. So block size and alpha interact.

**Occupancy, when shared memory is the binding constraint.** An SM has 100 KB.
A block asking 82 KB leaves room for exactly one block whatever its size, so
threads per SM is the block size: 256 gives 17% occupancy against 1536, and
1024 gives 67%. That is the confound in the `__half`-in-shared measurement,
which ran at the value where a shared table costs the most.

## Never swept

`JHQ_BLOCK` has one value in every recorded CSV row (1024) and one value in
every standalone script (either 256 or 1024, per script). **It has never been
compared against itself on the same configuration.**

That is worth fixing before the numbers above are quoted anywhere, because it
is cheap and it decides whether the two groups can be merged: one sweep of
128/256/512/1024 on one dataset, holding everything else, says whether the
choice matters at all.
