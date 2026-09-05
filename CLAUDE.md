# Working rules for this repository

## Every code change is a new version directory

`jhq_vNN_<name>/` directories exist so a change can be compared against what it
replaced. **Never edit an existing one in place.** Copy it first.

```sh
cp -r jhq_v<prev>_<name> jhq_v<next>_<newname>
# the sources hard-code their own include paths; a copy that still includes the
# predecessor's headers compiles against the wrong declarations
sed -i '' 's|jhq_v<prev>_<name>/|jhq_v<next>_<newname>/|g' \
    jhq_v<next>_<newname>/*.cu jhq_v<next>_<newname>/*.cuh
# then add a CMake target for it via add_jhq_dir; leave the old target alone
```

Afterwards `git status --porcelain jhq_v<prev>_<name>` must be empty.

**Highest version: v25** (`jhq_v25_streamed_residual`). The next change is v26.

### Frozen, never to be modified

| directory | tag | what it fixes |
|---|---|---|
| `jhq_v23_faithful/` | `paper-faithful-v1` | the validated search semantics |

Editing it makes the working tree stop matching the tag the paper cites. If a
change is needed there, it is a new version directory instead.

## A version replaces, it does not accommodate

A new version directory carries **one** way of doing each thing. When it
supersedes an approach, the superseded path is deleted, not kept alive beside
it behind a flag.

Keeping both is what produces code that cannot be reasoned about:

- The uneven bit split was added to `cpu/pq_codebook.cpp` so equation 4 could
  be built at any Ds, while the GPU encoders continued to assume one shared
  level array. Both paths existed; only one worked; nothing said which. It
  corrupted the heap the first time the two met, and in the meantime an
  admissibility rule written into three documents described the dead one.
- Every extra flag doubles what a result can mean. `JHQ_RES_HIST`,
  `JHQ_ENCODE_GEMM`, `JHQ_ENCODE_SEPARABLE` and the rest each make a CSV row
  ambiguous unless the row records them — which is why the harness now records
  every `JHQ_*` the child saw.

So: an option earns its place only when both settings are **measured and
reported**, like `JHQ_RESID_LUT`, where fused and materialised were shown equal
to 1.192e-07 and the switch exists to reproduce that. An option that exists
"in case" is dead weight, and dead weight here has been actively misleading.

If an old path is worth keeping for comparison, it keeps its own version
directory. That is what they are for.

## Syncing to the GPU box

The box reaches GitHub only after `source /etc/network_turbo` (AutoDL's proxy).
With it, `git fetch` works and the box can be put at a known commit in one
step. Without it the fetch dies with a GnuTLS error and the temptation is to
ship files by hand.

**Do not ship files by hand.** Push, then on the box:

```sh
source /etc/network_turbo
cd /root/JHQ_GPU
git fetch https://github.com/lrlorry/JHQ_GPU.git <branch> && git reset --hard FETCH_HEAD
```

Run it detached (`setsid nohup ... &`) and read a log: the ssh connection
drops often enough that an interactive fetch gets cut off half way.

Piecemeal shipping is what produced the worst bug of this session. `cpu/` was
never sent while `jhq_v2*/` was, so every test ran against a codebook from
before the uneven-bit-split commit, and a finding about which M are admissible
turned out to be a finding about code that had been superseded. The box must
be at a commit, not at a mixture.

## Two other rules that have cost time here

**The working tree is shared with other Claude sessions.** Stage explicit
paths; `git add -A` commits another session's in-flight work under your message.

**A script that runs after a failed build reports the old binary's output.**
Every build script must abort on a non-zero build:

```sh
rc=$?; echo "build_rc=$rc" >> $LOG
[ $rc -ne 0 ] && { echo "=== DONE build failed, nothing run ===" >> $LOG; exit 1; }
```

Three rounds of debugging in one session were spent on an "unfixed" bug that
was really a build failure whose stale binary kept reporting the same error.
