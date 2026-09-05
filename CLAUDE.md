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
