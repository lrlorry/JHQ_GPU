#!/bin/bash
# Copy exactly the graphics referenced by the current manuscript; fail if absent.
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
from pathlib import Path
import re,shutil
root=Path.cwd().parent
main=(root/'tex/ADC/mainADC.tex').read_text()
files=[root/'tex'/f'{p}.tex' for p in re.findall(r'\\input\{(ADC/[^}]+)\}',main)]
names=sorted(set(re.findall(r'figs/(\w+\.pdf)','\n'.join(p.read_text() for p in files))))
for name in names:
 src=root/'figures/out'/name
 if not src.is_file():raise FileNotFoundError(src)
 shutil.copy2(src,root/'tex/figs'/name)
print(f'Synced {len(names)} figures')
PY
