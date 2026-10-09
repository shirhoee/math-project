import os
import shutil
from pathlib import Path

# Directories
dirs = [
    'v2/.agent/rules',
    'v2/.agent/workflows',
    'v2/docs',
    'v2/src/geometry',
    'v2/src/depth',
    'v2/src/mesh',
    'v2/src/server',
    'v2/web',
    'v2/tests',
    'v2/assets/samples',
    'v2/cache',
    'v2/outputs'
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

# Copy samples
v1_samples = 'v1/assets/samples'
samples_to_copy = ['indoor.jpg', 'landscape.jpg', 'closeup.jpg']
for s in samples_to_copy:
    src = os.path.join(v1_samples, s)
    dst = os.path.join('v2/assets/samples', s)
    if os.path.exists(src):
        shutil.copy2(src, dst)

# Create top-level README
Path('README.md').write_text('# Project\n\nV1 is archived in 1/.\nPlease navigate to 2/ for the current project.\n')

