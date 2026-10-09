import os
import shutil
import glob

# Remove root untracked files
root_files_to_remove = [
    "dummy.jpg", "temp_resized_app.jpg", "tree.txt", "update.py", "v1_tree.txt", "v2_tree.txt", "test_depth_timing.py"
]
for f in root_files_to_remove:
    if os.path.exists(f):
        os.remove(f)

# Archive setup scripts
os.makedirs("archive", exist_ok=True)
for f in glob.glob("setup_*.py"):
    shutil.move(f, os.path.join("archive", f))

if os.path.exists("cache"):
    shutil.rmtree("cache")
if os.path.exists("temp_cache"):
    shutil.rmtree("temp_cache")

# v1 cleanup
os.chdir("v1")

if os.path.exists("temp_cache"):
    shutil.rmtree("temp_cache")

if os.path.exists("PROJECT_AUDIT"):
    shutil.rmtree("PROJECT_AUDIT")

v1_files_to_remove = glob.glob("temp_*.jpg") + glob.glob("debug_layer*.png") + glob.glob("Dramatic_*.png") + glob.glob("Normal_*.png") + glob.glob("Subtle_*.png") + ["dummy.jpg", "temp_test.jpg", "CHECKSUMS.txt", "ARCHIVE_NOTE.md"]

for f in v1_files_to_remove:
    if os.path.exists(f):
        os.remove(f)

# Move loose scripts to tools or archive
if os.path.exists("diag_fill.py"):
    shutil.move("diag_fill.py", os.path.join("tools", "diag_fill.py"))
if os.path.exists("tune_settings.py"):
    shutil.move("tune_settings.py", os.path.join("tools", "tune_settings.py"))

print("Cleanup complete!")
