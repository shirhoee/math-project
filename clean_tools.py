import os
import shutil
import glob

os.chdir("v1/tools")
os.makedirs("archive", exist_ok=True)

keep_files = ["cli.py", "make_demo.py", "benchmark.py", "quality_check.py", "__init__.py"]
all_py = glob.glob("*.py")
for f in all_py:
    if f not in keep_files:
        shutil.move(f, os.path.join("archive", f))

print("Tools cleaned!")
