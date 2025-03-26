
import difflib


file1 = "wass_data/output/000000_wd/mesh_cam_mio.xyzC"
file2 = "wass_data/output/000000_wd/mesh_cam.xyzC"

with open(file1) as f1, open(file2) as f2:
    diff = list(difflib.unified_diff(f1.readlines(), f2.readlines()))

if diff:
    print("Files have differences:")
    print("".join(diff))
else:
    print("Files are identical")

