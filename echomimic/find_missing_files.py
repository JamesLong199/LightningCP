#!/usr/bin/env python3

import os
import glob

# Directories to compare
dir1 = "result_hdtf/echomimic_sfc_bg_20250510_2226"
dir2 = "result_hdtf/echomimic_sfc_no_estimate_20250510_2117"

# Get list of mp4 files in each directory
files1 = set([os.path.basename(f) for f in glob.glob(f"{dir1}/*.mp4")])
files2 = set([os.path.basename(f) for f in glob.glob(f"{dir2}/*.mp4")])

# Find differences
only_in_dir1 = files1 - files2
only_in_dir2 = files2 - files1

# Print results
print(f"Files only in {dir1}:")
for file in sorted(only_in_dir1):
    print(f"  {file}")

print(f"\nFiles only in {dir2}:")
for file in sorted(only_in_dir2):
    print(f"  {file}")

print(f"\nSummary:")
print(f"  {len(files1)} files in {dir1}")
print(f"  {len(files2)} files in {dir2}")
print(f"  {len(only_in_dir1)} files unique to {dir1}")
print(f"  {len(only_in_dir2)} files unique to {dir2}")