"""
Standalone test for whiten_hair_approximation.

Usage:
    python test_hair.py

Expects test_face.jpg in the project folder.
Writes hair_test_result.jpg and debug_mask.jpg next to it.
"""

import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from aging import whiten_hair_approximation

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "test_face.jpg")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "hair_test_result.jpg")

if not os.path.isfile(INPUT_PATH):
    print(f"ERROR: test_face.jpg not found at {INPUT_PATH}")
    print("Please place a portrait photo named test_face.jpg in the project folder and re-run.")
    sys.exit(1)

img = cv2.imread(INPUT_PATH)
if img is None:
    print(f"ERROR: Could not read {INPUT_PATH} (unsupported format or corrupted file).")
    sys.exit(1)

print(f"Loaded image: {img.shape[1]}x{img.shape[0]} px")

# Force intensity=1.0 and enable debug output + mask save
out = whiten_hair_approximation(img, intensity=1.0, _debug=True)

diff = float(np.abs(out.astype(np.int32) - img.astype(np.int32)).mean())
print(f"Mean absolute pixel difference (before vs after): {diff:.2f}")
if diff < 2.0:
    print("WARNING: Difference is very small — hair whitening may not have applied.")
else:
    print("OK: Visible change detected.")

cv2.imwrite(OUTPUT_PATH, out)
print(f"Result saved to: {OUTPUT_PATH}")
print(f"Debug mask saved to: debug_mask.jpg  (white=hair region, black=excluded)")
