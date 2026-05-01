import sys, cv2, numpy as np
sys.path.insert(0, ".")
if "aging" in sys.modules:
    del sys.modules["aging"]
from aging import whiten_hair_approximation

h, w = 480, 360
base = np.zeros((h, w, 3), dtype=np.uint8)
base[:] = (155, 185, 210)

tests = [
    ("Dark brown", (28, 22, 18)),
    ("Light blonde", (200, 195, 185)),
    ("Red", (30, 60, 160)),
]

for name, color in tests:
    img = base.copy()
    cv2.ellipse(img, (180, 30), (160, 80), 0, 0, 360, color, -1)
    out = whiten_hair_approximation(img, intensity=1.0, _debug=False)
    diff = float(np.abs(out.astype(np.int32) - img.astype(np.int32)).mean())
    hair_b = float(img[:80, 20:340].mean())
    hair_a = float(out[:80, 20:340].mean())
    skin_b = float(img[200:350, 80:280].mean())
    skin_a = float(out[200:350, 80:280].mean())
    ok = (hair_a > hair_b + 8) and (abs(skin_a - skin_b) < 20)
    print(name + ": diff=" + str(round(diff,2)) + " hair " + str(round(hair_b)) + "->" + str(round(hair_a)) + " skin_change=" + str(round(abs(skin_a-skin_b),1)) + " " + ("PASS" if ok else "FAIL"))
