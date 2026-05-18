import requests, json, sys, os

url = "http://localhost:8081/api/aging"
img_path = r"C:\Users\user\Documents\GitHub\DSP\test_face.jpg"
if not os.path.exists(img_path):
    print("test_face.jpg not found")
    sys.exit(1)

with open(img_path, "rb") as f:
    img_bytes = f.read()

print(f"Image size: {len(img_bytes)} bytes")
print("=" * 50)

for i, years in enumerate([10, 24, 35, 50], 1):
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    data = {
        "mode": "aging",
        "method": "freq",
        "years": str(years),
        "strength": "0.5",
        "contrast": "0.5",
        "tone": "0.5",
        "noise": "0.0",
        "detail": "0.5",
        "deterministic": "true",
        "hpf": "gaussian_hpf",
        "lpf": "gaussian_lpf",
    }
    resp = requests.post(url, files=files, data=data, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    base = j.get("apparentAgeBefore")
    sim = j.get("apparentAgeAfter")
    print(f"Test {i}: +{years} years | Base={base} Simulated={sim}")

print("=" * 50)
print("If all Base values are identical, the fix is working.")
