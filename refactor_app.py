import sys
from pathlib import Path

def rewrite_app_py():
    with open('app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    skip = False
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Remove _REPLICATE_MODEL and call_replicate_aging completely
        if line.startswith('# Model identifier on Replicate') or line.startswith('_REPLICATE_MODEL ='):
            skip = True
        
        if skip and line.startswith('@app.post("/api/aging-ai")'):
            skip = False

        if skip:
            i += 1
            continue

        # Initialize LocalAgingService right before Flask app instantiation
        if line.startswith('app = Flask(__name__)'):
            new_lines.append('from face_aging.aging_service import LocalAgingService\n')
            new_lines.append('try:\n')
            new_lines.append('    local_aging_service = LocalAgingService()\n')
            new_lines.append('except Exception as e:\n')
            new_lines.append('    print(f"Failed to load LocalAgingService: {e}")\n')
            new_lines.append('    local_aging_service = None\n\n')
            new_lines.append(line)
            i += 1
            continue

        # Rewrite the body of api_aging_ai
        if line.startswith('def api_aging_ai():'):
            new_lines.append(line)
            new_lines.append('    """\n')
            new_lines.append('    Local AI aging endpoint.\n')
            new_lines.append('    """\n')
            new_lines.append('    print("AGING AI ENDPOINT CALLED - LOCAL MODE", flush=True)\n')
            new_lines.append('    started = time.perf_counter()\n\n')
            
            new_lines.append('    if local_aging_service is None:\n')
            new_lines.append('        return _error("Local aging service is not initialized.", 503)\n\n')
            
            new_lines.append('    raw_file = request.files.get("image")\n')
            new_lines.append('    if raw_file is None:\n')
            new_lines.append('        return _error("No image uploaded.", 400)\n\n')
            
            new_lines.append('    image_bytes = raw_file.read()\n')
            new_lines.append('    if not image_bytes:\n')
            new_lines.append('        return _error("Uploaded image is empty.", 400)\n\n')
            
            new_lines.append('    try:\n')
            new_lines.append('        image = _read_image_from_bytes(image_bytes)\n')
            new_lines.append('    except Exception as exc:\n')
            new_lines.append('        return _error(f"Could not decode image: {exc}", 400)\n\n')
            
            new_lines.append('    try:\n')
            new_lines.append('        strength = _clamp(float(request.form.get("strength", 0.5)), 0.0, 1.0)\n')
            new_lines.append('        mode     = (request.form.get("mode", "aging") or "aging").strip().lower()\n')
            new_lines.append('        years    = max(-50, min(50, int(float(request.form.get("years", 10)))))\n')
            new_lines.append('    except ValueError:\n')
            new_lines.append('        return _error("Invalid aging parameters.", 400)\n\n')
            
            new_lines.append('    if mode not in {"aging", "deaging"}:\n')
            new_lines.append('        return _error("Invalid mode. Use \'aging\' or \'deaging\'.", 400)\n\n')

            new_lines.append('    # Local Inference\n')
            new_lines.append('    try:\n')
            new_lines.append('        out = local_aging_service.run_inference(image)\n')
            new_lines.append('    except Exception as exc:\n')
            new_lines.append('        print(f"[api_aging_ai] Inference error: {exc}", flush=True)\n')
            new_lines.append('        return jsonify({"ok": False, "error": str(exc)}), 500\n\n')

            new_lines.append('    # Resize to match input dimensions\n')
            new_lines.append('    if out.shape[:2] != image.shape[:2]:\n')
            new_lines.append('        out = cv2.resize(out, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)\n\n')

            new_lines.append('    try:\n')
            new_lines.append('        encoded = _encode_base64_jpeg(out)\n')
            new_lines.append('    except ValueError as exc:\n')
            new_lines.append('        return _error(str(exc), 500)\n\n')

            new_lines.append('    saved = _save_image(out, "aging-ai-local")\n')
            new_lines.append('    elapsed_ms = (time.perf_counter() - started) * 1000.0\n')
            new_lines.append('    image_url = f"/results/{Path(saved).name}" if saved else None\n\n')
            
            new_lines.append('    apparent_age_before = _estimate_base_age(image)\n')
            new_lines.append('    apparent_age_after  = _apparent_age_after(apparent_age_before, mode, years, strength)\n\n')
            
            new_lines.append('    return jsonify({\n')
            new_lines.append('        "ok":                True,\n')
            new_lines.append('        "image":             encoded,\n')
            new_lines.append('        "image_url":         image_url,\n')
            new_lines.append('        "method":            "ai",\n')
            new_lines.append('        "mode":              mode,\n')
            new_lines.append('        "years":             years,\n')
            new_lines.append('        "modelStatus":       "local-sam-ffhq",\n')
            new_lines.append('        "apparentAgeBefore": apparent_age_before,\n')
            new_lines.append('        "apparentAgeAfter":  apparent_age_after,\n')
            new_lines.append('        "saved":             saved,\n')
            new_lines.append('        "durationMs":        round(elapsed_ms, 2),\n')
            new_lines.append('    })\n\n')
            
            # Now we need to skip the rest of the old api_aging_ai body until the next @app.post
            i += 1
            while i < len(lines) and not lines[i].startswith('@app.post'):
                i += 1
            continue

        new_lines.append(line)
        i += 1

    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print('app.py updated successfully!')

if __name__ == "__main__":
    rewrite_app_py()
