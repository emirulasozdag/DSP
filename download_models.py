import os
import sys
import urllib.request
import bz2
from pathlib import Path

def download_aging_model():
    models_dir = "models"
    model_path = os.path.join(models_dir, "aging.pt")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"[{models_dir}] directory created.")
        
    if os.path.exists(model_path):
        print(f"[✓] {model_path} already exists. Skipping download.")
        return

    print("=" * 60)
    print("Downloading Local AI Aging Model File (aging.pt)...")
    print("Note: This file is approximately 2.27 GB in size.")
    print("=" * 60)
    
    gdrive_file_id = "1QTK-O0xoUZavCUCno-YKxIhsbXSGY5ia"
    
    if gdrive_file_id == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        print("\nError: Google Drive File ID (gdrive_file_id) is not set!")
        return

    try:
        import gdown
        print(f"Starting download from Google Drive (ID: {gdrive_file_id})...")
        gdown.download(id=gdrive_file_id, output=model_path, quiet=False)
        print("\n[✓] Successfully downloaded and saved to: models/aging.pt")
    except ImportError:
        print("\nError: 'gdown' package is not installed!")
        print("Please run 'pip install gdown' or install dependencies using 'requirements.txt'.")
    except Exception as e:
        print(f"\nAn error occurred during download: {e}")
        print("Please download the file manually and place it as 'models/aging.pt'.")

def download_dlib_model():
    models_dir = Path("models")
    dat_path = models_dir / "shape_predictor_68_face_landmarks.dat"
    bz2_path = models_dir / "shape_predictor_68_face_landmarks.dat.bz2"
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    
    if dat_path.exists():
        print(f"[✓] {dat_path} already exists. Skipping dlib download.")
        return

    print("=" * 60)
    print("Downloading dlib 68-Point Face Landmark Model...")
    print("Note: This file is approximately 99 MB (compressed 64 MB).")
    print("=" * 60)
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        def report_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded * 100 / total_size
                print(f"\rDownloading: {percent:.2f}% ({downloaded}/{total_size} bytes)", end="", flush=True)
            else:
                print(f"\rDownloading: {downloaded} bytes", end="", flush=True)

        urllib.request.urlretrieve(model_url, bz2_path, reporthook=report_hook)
        print("\n[✓] dlib model download completed successfully.")
        
        print("Extracting bz2 archive...")
        with bz2.BZ2File(bz2_path) as fr, open(dat_path, "wb") as fw:
            data = fr.read()
            fw.write(data)
        print(f"[✓] Extraction completed. Model saved at: {dat_path}")
        
    except Exception as e:
        print(f"\nError occurred during dlib download/extraction: {e}")
        if dat_path.exists():
            dat_path.unlink()
    finally:
        if bz2_path.exists():
            bz2_path.unlink()

if __name__ == "__main__":
    download_aging_model()
    print()
    download_dlib_model()
