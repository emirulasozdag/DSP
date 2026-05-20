import os
import sys

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
    
    # Google Drive File ID for downloading the model automatically
    gdrive_file_id = "1QTK-O0xoUZavCUCno-YKxIhsbXSGY5ia"
    
    if gdrive_file_id == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        print("\nError: Google Drive File ID (gdrive_file_id) is not set!")
        print("Please follow these steps:")
        print("1. Upload 'models/aging.pt' (2.27 GB) to your own Google Drive / Dropbox / Hugging Face.")
        print("2. Create a public shareable link.")
        print("3. Download it and place it in the 'models/' directory as 'aging.pt'.")
        print("\nAlternatively:")
        print("Put your Google Drive file ID in the 'gdrive_file_id' variable inside this script")
        print("and run 'python download_models.py' to download it automatically.")
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

if __name__ == "__main__":
    download_aging_model()
